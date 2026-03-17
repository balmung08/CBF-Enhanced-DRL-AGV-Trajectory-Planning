"""
CBF-PPO: Control Barrier Function Enhanced Proximal Policy Optimization
Based on: "Trajectory Planning for 4WIS Autonomous Ground Vehicles in Ro/Ro Terminals Using CBF-enhanced Deep Reinforcement Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.optim import Adam
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings
from algorithm.compute_cbf import ControlBarrierFunction
warnings.filterwarnings('ignore')


# ============================================================
# 1. 超参数配置
# ============================================================
@dataclass
class CBFPPOConfig:
    # PPO 超参数
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    mini_batch_size: int = 256
    max_episode_steps: int = 1024

    # CBF 超参数（论文 Section 3.1 & 3.3）
    cbf_c1: float = 10.0  # 二阶CBF系数 c1（对应 ḣ 项）
    cbf_c2: float = 10.0  # 二阶CBF系数 c2（对应 h 项）
    D_safe: float = 1e-2  # 安全距离阈值
    delta: float = 1e-3  # CBF平滑常数 δ
    cbf_violation_eps: float = 0.05  # 最大允许违约率 ε
    lr_lagrange: float = 0.01  # 对偶学习率 α_λ

    # 状态/动作空间
    lidar_dim: int = 180  # 2D LiDAR距离序列维度
    robot_state_dim: int = 3  # {η, v, t}
    goal_dim: int = 3  # {x_tar^r, y_tar^r, θ_tar^r}
    obs_dim: int = lidar_dim + robot_state_dim + goal_dim  # 总观测维度
    action_dim: int = 2  # {a, Δη}

    # 网络结构
    cnn_channels: int = 64
    cnn_kernel: int = 5
    lstm_hidden: int = 256
    attn_heads: int = 4
    fc_hidden: int = 128

    # 动作约束（4WIS AGV 物理参数）
    a_max: float = 2.0  # 最大加速度 m/s²
    a_min: float = -2.0
    eta_rate_max: float = math.radians(180)  # 最大转向角速率 rad/s
    eta_rate_min: float = -math.radians(180)
    v_max: float = 5.0
    eta_max: float = math.radians(45)
    L: float = 2.719  # 半轴距（wheelbase/2）m
    W: float = 1.285  # 半轮距（track width/2）m



class FeatureExtractor(nn.Module):


    def __init__(self, cfg: CBFPPOConfig):
        super().__init__()
        self.lidar_dim = cfg.lidar_dim
        self.robot_state_dim = cfg.robot_state_dim
        self.goal_dim = cfg.goal_dim

        # LiDAR 1D-CNN 分支
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cfg.cnn_channels, kernel_size=cfg.cnn_kernel, padding=cfg.cnn_kernel // 2),
            nn.ReLU(),
            nn.Conv1d(cfg.cnn_channels, cfg.cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        cnn_out_dim = cfg.cnn_channels * cfg.lidar_dim

        # LSTM（处理CNN输出序列）
        self.lstm = nn.LSTM(
            input_size=cfg.cnn_channels,
            hidden_size=cfg.lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        # Multi-head Attention（对LSTM全序列输出施加注意力）
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.lstm_hidden,
            num_heads=cfg.attn_heads,
            batch_first=True,
            dropout=0.0
        )

        # 状态 & 目标编码
        state_goal_in = cfg.robot_state_dim + cfg.goal_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_goal_in, cfg.fc_hidden),
            nn.ReLU(),
        )

        # 融合层
        self.fuse = nn.Sequential(
            nn.Linear(cfg.lstm_hidden + cfg.fc_hidden, cfg.fc_hidden * 2),
            nn.ReLU(),
        )
        self.output_dim = cfg.fc_hidden * 2

    def forward(self, obs: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple]:
        """
        obs: [B, obs_dim] = [B, 180+3+3]
        """
        B = obs.size(0)
        lidar = obs[:, :self.lidar_dim]  # [B, 180]
        robot_state = obs[:, self.lidar_dim: self.lidar_dim + self.robot_state_dim]
        goal = obs[:, self.lidar_dim + self.robot_state_dim:]

        # --- LiDAR 处理 ---
        lidar_in = lidar.unsqueeze(1)  # [B, 1, 180]
        cnn_out = self.cnn(lidar_in)  # [B, C, 180]
        cnn_seq = cnn_out.permute(0, 2, 1)  # [B, 180, C] → LSTM时序输入

        lstm_out, new_hidden = self.lstm(cnn_seq, hidden)  # [B, 180, H]

        # Multi-head Self-Attention（保留完整时序信息）
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # [B, 180, H]
        # 全局平均池化作为序列表示
        lidar_feat = attn_out.mean(dim=1)  # [B, H]

        # --- 状态+目标 ---
        sg = torch.cat([robot_state, goal], dim=-1)
        sg_feat = self.state_fc(sg)  # [B, fc_hidden]

        # --- 融合 ---
        fused = self.fuse(torch.cat([lidar_feat, sg_feat], dim=-1))  # [B, fc_hidden*2]
        return fused, new_hidden



class ActorNetwork(nn.Module):
    """
    策略网络：输出动作均值和对数标准差
    """
    def __init__(self, cfg: CBFPPOConfig):
        super().__init__()
        self.feature_extractor = FeatureExtractor(cfg)
        feat_dim = self.feature_extractor.output_dim

        self.mean_head = nn.Sequential(
            nn.Linear(feat_dim, cfg.fc_hidden),
            nn.Tanh(),
            nn.Linear(cfg.fc_hidden, cfg.action_dim),
            nn.Tanh()  # 归一化到[-1,1]，再scale到实际范围
        )
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim) - 0.5)

    def forward(self, obs: torch.Tensor,
                hidden=None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        feat, new_hidden = self.feature_extractor(obs, hidden)
        mean = self.mean_head(feat)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, new_hidden

    def get_distribution(self, obs: torch.Tensor, hidden=None):
        mean, std, new_hidden = self.forward(obs, hidden)
        dist = Normal(mean, std)
        return dist, new_hidden

    def get_action(self, obs: torch.Tensor, hidden=None, deterministic=False):
        dist, new_hidden = self.get_distribution(obs, hidden)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, new_hidden


class CriticNetwork(nn.Module):
    """
    价值网络：估计状态价值 V(s)
    """
    def __init__(self, cfg: CBFPPOConfig):
        super().__init__()
        self.feature_extractor = FeatureExtractor(cfg)
        feat_dim = self.feature_extractor.output_dim
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, cfg.fc_hidden),
            nn.Tanh(),
            nn.Linear(cfg.fc_hidden, 1)
        )

    def forward(self, obs: torch.Tensor, hidden=None):
        feat, new_hidden = self.feature_extractor(obs, hidden)
        value = self.value_head(feat)
        return value.squeeze(-1), new_hidden


# ============================================================
# 5. 经验回放缓冲区（存储轨迹 + CBF值）
# ============================================================
class ReplayBuffer:
    """
    存储MDP转移数据及对应CBF值
    D = {(s_t, a_t, r_t, s_{t+1})^N_{t=1}, {h_t, ḣ_t, ḧ_t}^N_{t=1}}
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_list = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        # CBF相关
        self.cbf_h = []
        self.cbf_h_dot = []
        self.cbf_violations = []  # ξ(s,a)

    def add(self, obs, action, reward, log_prob, value, done,
            cbf_h=0.0, cbf_h_dot=0.0, cbf_violation=0.0):
        self.obs_list.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.cbf_h.append(cbf_h)
        self.cbf_h_dot.append(cbf_h_dot)
        self.cbf_violations.append(cbf_violation)

    def size(self):
        return len(self.obs_list)

    def to_tensors(self, device):
        return {
            'obs': torch.FloatTensor(np.array(self.obs_list)).to(device),
            'actions': torch.FloatTensor(np.array(self.actions)).to(device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(device),
            'values': torch.FloatTensor(np.array(self.values)).to(device),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'cbf_violations': torch.FloatTensor(np.array(self.cbf_violations)).to(device),
        }


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float, gae_lambda: float,
                last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation (GAE)
    Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l},  δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    n = len(rewards)
    advantages = np.zeros(n)
    gae = 0.0
    next_value = last_value

    for t in reversed(range(n)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


class CBFPPO:


    def __init__(self, cfg: CBFPPOConfig, device: str = 'cpu'):
        self.cfg = cfg
        self.device = torch.device(device)

        # 网络
        self.actor = ActorNetwork(cfg).to(self.device)
        self.critic = CriticNetwork(cfg).to(self.device)

        # 优化器
        self.actor_optim = Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr_critic)

        # CBF & 安全过滤器
        self.cbf = ControlBarrierFunction(cfg)

        # 拉格朗日乘子 λ（Eq.23，初始化为0）
        self.lagrange_multiplier = 0.0

        # 缓冲区
        self.buffer = ReplayBuffer()

        # 训练统计
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'cbf_loss': [],
            'lagrange': [],
            'J_CBF': [],
            'mean_reward': [],
            'collision_rate': [],
        }

    def select_action(self, obs: np.ndarray,
                      state: Optional[np.ndarray] = None,
                      p_obs: Optional[np.ndarray] = None,
                      deterministic: bool = False,
                      apply_filter: bool = False
                      ) -> Tuple[np.ndarray, float, float]:

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, log_prob_t, _ = self.actor.get_action(obs_t, deterministic=deterministic)
            value_t, _ = self.critic(obs_t)

        action = action_t.cpu().numpy()[0]
        log_prob = log_prob_t.cpu().item()
        value = value_t.cpu().item()

        # 将归一化动作映射到物理范围
        action_phys = self._scale_action(action)

        return action_phys, log_prob, value

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        将[-1,1]归一化动作映射到物理范围
        """
        a = action[0] * self.cfg.a_max
        d_eta = action[1] * self.cfg.eta_rate_max
        return np.array([a, d_eta])

    def compute_cbf_values(self, state: np.ndarray, action: np.ndarray,
                           p_obs: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """
        计算CBF值 h, ḣ 和违约度 ξ（Algorithm 1, Line 5）
        """
        if p_obs is None:
            return 0.0, 0.0, 0.0
        h, h_dot, _ = self.cbf.compute_h(state, p_obs)
        xi = self.cbf.violation_degree(state, action, p_obs)
        return h, h_dot, xi

    def store_transition(self, obs, action, reward, log_prob, value, done,
                         state=None, action_phys=None, p_obs=None):
        """
        存储转移数据（含CBF值）
        """
        cbf_h, cbf_h_dot, cbf_xi = 0.0, 0.0, 0.0
        if state is not None and action_phys is not None:
            cbf_h, cbf_h_dot, cbf_xi = self.compute_cbf_values(state, action_phys, p_obs)

        self.buffer.add(obs, action, reward, log_prob, value, done,
                        cbf_h, cbf_h_dot, cbf_xi)

    def update(self, last_obs: Optional[np.ndarray] = None) -> dict:
        """
        执行CBF-PPO参数更新
        """
        # 获取最终状态的值估计（用于GAE）
        last_value = 0.0
        if last_obs is not None:
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value, _ = self.critic(obs_t)
            last_value = last_value.cpu().item()

        data = self.buffer.to_tensors(self.device)

        # 计算GAE优势
        advantages, returns = compute_gae(
            data['rewards'], data['values'].cpu().numpy(),
            data['dones'], self.cfg.gamma, self.cfg.gae_lambda, last_value
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = data['obs'].size(0)
        indices = np.arange(N)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_cbf_loss = 0.0

        # --- 内层训练循环（Epoch × Mini-batch）---
        for epoch in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.cfg.mini_batch_size):
                mb_idx = indices[start: start + self.cfg.mini_batch_size]
                mb_idx_t = torch.LongTensor(mb_idx).to(self.device)

                mb_obs = data['obs'][mb_idx_t]
                mb_actions = data['actions'][mb_idx_t]
                mb_old_log_probs = data['log_probs'][mb_idx_t]
                mb_advantages = advantages[mb_idx_t]
                mb_returns = returns_t[mb_idx_t]
                mb_cbf_violations = data['cbf_violations'][mb_idx_t]

                # --- PPO目标（Eq.22）---
                dist, _ = self.actor.get_distribution(mb_obs)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()  # r_t(θ)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon,
                                    1 + self.cfg.clip_epsilon) * mb_advantages
                L_ppo = torch.min(surr1, surr2).mean() + self.cfg.entropy_coef * entropy

                # --- CBF损失 ---
                # L_CBF = E[log π_θ(a|s) · ξ_t]
                L_cbf = (new_log_probs * mb_cbf_violations).mean()

                # --- 总损失 ---
                # L = -L_PPO + λ·L_CBF
                total_loss = -L_ppo + self.lagrange_multiplier * L_cbf

                # --- Actor更新 ---
                self.actor_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()

                # --- Critic更新 ---
                values_pred, _ = self.critic(mb_obs)
                L_cri = F.mse_loss(values_pred, mb_returns)
                self.critic_optim.zero_grad()
                L_cri.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optim.step()

                total_policy_loss += L_ppo.item()
                total_value_loss += L_cri.item()
                total_cbf_loss += L_cbf.item()

        n_updates = self.cfg.n_epochs * math.ceil(N / self.cfg.mini_batch_size)

        # --- 拉格朗日乘子更新 --- J_CBF = (1/|D|) Σ ξ_t
        J_CBF = data['cbf_violations'].mean().item()
        # λ ← max(0, λ + α_λ · J_CBF)
        self.lagrange_multiplier = max(0.0, self.lagrange_multiplier + self.cfg.lr_lagrange * J_CBF)

        # 记录统计
        step_stats = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'cbf_loss': total_cbf_loss / n_updates,
            'lagrange': self.lagrange_multiplier,
            'J_CBF': J_CBF,
            'mean_reward': float(np.mean(data['rewards'])),
        }
        for k, v in step_stats.items():
            if k in self.stats:
                self.stats[k].append(v)

        # 清空缓冲区
        self.buffer.reset()
        return step_stats

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'lagrange': self.lagrange_multiplier,
            'stats': self.stats,
        }, path)
        print(f"[CBF-PPO] 模型保存至: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.lagrange_multiplier = ckpt.get('lagrange', 0.0)
        self.stats = ckpt.get('stats', self.stats)
        print(f"[CBF-PPO] 模型加载自: {path}")

