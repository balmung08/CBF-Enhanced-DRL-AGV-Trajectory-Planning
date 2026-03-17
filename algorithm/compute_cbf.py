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

"""
CBF-QP 在线安全过滤器 — CasADi 实现
对应论文 Section 3.5 / Eq.24

依赖：pip install casadi numpy

用法：
    from cbf_qp_casadi import CBFQPFilter
    filt = CBFQPFilter(cfg)
    u_safe = filt.filter(state, u_nominal, p_obs)
"""

import math
import numpy as np
import casadi as ca
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
#  二阶 CBF 值计算（与主代码共用，对应论文 Eq.8-13）
# ──────────────────────────────────────────────────────────────────────

def compute_cbf(state: np.ndarray,
                p_obs: np.ndarray,
                L: float, W: float,
                D_safe: float, delta: float
                ) -> Tuple[float, float, float, float, float]:
    """
    计算前边缘CBF值 h, ḣ 及二阶系数 A, B, C。

    参数
    ----
    state  : [x, y, v, θ, η]  — AGV完整状态
    p_obs  : [x_obs, y_obs]   — 最近障碍物点（全局坐标）
    L, W   : 半轴距、半轮距 (m)
    D_safe : 安全距离阈值
    delta  : CBF平滑常数

    返回
    ----
    (h, h_dot, A, B, C)  — 对应 Eq.9, 11, 13
    """
    x, y, v, theta, eta = state
    x_obs, y_obs = p_obs

    ct, st = math.cos(theta), math.sin(theta)

    # 车辆前左角点 p1（Eq.7）
    p1x = x + ct * L - st * W
    p1y = y + st * L + ct * W

    dx = x_obs - p1x
    dy = y_obs - p1y

    # 有向距离 d（Eq.8）
    d = dx * ct + dy * st

    # h = d² - D_safe² - δ（Eq.9）
    h = d * d - D_safe * D_safe - delta

    # 转向曲率 k
    k = (L * math.tan(eta)) / (L * L + W * W)

    # 垂直几何量 r⊥（Eq.11）
    r_perp = -dx * st + dy * ct + L / 2.0

    # ḣ（Eq.11）
    h_dot = -2.0 * d * v + 2.0 * d * v * k * r_perp

    # 二阶系数 A, B, C（Eq.13）
    A = -2.0 * d * (1.0 - k * r_perp)
    se = math.sin(eta)
    B = 0.0 if abs(se) < 1e-6 else (2.0 * d * v * r_perp / (se * math.cos(eta)))
    C = 2.0 * v * v * (
        1.0 - 2.0 * k * r_perp
        + k * k * (r_perp * r_perp - 2.0 * d * (d + W / 2.0))
    )

    return h, h_dot, A, B, C


# ──────────────────────────────────────────────────────────────────────
#  CasADi CBF-QP 安全过滤器
# ──────────────────────────────────────────────────────────────────────


class CBFConfig:

    # CBF 超参数（论文 Section 3.1 & 3.3）
    cbf_c1: float = 10.0  # 二阶CBF系数 c1（对应 ḣ 项）
    cbf_c2: float = 10.0  # 二阶CBF系数 c2（对应 h 项）
    D_safe: float = 1e-2  # 安全距离阈值
    delta: float = 1e-3  # CBF平滑常数 δ
    cbf_violation_eps: float = 0.05  # 最大允许违约率 ε
    lr_lagrange: float = 0.01  # 对偶学习率 α_λ
    L: float = 2.719  # 半轴距（wheelbase/2）m
    W: float = 1.285  # 半轮距（track width/2）m


# Training CBF Calculator
class ControlBarrierFunction:
    """
    二阶CBF用于AGV碰撞避免安全保障
    """

    def __init__(self, cfg: CBFConfig):
        self.D_safe = cfg.D_safe
        self.delta = cfg.delta
        self.c1 = cfg.cbf_c1
        self.c2 = cfg.cbf_c2
        self.L = cfg.L
        self.W = cfg.W

    def get_wheel_positions(self, x: float, y: float, theta: float) -> List[np.ndarray]:
        """
        计算4个车轮全局坐标（Eq.3 & Eq.7）
        车轮索引: 1=左前, 2=右前, 3=左后, 4=右后
        """
        wheels_local = [
            np.array([self.L, self.W]),  # 轮1
            np.array([self.L, -self.W]),  # 轮2
            np.array([-self.L, self.W]),  # 轮3
            np.array([-self.L, -self.W]),  # 轮4
        ]
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        center = np.array([x, y])
        return [center + R @ w for w in wheels_local]

    def compute_h(self, state: np.ndarray, p_obs: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        计算CBF值 h, ḣ 以及系数 A, B, C（用于构建QP约束）
        """
        x, y, v, theta, eta = state
        x_obs, y_obs = p_obs

        # 获取车辆前边缘端点 p1, p2
        wheels = self.get_wheel_positions(x, y, theta)
        p1 = wheels[0]  # 左前轮
        p2 = wheels[1]  # 右前轮（前边缘 p1->p2）

        # 相对位置向量 Δp = p_obs - p1 (Eq.8)
        delta_x = x_obs - p1[0]
        delta_y = y_obs - p1[1]

        # 有向距离 d = Δx·cosθ + Δy·sinθ (Eq.8)
        d = delta_x * math.cos(theta) + delta_y * math.sin(theta)

        # CBF值 h = d² - D_safe² - δ (Eq.9)
        h = d ** 2 - self.D_safe ** 2 - self.delta

        # 转向曲率 k = L·tan(η)/(L²+W²)
        k = (self.L * math.tan(eta)) / (self.L ** 2 + self.W ** 2)

        # 垂直几何量 r⊥ = -Δx·sinθ + Δy·cosθ + L/2 (Eq.11)
        r_perp = -delta_x * math.sin(theta) + delta_y * math.cos(theta) + self.L / 2

        # ḣ = -2dv + 2dv·k·r⊥ (Eq.11)
        h_dot = -2 * d * v + 2 * d * v * k * r_perp

        # 二阶系数 A, B, C (Eq.13)
        A = -2 * d * (1 - k * r_perp)
        # sec²(η)/tan(η) = 1/(sin(η)cos(η))，在η≈0时做数值保护
        if abs(math.sin(eta)) < 1e-6:
            B = 0.0
        else:
            B = 2 * d * v * r_perp / (math.sin(eta) * math.cos(eta))
        C = 2 * v ** 2 * (1 - 2 * k * r_perp + k ** 2 * ((r_perp) ** 2 - 2 * d * (d + self.W / 2)))

        return h, h_dot, np.array([A, B, C])

    def violation_degree(self, state: np.ndarray, action: np.ndarray,
                         p_obs: Optional[np.ndarray]) -> float:
        """
        CBF违约度 ξ(s,a) = max(0, -(ḧ + c1·ḣ + c2·h)) (Eq.20)
        """
        if p_obs is None:
            return 0.0
        h, h_dot, (A, B, C) = self.compute_h(state, p_obs)
        a, omega = action
        h_ddot = A * a + B * omega + C
        cbf_val = h_ddot + self.c1 * h_dot + self.c2 * h
        return max(0.0, -cbf_val)



# Action Safety Filter

class CBFQPFilter:

    def __init__(
        self,
        L: float = 2.719,
        W: float = 1.285,
        c1: float = 10.0,
        c2: float = 10.0,
        D_safe: float = 0.01,
        delta: float = 1e-3,
        a_min: float = -2.0,
        a_max: float = 2.0,
        eta_rate_min: float = -math.radians(180),
        eta_rate_max: float = math.radians(180),
    ):
        self.L, self.W = L, W
        self.c1, self.c2 = c1, c2
        self.D_safe, self.delta = D_safe, delta
        self.lbx = np.array([a_min, eta_rate_min])
        self.ubx = np.array([a_max, eta_rate_max])

        self._solver = self._build_solver()

    def _build_solver(self) -> ca.Function:
        """
        构造参数化 QP：
        """
        u = ca.SX.sym("u", 2)   # [a*, ω*]
        p = ca.SX.sym("p", 5)   # [a_nom, ω_nom, A, B, rhs]


        obj = (u[0] - p[0]) ** 2 + (u[1] - p[1]) ** 2
        g = p[2] * u[0] + p[3] * u[1]
        qp = {"x": u, "p": p, "f": obj, "g": g}

        opts = {
            "print_time": False,
            "error_on_fail": False,
            "osqp": {
                "verbose": False,
                "warm_starting": True,
                "eps_abs": 1e-6,
                "eps_rel": 1e-6,
                "max_iter": 2000,
            },
        }
        solver = ca.qpsol("cbf_qp", "osqp", qp, opts)
        return solver

    # ── 投影法兜底 ───
    def _fallback_projection(
        self,
        a_nom: float, w_nom: float,
        A: float, B: float, rhs: float,
    ) -> np.ndarray:
        """
        最小范数投影：将 (a_nom, w_nom) 投影到约束超平面 A·a + B·ω = rhs 上。
        """
        ab2 = A * A + B * B
        if ab2 < 1e-10:
            return np.array([a_nom, w_nom])
        lhs = A * a_nom + B * w_nom
        scale = (rhs - lhs) / ab2
        return np.clip(
            [a_nom + scale * A, w_nom + scale * B],
            self.lbx, self.ubx,
        )

    def filter(self, state: np.ndarray, u_nominal: np.ndarray, p_obs: Optional[np.ndarray]) -> np.ndarray:
        """
        在线QP安全过滤器。
        """
        if p_obs is None:
            return u_nominal

        h, h_dot, A, B, C = compute_cbf(
            state, p_obs, self.L, self.W, self.D_safe, self.delta
        )


        rhs = -C - self.c1 * h_dot - self.c2 * h

        a_nom, w_nom = float(u_nominal[0]), float(u_nominal[1])

        if A * a_nom + B * w_nom >= rhs:
            return u_nominal

        p_val = np.array([a_nom, w_nom, A, B, rhs], dtype=float)

        sol = self._solver(
            x0=np.clip(u_nominal, self.lbx, self.ubx),  # 暖启动
            p=p_val,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=np.array([rhs]),
            ubg=np.array([1e10]),
        )

        stats = self._solver.stats()
        if not stats.get("success", True):
            # QP 求解失败 → 投影法兜底
            return self._fallback_projection(a_nom, w_nom, A, B, rhs)

        u_star = np.array(sol["x"]).flatten()
        return np.clip(u_star, self.lbx, self.ubx)
