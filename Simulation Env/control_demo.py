import math
import sim as vrep_sim
import time
import numpy as np


class AckermannDemo:
    def __init__(self):
        # --- 参数配置 (对应你公式里的 config) ---
        self.L = 5.44  # 车长 (Wheelbase)
        self.W = 2.57  # 车宽 (Track width)
        self.WHEEL_RADIUS = 0.3  # 轮半径 (用于 v 换算 omega)
        self.ANGLE_THRESHOLD = 0.001

        # 建立连接
        vrep_sim.simxFinish(-1)
        self.client_ID = vrep_sim.simxStart('127.0.0.1', 19997, True, False, 5000, 5)

        if self.client_ID != -1:
            print("Connected to CoppeliaSim")
            vrep_sim.simxStartSimulation(self.client_ID, vrep_sim.simx_opmode_oneshot)
            self._get_handles()
        else:
            raise ConnectionError("Could not connect to Remote API")

    def _get_handles(self):
        # 转向句柄 (Position Control)
        self.steer_h = []
        for i in [10, 20, 30, 40]:  # 10:左前, 20:右前, 30:左后, 40:右后
            _, h = vrep_sim.simxGetObjectHandle(self.client_ID, f'frame_wheel{i}', vrep_sim.simx_opmode_blocking)
            self.steer_h.append(h)

        # 驱动句柄 (Velocity Control)
        self.drive_h = []
        for i in [1, 2, 3, 4]:  # 1:左前, 2:右前, 3:左后, 4:右后
            _, h = vrep_sim.simxGetObjectHandle(self.client_ID, f'frame_wheel{i}', vrep_sim.simx_opmode_blocking)
            self.drive_h.append(h)

    def _ackermann_angles(self, delta: float) -> np.ndarray:
        """根据你提供的公式计算四个轮子的转向角度"""
        if abs(delta) < self.ANGLE_THRESHOLD:
            return np.zeros(4)

        cot_delta = 1.0 / np.tan(delta)
        # 前左前右
        delta_fl = np.arctan(1.0 / (cot_delta - self.W / self.L))
        delta_fr = np.arctan(1.0 / (cot_delta + self.W / self.L))

        # 返回 [左前, 右前, 左后, 右后]
        # 注意：公式里后轮是反向转向 (-delta_fl, -delta_fr)
        return np.array([delta_fl, delta_fr, -delta_fl, -delta_fr])

    def _ackermann_velocities(self, v: float, delta: float) -> np.ndarray:
        """根据你提供的公式计算四个轮子的驱动线速度"""
        if abs(delta) < self.ANGLE_THRESHOLD:
            return np.array([v] * 4)

        angles = self._ackermann_angles(delta)
        tan_delta = np.tan(delta)

        # 基于三角几何补偿线速度
        v_fl = v * tan_delta / np.sin(angles[0])
        v_fr = v * tan_delta / np.sin(angles[1])

        # 对应驱动轮速度分配
        return np.array([v_fl, v_fr, v_fl, v_fr])

    def control(self, target_v, target_delta_deg):
        """执行控制逻辑"""
        delta_rad = math.radians(target_delta_deg)

        # 1. 计算角度和速度
        angles = self._ackermann_angles(delta_rad)
        velocities = self._ackermann_velocities(target_v, delta_rad)

        # 2. 发送转向位置指令
        for i in range(4):
            vrep_sim.simxSetJointTargetPosition(self.client_ID, self.steer_h[i], angles[i],
                                                vrep_sim.simx_opmode_oneshot)

        # 3. 发送驱动角速度指令 (omega = v / r)
        for i in range(4):
            omega = -velocities[i] / self.WHEEL_RADIUS
            vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.drive_h[i], omega, vrep_sim.simx_opmode_oneshot)

    def stop(self):
        vrep_sim.simxStopSimulation(self.client_ID, vrep_sim.simx_opmode_blocking)
        vrep_sim.simxFinish(self.client_ID)


if __name__ == "__main__":
    car = AckermannDemo()
    try:
        # 测试：速度 1.5m/s，转角 25度
        v_test = 1.5
        steer_test = 25.0

        print(f"Running Demo: v={v_test}, steer={steer_test}")
        while True:
            car.control(v_test, steer_test)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")
        car.stop()