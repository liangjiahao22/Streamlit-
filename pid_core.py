"""
pid_core.py
智能PID恒温混水装置 - 核心算法模块

本模块包含：
1. SystemConfig        - 系统物理参数配置
2. RealisticMixer      - 真实混水物理模型（阀门惯性+传输延迟）
3. PIDController       - 增强型PID控制器（积分限幅、微分先行、抗饱和）
4. AutoTuner           - PID参数自动整定（Ziegler-Nichols）
5. PerformanceMetrics  - 系统性能指标计算

作者：梁嘉豪
日期：2026-09
版本：V1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ============================================================
# 1. 系统物理参数配置
# ============================================================
@dataclass
class SystemConfig:
    """混水系统物理参数配置"""
    hot_temp: float = 60.0  # 热水温度 ℃
    cold_temp: float = 15.0  # 冷水温度 ℃
    valve_response_time: float = 2.0  # 阀门从全关到全开的时间 秒
    transport_delay: float = 1.0  # 水流从阀门到传感器的延迟 秒
    dt: float = 0.1  # 仿真步长 秒
    setpoint: float = 40.0  # 目标温度 ℃


# ============================================================
# 2. 真实混水物理模型
# ============================================================
class RealisticMixer:
    """
    真实混水物理模型

    包含两个非理想环节：
    - 阀门一阶惯性响应：阀门开度以有限速率变化
    - 水流传输纯延迟：传感器读到的温度是延迟前的水温

    数学模型：
        T_mix = α * T_hot + (1 - α) * T_cold
    其中 α 为热水流量占比（阀门开度）
    """

    def __init__(self, config: SystemConfig = None):
        if config is None:
            config = SystemConfig()
        self.config = config
        self.hot_temp = config.hot_temp
        self.cold_temp = config.cold_temp
        self.valve_response_time = config.valve_response_time
        self.transport_delay = config.transport_delay
        self.dt = config.dt
        self.initial_hot_temp = config.hot_temp

        self.target_ratio = 0.3
        self.actual_ratio = 0.3
        self.temp_history: List[float] = []
        self.delay_steps = int(self.transport_delay / self.dt)

        # 用于记录系统扰动
        self.disturbance_log: List[Dict] = []

    def update(self, new_target_ratio: float) -> Tuple[float, float]:
        """
        更新一个仿真步长

        参数：
            new_target_ratio: 目标阀门开度（0-1）

        返回：
            sensor_temp: 传感器读取的温度
            actual_ratio: 实际阀门开度
        """
        # 限制目标开度范围
        self.target_ratio = float(np.clip(new_target_ratio, 0.0, 1.0))

        # 模拟阀门惯性响应：实际开度以最大速率向目标开度靠近
        max_change = (1.0 / self.valve_response_time) * self.dt
        if self.actual_ratio < self.target_ratio:
            self.actual_ratio = min(self.actual_ratio + max_change, self.target_ratio)
        elif self.actual_ratio > self.target_ratio:
            self.actual_ratio = max(self.actual_ratio - max_change, self.target_ratio)

        # 计算阀门出口水温（热平衡方程）
        valve_out_temp = (self.actual_ratio * self.hot_temp +
                          (1 - self.actual_ratio) * self.cold_temp)

        # 模拟水流传输延迟
        self.temp_history.append(valve_out_temp)
        if len(self.temp_history) > self.delay_steps:
            sensor_temp = self.temp_history.pop(0)
        else:
            sensor_temp = self.temp_history[0]

        return sensor_temp, self.actual_ratio

    def apply_disturbance(self, strength: float, disturbance_type: str = "hot_up"):
        """
        施加系统扰动

        参数：
            strength: 扰动强度
            disturbance_type: 扰动类型
                - "hot_up": 热水温度上升（模拟冷水压力下降）
                - "hot_down": 热水温度下降（模拟热水供应不足）
                - "flow_change": 流量变化
        """
        if disturbance_type == "hot_up":
            self.hot_temp += strength
        elif disturbance_type == "hot_down":
            self.hot_temp -= strength
        elif disturbance_type == "flow_change":
            self.hot_temp += strength * 0.5
            self.cold_temp -= strength * 0.3

        self.disturbance_log.append({
            "type": disturbance_type,
            "strength": strength,
            "hot_temp_after": self.hot_temp
        })

    def reset(self):
        """重置模型到初始状态"""
        self.hot_temp = self.initial_hot_temp
        self.target_ratio = 0.3
        self.actual_ratio = 0.3
        self.temp_history = []
        self.disturbance_log = []


# ============================================================
# 3. 增强型PID控制器
# ============================================================
class PIDController:
    """
    增强型增量式PID控制器

    特性：
    - 积分限幅：防止积分饱和
    - 微分先行：避免设定值突变引起的微分冲击
    - 抗饱和开关：可动态开启/关闭积分限幅
    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 setpoint: float = 40.0, dt: float = 0.1,
                 integral_max: float = 2.0, integral_min: float = -2.0,
                 anti_windup: bool = True, derivative_first: bool = True):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt
        self.integral_max = integral_max
        self.integral_min = integral_min
        self.anti_windup = anti_windup
        self.derivative_first = derivative_first

        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.integral = 0.0

        # 记录各环节输出（用于可视化）
        self.history_P: List[float] = []
        self.history_I: List[float] = []
        self.history_D: List[float] = []

    def update(self, measurement: float) -> float:
        """
        执行一步PID计算

        参数：
            measurement: 当前测量值（温度）

        返回：
            output: 控制输出
        """
        error = self.setpoint - measurement

        # ---- 比例环节 ----
        P = self.Kp * error

        # ---- 积分环节（带限幅） ----
        if self.anti_windup:
            self.integral += error * self.dt
            self.integral = float(np.clip(self.integral,
                                          self.integral_min,
                                          self.integral_max))
        else:
            self.integral += error * self.dt
        I = self.Ki * self.integral

        # ---- 微分环节 ----
        if self.derivative_first:
            # 微分先行：对测量值求导，避免设定值突变引起冲击
            derivative = -(measurement - self.prev_measurement) / self.dt
        else:
            # 传统微分：对误差求导
            derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative

        # ---- 求和 ----
        output = P + I + D

        # 更新历史
        self.prev_error = error
        self.prev_measurement = measurement
        self.history_P.append(P)
        self.history_I.append(I)
        self.history_D.append(D)

        return output

    def reset(self):
        """重置控制器状态"""
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.integral = 0.0
        self.history_P = []
        self.history_I = []
        self.history_D = []


# ============================================================
# 4. PID参数自动整定器
# ============================================================
class AutoTuner:
    """
    PID参数自动整定器

    支持两种方法：
    1. Ziegler-Nichols 临界比例度法
    2. Cohen-Coon 开环响应法
    """

    @staticmethod
    def ziegler_nichols(Ku: float, Tu: float) -> Dict[str, float]:
        """
        Ziegler-Nichols 临界比例度法

        参数：
            Ku: 临界增益
            Tu: 临界振荡周期

        返回：
            {"Kp": ..., "Ki": ..., "Kd": ...}
        """
        Kp = 0.6 * Ku
        Ti = 0.5 * Tu
        Td = 0.125 * Tu

        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td

        return {"Kp": round(Kp, 4), "Ki": round(Ki, 4), "Kd": round(Kd, 4)}

    @staticmethod
    def cohen_coon(K: float, tau: float, theta: float) -> Dict[str, float]:
        """
        Cohen-Coon 开环响应法

        参数：
            K: 过程增益
            tau: 时间常数
            theta: 纯延迟时间

        返回：
            {"Kp": ..., "Ki": ..., "Kd": ...}
        """
        if tau <= 0 or theta <= 0 or K <= 0:
            return {"Kp": 0.25, "Ki": 0.03, "Kd": 0.02}

        r = theta / tau
        Kp = (1.0 / K) * (tau / theta) * (1.35 + 0.27 * r)
        Ti = theta * (2.5 - 2.0 * r) / (1.0 - 0.39 * r)
        Td = theta * 0.37 / (1.0 - 0.81 * r)

        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td

        return {"Kp": round(Kp, 4), "Ki": round(Ki, 4), "Kd": round(Kd, 4)}

    @staticmethod
    def auto_search(setpoint: float, config: SystemConfig,
                    kp_range: Tuple[float, float] = (0.05, 0.6),
                    ki_range: Tuple[float, float] = (0.0, 0.08),
                    kd_range: Tuple[float, float] = (0.0, 0.08),
                    n_trials: int = 8) -> List[Dict]:
        """
        自动参数搜索：在多组参数下运行仿真，返回性能指标列表

        参数：
            setpoint: 目标温度
            config: 系统物理配置
            kp_range, ki_range, kd_range: 参数搜索范围
            n_trials: 搜索次数

        返回：
            性能指标列表
        """
        results = []
        np.random.seed(42)

        for _ in range(n_trials):
            kp = np.random.uniform(*kp_range)
            ki = np.random.uniform(*ki_range)
            kd = np.random.uniform(*kd_range)

            metrics = PerformanceMetrics.run_simulation(
                config=config, setpoint=setpoint,
                Kp=kp, Ki=ki, Kd=kd,
                sim_time=80.0
            )

            results.append({
                "Kp": round(kp, 4), "Ki": round(ki, 4), "Kd": round(kd, 4),
                "overshoot": metrics["overshoot"],
                "settling_time": metrics["settling_time"],
                "steady_error": metrics["steady_error"],
                "score": metrics["score"]
            })

        results.sort(key=lambda x: x["score"])
        return results


# ============================================================
# 5. 性能指标计算
# ============================================================
class PerformanceMetrics:
    """系统性能指标计算与仿真运行"""

    @staticmethod
    def run_simulation(config: SystemConfig, setpoint: float,
                       Kp: float, Ki: float, Kd: float,
                       sim_time: float = 100.0,
                       disturbance_time: float = 40.0,
                       disturbance_strength: float = 5.0,
                       disturbance_type: str = "hot_up",
                       anti_windup: bool = True,
                       derivative_first: bool = True) -> Dict:
        """
        运行一次完整仿真，返回温度曲线、阀门开度和性能指标
        """
        dt = config.dt
        n_steps = int(sim_time / dt)

        mixer = RealisticMixer(config)
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=setpoint,
                            dt=dt, anti_windup=anti_windup,
                            derivative_first=derivative_first)

        t_data = []
        temp_data = []
        ratio_data = []
        p_data = []
        i_data = []
        d_data = []

        current_target_ratio = 0.3
        disturbance_applied = False

        for i in range(n_steps):
            current_time = i * dt
            t_data.append(current_time)

            # 更新物理模型
            current_temp, actual_ratio = mixer.update(current_target_ratio)
            temp_data.append(current_temp)
            ratio_data.append(actual_ratio)

            # 施加扰动
            if current_time >= disturbance_time and not disturbance_applied:
                mixer.apply_disturbance(disturbance_strength, disturbance_type)
                disturbance_applied = True

            # PID控制
            control_signal = pid.update(current_temp)
            current_target_ratio += control_signal * 0.005
            current_target_ratio = float(np.clip(current_target_ratio, 0.0, 1.0))

            p_data.append(pid.history_P[-1])
            i_data.append(pid.history_I[-1])
            d_data.append(pid.history_D[-1])

        # 计算性能指标
        metrics = PerformanceMetrics.compute(
            t_data, temp_data, setpoint, disturbance_time
        )

        return {
            "t": t_data, "temps": temp_data, "ratios": ratio_data,
            "P": p_data, "I": i_data, "D": d_data,
            "setpoint": setpoint, "disturbance_time": disturbance_time,
            **metrics
        }

    @staticmethod
    def compute(t: List[float], temps: List[float],
                setpoint: float, disturbance_time: float) -> Dict:
        """
        计算性能指标：
        - overshoot: 最大超调量 (%)
        - settling_time: 调节时间 (±1℃误差带, 秒)
        - steady_error: 稳态误差 (℃)
        - recovery_time: 扰动后恢复时间 (秒)
        - score: 综合评分（越小越好）
        """
        temps_arr = np.array(temps)
        t_arr = np.array(t)

        # 扰动前的数据（用于计算超调量和调节时间）
        pre_disturb = temps_arr[t_arr < disturbance_time]
        pre_t = t_arr[t_arr < disturbance_time]

        if len(pre_disturb) == 0:
            pre_disturb = temps_arr
            pre_t = t_arr

        # 1. 最大超调量
        max_temp = np.max(pre_disturb)
        overshoot = max((max_temp - setpoint) / setpoint * 100, 0.0)

        # 2. 调节时间（进入±1℃误差带并保持稳定）
        settling_time = 0.0
        for i in range(len(pre_disturb)):
            if np.all(np.abs(pre_disturb[i:] - setpoint) <= 1.0):
                settling_time = pre_t[i]
                break
        if settling_time == 0.0:
            settling_time = pre_t[-1] if len(pre_t) > 0 else 0.0

        # 3. 稳态误差（最后10秒的平均绝对误差）
        n_steady = min(100, len(temps_arr))
        steady_error = float(np.mean(np.abs(temps_arr[-n_steady:] - setpoint)))

        # 4. 扰动后恢复时间
        post_disturb = temps_arr[t_arr >= disturbance_time]
        post_t = t_arr[t_arr >= disturbance_time]
        recovery_time = 0.0
        for i in range(len(post_disturb)):
            if np.all(np.abs(post_disturb[i:] - setpoint) <= 1.0):
                recovery_time = post_t[i] - disturbance_time
                break
        if recovery_time == 0.0 and len(post_t) > 0:
            recovery_time = post_t[-1] - disturbance_time

        # 5. 综合评分（加权求和，越小越好）
        score = (overshoot * 2.0 + settling_time * 0.5 +
                 steady_error * 10.0 + recovery_time * 0.3)

        return {
            "overshoot": round(float(overshoot), 4),
            "settling_time": round(float(settling_time), 2),
            "steady_error": round(float(steady_error), 4),
            "recovery_time": round(float(recovery_time), 2),
            "score": round(float(score), 4)
        }