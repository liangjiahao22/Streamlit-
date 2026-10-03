"""
pid_animation.py
智能PID恒温混水装置 - 实时动态仿真（独立运行版）
运行方式：python pid_animation.py
作者：梁嘉豪
日期：2026-09
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pid_core import SystemConfig, RealisticMixer, PIDController

plt.rcParams['toolbar'] = 'None'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # ---- 初始化 ----
    config = SystemConfig()
    mixer = RealisticMixer(config)
    pid = PIDController(Kp=0.25, Ki=0.03, Kd=0.02, setpoint=40.0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('智能PID恒温混水装置 - 实时动态仿真', fontsize=14)
    
    line_temp, = ax1.plot([], [], 'b-', linewidth=2, label='实际水温')
    line_target, = ax1.plot([], [], 'r--', linewidth=1.5, label='目标温度 (40℃)')
    line_disturb = ax1.axvline(x=40, color='gray', linestyle=':',
                               alpha=0.7, label='扰动注入')
    
    line_ratio, = ax2.plot([], [], 'g-', linewidth=1.5, label='阀门开度')
    ax2.axvline(x=40, color='gray', linestyle=':', alpha=0.7)
    
    ax1.set_ylabel('温度 (℃)')
    ax1.set_ylim(15, 70)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('阀门开度')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # ---- 数据缓存 ----
    t_data, temp_data, ratio_data, target_data = [], [], [], []
    state = {"target_ratio": 0.3, "disturbance_applied": False}
    
    def update(frame):
        current_time = frame * 0.1
        t_data.append(current_time)
        
        temp, ratio = mixer.update(state["target_ratio"])
        temp_data.append(temp)
        ratio_data.append(ratio)
        target_data.append(40.0)
        
        # 第40秒施加扰动
        if current_time >= 40 and not state["disturbance_applied"]:
            mixer.apply_disturbance(5.0, "hot_up")
            state["disturbance_applied"] = True
        
        # PID控制
        control = pid.update(temp)
        state["target_ratio"] += control * 0.005
        state["target_ratio"] = float(np.clip(state["target_ratio"], 0.0, 1.0))
        
        # 更新曲线
        line_temp.set_data(t_data, temp_data)
        line_target.set_data(t_data, target_data)
        line_ratio.set_data(t_data, ratio_data)
        
        ax1.set_xlim(0, current_time + 5)
        ax2.set_xlim(0, current_time + 5)
        
        return line_temp, line_target, line_ratio
    
    ani = FuncAnimation(fig, update, frames=600, interval=50,
                        blit=True, repeat=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()