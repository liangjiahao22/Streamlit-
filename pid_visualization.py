"""
pid_visualization.py
智能PID恒温混水装置 - 可视化模块

提供所有绘图函数：
1. plot_simulation        - 单次仿真结果（温度+阀门+PID分量）
2. plot_comparison        - 多组PID参数对比
3. plot_metrics_table     - 性能指标表格
4. plot_pid_components    - PID三个分量可视化

作者：梁嘉豪
日期：2026-09
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# 全局中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['toolbar'] = 'None'


def plot_simulation(result: Dict, show_pid_components: bool = False):
    """
    绘制单次仿真结果
    
    参数：
        result: PerformanceMetrics.run_simulation() 的返回值
        show_pid_components: 是否显示PID分量
    """
    t = result["t"]
    temps = result["temps"]
    ratios = result["ratios"]
    setpoint = result["setpoint"]
    dist_time = result["disturbance_time"]
    
    if show_pid_components:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        ax1, ax2, ax3 = axes
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax3 = None
    
    # ---- 温度曲线 ----
    ax1.plot(t, temps, 'b-', linewidth=2, label='实际水温')
    ax1.axhline(y=setpoint, color='r', linestyle='--',
                linewidth=1.5, label=f'目标温度 ({setpoint}℃)')
    ax1.axvline(x=dist_time, color='gray', linestyle=':',
                alpha=0.7, label='扰动注入时刻')
    ax1.fill_between(t, setpoint - 1, setpoint + 1,
                     color='green', alpha=0.1, label='±1℃误差带')
    ax1.set_ylabel('温度 (℃)', fontsize=11)
    ax1.set_title('智能PID恒温控制仿真 - 温度响应', fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(min(min(temps) - 5, 10), max(max(temps) + 5, 70))
    
    # ---- 阀门开度曲线 ----
    ax2.plot(t, ratios, 'g-', linewidth=1.5, label='阀门开度')
    ax2.axvline(x=dist_time, color='gray', linestyle=':', alpha=0.7)
    ax2.set_ylabel('阀门开度比例', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 1.05)
    
    # ---- PID分量曲线 ----
    if show_pid_components and ax3 is not None:
        ax3.plot(t, result["P"], 'r-', linewidth=1.2, label='比例项 P')
        ax3.plot(t, result["I"], 'b-', linewidth=1.2, label='积分项 I')
        ax3.plot(t, result["D"], 'g-', linewidth=1.2, label='微分项 D')
        ax3.axvline(x=dist_time, color='gray', linestyle=':', alpha=0.7)
        ax3.set_xlabel('时间 (秒)', fontsize=11)
        ax3.set_ylabel('PID分量输出', fontsize=11)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.set_xlabel('时间 (秒)', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_comparison(results_list: List[Dict], labels: List[str] = None):
    """
    绘制多组PID参数的对比曲线
    
    参数：
        results_list: 多个仿真结果的列表
        labels: 每个结果的标签
    """
    if labels is None:
        labels = [f"方案{i+1}" for i in range(len(results_list))]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for i, (result, label) in enumerate(zip(results_list, labels)):
        t = result["t"]
        ax1.plot(t, result["temps"], color=colors[i],
                 linewidth=1.5, label=f"{label}: Kp={result.get('Kp','?')}, "
                                       f"Ki={result.get('Ki','?')}, "
                                       f"Kd={result.get('Kd','?')}")
        ax2.plot(t, result["ratios"], color=colors[i],
                 linewidth=1.2, alpha=0.7)
    
    setpoint = results_list[0]["setpoint"]
    dist_time = results_list[0]["disturbance_time"]
    
    ax1.axhline(y=setpoint, color='black', linestyle='--',
                linewidth=1.0, alpha=0.6, label=f'目标温度 ({setpoint}℃)')
    ax1.axvline(x=dist_time, color='gray', linestyle=':', alpha=0.7)
    ax1.set_ylabel('温度 (℃)', fontsize=11)
    ax1.set_title('多组PID参数对比', fontsize=13)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.axvline(x=dist_time, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('时间 (秒)', fontsize=11)
    ax2.set_ylabel('阀门开度', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig


def plot_metrics_table(results_list: List[Dict], labels: List[str] = None):
    """
    绘制性能指标对比表格（图形化）
    """
    if labels is None:
        labels = [f"方案{i+1}" for i in range(len(results_list))]
    
    metrics = ["overshoot", "settling_time", "steady_error", "score"]
    metric_names = ["超调量(%)", "调节时间(s)", "稳态误差(℃)", "综合评分"]
    
    data = []
    for result, label in zip(results_list, labels):
        row = [label]
        for m in metrics:
            row.append(f"{result.get(m, 0):.3f}")
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, max(3, len(results_list) * 0.6 + 1.5)))
    ax.axis('off')
    
    table = ax.table(
        cellText=data,
        colLabels=["方案"] + metric_names,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 表头样式
    for i in range(len(metric_names) + 1):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    ax.set_title('性能指标对比', fontsize=13, pad=20)
    plt.tight_layout()
    return fig