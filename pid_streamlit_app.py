"""
pid_streamlit_app.py
智能PID恒温混水装置 - Web交互式仿真平台（主应用）

功能：
1. 单次仿真与实时动态演示（拆分为独立小图，方便截图）
2. 多参数对比分析
3. PID自动整定
4. 数据导出

作者：梁嘉豪
日期：2026-09
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from pid_core import (
    SystemConfig, RealisticMixer, PIDController,
    AutoTuner, PerformanceMetrics
)
from pid_visualization import (
    plot_comparison, plot_metrics_table, plot_simulation
)

# ================= 全局字体设置（防止中文乱码） =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 页面配置 =================
st.set_page_config(
    page_title="智能PID恒温混水装置 - 仿真平台",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 智能PID恒温混水装置 — 交互式仿真平台")
st.markdown("基于真实物理模型（阀门惯性+传输延迟）的PID控制仿真，"
            "支持实时参数调整、性能指标计算与数据导出。")


# ================= 侧边栏参数 =================
with st.sidebar:
    st.header("⚙️ 参数设置")

    # ---- PID参数 ----
    st.subheader("PID控制器参数")
    Kp = st.slider("比例增益 Kp", 0.0, 1.0, 0.25, 0.01,
                   help="越大响应越快，过大会导致振荡")
    Ki = st.slider("积分增益 Ki", 0.0, 0.1, 0.03, 0.001,
                   help="消除稳态误差，过大会导致超调")
    Kd = st.slider("微分增益 Kd", 0.0, 0.1, 0.02, 0.001,
                   help="抑制振荡，提前预测误差变化")

    st.divider()

    # ---- 系统物理参数 ----
    st.subheader("系统物理参数")
    setpoint = st.slider("目标水温 (℃)", 30.0, 50.0, 40.0, 0.5)
    hot_temp = st.slider("热水初始温度 (℃)", 50.0, 70.0, 60.0, 1.0)
    cold_temp = st.slider("冷水初始温度 (℃)", 10.0, 20.0, 15.0, 1.0)
    valve_response = st.slider("阀门响应时间 (秒)", 0.5, 5.0, 2.0, 0.1)
    transport_delay = st.slider("水流传输延迟 (秒)", 0.5, 3.0, 1.0, 0.1)

    st.divider()

    # ---- 扰动设置 ----
    st.subheader("扰动设置")
    disturbance_time = st.slider("扰动发生时间 (秒)", 20.0, 60.0, 40.0, 1.0)
    disturbance_strength = st.slider("扰动强度", 0.0, 10.0, 5.0, 0.5)
    disturbance_type = st.selectbox(
        "扰动类型",
        ["hot_up", "hot_down", "flow_change"],
        format_func=lambda x: {
            "hot_up": "热水温度上升（冷水压力下降）",
            "hot_down": "热水温度下降（热水供应不足）",
            "flow_change": "流量变化"
        }[x]
    )

    st.divider()

    # ---- 高级选项 ----
    st.subheader("高级选项")
    anti_windup = st.checkbox("启用积分限幅（抗饱和）", value=True)
    derivative_first = st.checkbox("启用微分先行", value=True)

    sim_time = st.slider("仿真总时长 (秒)", 60.0, 150.0, 100.0, 10.0)


# ================= 主区域：Tab布局 =================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 单次仿真", "📊 多参数对比", "🎯 自动整定", "📥 数据导出"]
)

# ---- Tab 1: 单次仿真 ----
with tab1:
    # 第一步：运行仿真，并把结果存入 session_state
    if st.button("🚀 运行仿真", type="primary", use_container_width=True):
        with st.spinner("正在运行仿真..."):
            config = SystemConfig(
                hot_temp=hot_temp, cold_temp=cold_temp,
                valve_response_time=valve_response,
                transport_delay=transport_delay, dt=0.1
            )

            result = PerformanceMetrics.run_simulation(
                config=config, setpoint=setpoint,
                Kp=Kp, Ki=Ki, Kd=Kd, sim_time=sim_time,
                disturbance_time=disturbance_time,
                disturbance_strength=disturbance_strength,
                disturbance_type=disturbance_type,
                anti_windup=anti_windup,
                derivative_first=derivative_first
            )
            # 核心：把结果存入 session_state，这样点单选按钮时数据不会丢
            st.session_state["last_result"] = result

    # 第二步：只要有结果，就展示图表（不管点没点按钮）
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        # 绘制图表（单选，只画一张图，方便截图！）
        chart_option = st.radio(
            "选择查看的图表（方便截图排版）：",
            ["温度响应", "阀门开度", "PID分量"],
            horizontal=True
        )

        if chart_option == "温度响应":
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(result["t"], result["temps"], 'b-', linewidth=2, label='实际水温')
            ax.axhline(y=setpoint, color='r', linestyle='--', linewidth=1.5, label=f'目标温度 ({setpoint}℃)')
            ax.axvline(x=disturbance_time, color='gray', linestyle=':', alpha=0.7, label='扰动注入时刻')
            ax.fill_between(result["t"], setpoint-1, setpoint+1, color='green', alpha=0.1, label='±1℃误差带')
            ax.set_ylabel('温度 (℃)')
            ax.set_xlabel('时间 (秒)')
            ax.set_title('智能PID恒温控制仿真 - 温度响应', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        elif chart_option == "阀门开度":
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(result["t"], result["ratios"], 'g-', linewidth=1.5, label='实际热水阀门开度')
            ax.axvline(x=disturbance_time, color='gray', linestyle=':', alpha=0.7)
            ax.set_ylabel('阀门开度比例')
            ax.set_xlabel('时间 (秒)')
            ax.set_title('智能PID恒温控制仿真 - 阀门开度', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        elif chart_option == "PID分量":
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(result["t"], result["P"], 'r-', linewidth=1.2, label='比例项 P')
            ax.plot(result["t"], result["I"], 'b-', linewidth=1.2, label='积分项 I')
            ax.plot(result["t"], result["D"], 'g-', linewidth=1.2, label='微分项 D')
            ax.axvline(x=disturbance_time, color='gray', linestyle=':', alpha=0.7)
            ax.set_ylabel('PID分量输出')
            ax.set_xlabel('时间 (秒)')
            ax.set_title('智能PID恒温控制仿真 - PID分量', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # 性能指标卡片
        st.subheader("📊 系统性能指标")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("最大超调量", f"{result['overshoot']:.2f}%",
                      delta="优秀" if result['overshoot'] < 5 else "偏大",
                      delta_color="normal" if result['overshoot'] < 5 else "inverse")
        with col2:
            st.metric("调节时间", f"{result['settling_time']:.1f} 秒",
                      delta="快速" if result['settling_time'] < 15 else "偏长",
                      delta_color="normal" if result['settling_time'] < 15 else "inverse")
        with col3:
            st.metric("稳态误差", f"{result['steady_error']:.3f} ℃",
                      delta="优秀" if result['steady_error'] < 0.5 else "偏大",
                      delta_color="normal" if result['steady_error'] < 0.5 else "inverse")
        with col4:
            st.metric("扰动恢复时间", f"{result['recovery_time']:.1f} 秒",
                      delta="快速" if result['recovery_time'] < 20 else "偏长",
                      delta_color="normal" if result['recovery_time'] < 20 else "inverse")

    else:
        # 如果还没有结果，显示提示
        st.info("👈 请在左侧调整参数，然后点击「运行仿真」按钮开始演示")
        st.markdown("""
        ### 💡 使用提示
        1. **PID整定技巧**：先调大Kp直到系统开始振荡，再加Ki消除静差，最后加Kd抑制振荡
        2. **物理参数**：阀门响应时间和传输延迟越大，系统越难控制，需要更小的PID参数
        3. **扰动测试**：调整扰动强度可测试系统的抗干扰能力
        4. **对比分析**：切换到「多参数对比」Tab，可以同时对比多组PID参数的性能
        """)


# ---- Tab 2: 多参数对比 ----
with tab2:
    st.subheader("多组PID参数对比")
    st.markdown("输入多组PID参数，一键对比它们的控制效果。")

    n_schemes = st.number_input("对比方案数量", 2, 5, 3, 1)

    schemes = []
    cols = st.columns(n_schemes)
    defaults = [(0.25, 0.03, 0.02), (0.60, 0.03, 0.02),
                (0.25, 0.00, 0.02), (0.10, 0.01, 0.00), (0.25, 0.03, 0.10)]

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**方案 {i+1}**")
            kp = st.number_input(f"Kp_{i}", 0.0, 2.0, defaults[i][0], 0.01, key=f"kp_{i}")
            ki = st.number_input(f"Ki_{i}", 0.0, 0.2, defaults[i][1], 0.001, key=f"ki_{i}")
            kd = st.number_input(f"Kd_{i}", 0.0, 0.2, defaults[i][2], 0.001, key=f"kd_{i}")
            schemes.append({"Kp": kp, "Ki": ki, "Kd": kd})

    if st.button("🔬 运行对比仿真", type="primary", use_container_width=True):
        with st.spinner("正在运行对比仿真..."):
            config = SystemConfig(
                hot_temp=hot_temp, cold_temp=cold_temp,
                valve_response_time=valve_response,
                transport_delay=transport_delay, dt=0.1
            )

            results_list = []
            labels = []
            for i, scheme in enumerate(schemes):
                result = PerformanceMetrics.run_simulation(
                    config=config, setpoint=setpoint,
                    Kp=scheme["Kp"], Ki=scheme["Ki"], Kd=scheme["Kd"],
                    sim_time=sim_time,
                    disturbance_time=disturbance_time,
                    disturbance_strength=disturbance_strength,
                    disturbance_type=disturbance_type,
                    anti_windup=anti_windup,
                    derivative_first=derivative_first
                )
                result.update(scheme)
                results_list.append(result)
                labels.append(f"方案{i+1}")

            # 绘制对比曲线
            fig1 = plot_comparison(results_list, labels)
            st.pyplot(fig1, use_container_width=True)

            # 绘制性能指标表
            fig2 = plot_metrics_table(results_list, labels)
            st.pyplot(fig2, use_container_width=True)

            # 数据表格
            st.subheader("性能指标明细")
            df = pd.DataFrame([{
                "方案": labels[i],
                "Kp": results_list[i]["Kp"],
                "Ki": results_list[i]["Ki"],
                "Kd": results_list[i]["Kd"],
                "超调量(%)": results_list[i]["overshoot"],
                "调节时间(s)": results_list[i]["settling_time"],
                "稳态误差(℃)": results_list[i]["steady_error"],
                "综合评分": results_list[i]["score"]
            } for i in range(len(results_list))])
            st.dataframe(df, use_container_width=True)

            best_idx = int(np.argmin([r["score"] for r in results_list]))
            st.success(f"🏆 综合评分最优方案：**方案 {best_idx + 1}** "
                       f"(Kp={results_list[best_idx]['Kp']}, "
                       f"Ki={results_list[best_idx]['Ki']}, "
                       f"Kd={results_list[best_idx]['Kd']})")


# ---- Tab 3: 自动整定 ----
with tab3:
    st.subheader("PID参数自动整定")
    st.markdown("基于Ziegler-Nichols法和随机搜索，自动推荐最优PID参数。")

    method = st.radio("整定方法", ["随机搜索", "Ziegler-Nichols法（需输入Ku和Tu）"])

    if method == "随机搜索":
        n_trials = st.slider("搜索次数", 5, 30, 12, 1)
        if st.button("🎯 开始自动搜索", type="primary", use_container_width=True):
            with st.spinner(f"正在运行{n_trials}次仿真搜索..."):
                config = SystemConfig(
                    hot_temp=hot_temp, cold_temp=cold_temp,
                    valve_response_time=valve_response,
                    transport_delay=transport_delay, dt=0.1
                )
                results = AutoTuner.auto_search(
                    setpoint=setpoint, config=config, n_trials=n_trials
                )
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                best = results[0]
                st.success(f"🏆 推荐参数：**Kp={best['Kp']}, "
                           f"Ki={best['Ki']}, Kd={best['Kd']}** "
                           f"（超调{best['overshoot']:.2f}%，"
                           f"调节时间{best['settling_time']:.1f}秒）")
    else:
        Ku = st.number_input("临界增益 Ku", 0.1, 5.0, 1.0, 0.1)
        Tu = st.number_input("临界振荡周期 Tu (秒)", 0.5, 20.0, 4.0, 0.5)
        if st.button("🎯 计算推荐参数", type="primary", use_container_width=True):
            params = AutoTuner.ziegler_nichols(Ku, Tu)
            st.success(f"Ziegler-Nichols推荐参数："
                       f"**Kp={params['Kp']}, Ki={params['Ki']}, Kd={params['Kd']}**")


# ---- Tab 4: 数据导出 ----
with tab4:
    st.subheader("仿真数据导出")
    st.markdown("将上一次仿真结果导出为CSV或图片，用于报告和论文。")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        df = pd.DataFrame({
            "时间(s)": result["t"],
            "水温(℃)": result["temps"],
            "阀门开度": result["ratios"],
            "P分量": result["P"],
            "I分量": result["I"],
            "D分量": result["D"]
        })

        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"（共 {len(df)} 行数据，仅显示前20行）")

        # CSV下载
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 下载CSV数据",
            data=csv,
            file_name="pid_simulation_result.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 图片下载（这里保留原本的完整大图，供下载使用）
        fig = plot_simulation(result, show_pid_components=True)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        st.download_button(
            "🖼️ 下载完整仿真图（PNG）",
            data=buf.getvalue(),
            file_name="pid_simulation_plot.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.warning("⚠️ 请先在「单次仿真」Tab运行一次仿真。")


# ================= 页脚 =================
st.divider()
st.caption("© 2026 智能PID恒温混水装置项目 | 广州应用科技学院计算机学院 | "
           "作者：梁嘉豪")