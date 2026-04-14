import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import re
import glob
from pay_off_compute import compute_pay_off

# ================= 配置区域 =================
LOG_DIR_CLASS = './log/cnn/MEDMNIST/exp_D/class'  # Transfer 日志路径
LOG_DIR_FEDAVG = './log/cnn/MEDMNIST/exp_D/fedavg'
LOG_DIR_EXP_E = './log/cnn/CIFAR10/exp_E'
# OUTPUT_DIR = "./result/exp3"
OUTPUT_DIR = "./result/chinese/exp3"

# 端口定义
CLIENT_GENIUS = '50051'
CLIENT_FREERIDER = '50069'
# 普通节点列表 (根据 exp_D 文件列表推断: 50053 到 50067)
CLIENTS_ORDINARY = [str(port) for port in range(50053, 50069, 2)] 
# ===========================================

def setup_tifs_style() -> None:
    """Set a concise TIFS-like publication style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            # 修正：将 Times New Roman 移至 SimHei 之前
            "font.serif": ["SimHei", "Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False, 
            "mathtext.fontset": "stix",
            "font.size": 20,
            "axes.labelsize": 22,
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "savefig.bbox": "tight",
        }
    )

# 定义一个强制使用特定量级 (如 10^-2) 的 Formatter
class FixedOrderFormatter(ticker.ScalarFormatter):
    def __init__(self, order=0, useMathText=True):
        super().__init__(useMathText=useMathText)
        self.order = order

    def _set_order_of_magnitude(self):
        # 强制设置量级，而不是让 matplotlib 自动计算
        self.orderOfMagnitude = self.order

    # ======== 新增/重写以下方法 ========
    def get_offset(self):
        """
        重写偏移文本的生成逻辑。
        直接返回纯净的 LaTeX 数学表达式，去掉了原生的 \mathdefault 包装。
        这样会强制 Matplotlib 使用 stix 数学字体来渲染这个 10 的次幂和负号，
        完美避开 SimHei 缺负号的问题。
        """
        if not hasattr(self, 'orderOfMagnitude'):
            self.orderOfMagnitude = self.order
            
        if self.orderOfMagnitude != 0:
            # %d 格式化会将 -2 转换为带有 ASCII 连字符的纯文本，
            # 在外层 $...$ 的数学环境下，stix 字体会正确将其渲染为标准的负号。
            return r"$\times 10^{%d}$" % self.orderOfMagnitude
        return ""

def parse_log_file(filepath):
    """ 解析日志返回 (Rounds, Values) """
    rounds = []
    values = []
    if not os.path.exists(filepath):
        return np.array([]), np.array([])
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            line_clean = re.sub(r'\'', '', line)
            parts = line_clean.split()
            try:
                if len(parts) >= 2:
                    rounds.append(int(parts[0]))
                    values.append(float(parts[1]))
            except ValueError:
                continue
    if rounds:
        data = sorted(zip(rounds, values), key=lambda x: x[0])
        rounds, values = zip(*data)
        return np.array(rounds), np.array(values)
    return np.array([]), np.array([])

def get_cumulative_curve(port, log_type='transfer'):
    """ 读取指定端口的日志并计算累积曲线，统一插值到 0-100 轮 """
    filename = f'Transfer_{port}.txt' if log_type == 'transfer' else f'cost_{port}.txt'
    filepath = os.path.join(LOG_DIR_CLASS, log_type, filename)
    
    r, v = parse_log_file(filepath)
    if len(r) == 0:
        return None
    
    # 计算累积
    cum_v = np.cumsum(v)
    
    # 统一插值到标准 X 轴 (0-100) 以便计算平均
    x_grid = np.arange(0, 101)
    
    # 如果数据没到 100 轮，补全
    if r[-1] < 100:
        r = np.append(r, 100)
        cum_v = np.append(cum_v, cum_v[-1])
        
    # 插值
    y_interp = np.interp(x_grid, r, cum_v)
    return x_grid, y_interp

def plot_final_v2():
    print("正在绘制 V-D 最终修正版 v2: 四条曲线对比...")
    setup_tifs_style()
    # 1. 获取天才数据 (Transfer & Cost)
    res_g_trans = get_cumulative_curve(CLIENT_GENIUS, 'transfer')
    res_g_cost = get_cumulative_curve(CLIENT_GENIUS, 'cost')
    
    if not res_g_trans or not res_g_cost:
        print("Error: 天才节点数据缺失")
        return
    x_grid, y_g_trans = res_g_trans
    _, y_g_cost = res_g_cost

    # 2. 获取混子数据 (Transfer)
    res_f_trans = get_cumulative_curve(CLIENT_FREERIDER, 'transfer')
    if not res_f_trans:
        print("Error: 混子节点数据缺失")
        return
    _, y_f_trans = res_f_trans
    
    # 3. 计算普通节点平均曲线
    ord_curves = []
    for port in CLIENTS_ORDINARY:
        res = get_cumulative_curve(port, 'transfer')
        if res:
            ord_curves.append(res[1]) # 只存 Y 值
    
    if ord_curves:
        # 按列求平均
        y_ord_avg = np.mean(ord_curves, axis=0)
    else:
        y_ord_avg = np.zeros_like(x_grid)
        print("Warning: 未找到普通节点数据")

    # ================= 绘图 =================
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # 线条样式定义
    # Markevery=10: 每10个点画一个标
    mk_interval = 10
    
    # 1. Genius Transfer (Income) - 绿色星号
    ax.plot(x_grid, y_g_trans, color='#2ca02c', linewidth=2.5, linestyle='-', 
            marker='*', markersize=9, markevery=mk_interval, label='高质量参与方: 货币转移')
    
    # 2. Ordinary Average (Baseline) - 橙色圆点
    ax.plot(x_grid, y_ord_avg, color='#ff7f0e', linewidth=2.5, linestyle='-', 
            marker='o', markersize=6, markevery=mk_interval, label='普通参与方: 货币转移')
    
    # 3. Genius Cost (Cost) - 蓝色方块 (虚线)
    ax.plot(x_grid, y_g_cost, color='#1f77b4', linewidth=2.0, linestyle='--', 
            marker='s', markersize=6, markevery=mk_interval, alpha=0.9, label='高质量参与方: 成本')

    # 4. Free-rider Transfer (Penalty) - 红色叉号
    ax.plot(x_grid, y_f_trans, color='#d62728', linewidth=2.5, linestyle='-', 
            marker='x', markersize=7, markevery=mk_interval, label='低质量参与方: 货币转移')
    
    # 0 轴基准线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.4)

    # === 修改 1: 添加第 100 轮的贯穿竖线 ===
    # zorder=0 保证线在图层最下方，不会挡住曲线
    ax.axvline(x=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0, label='训练结束')

    # === 修改 2: X 轴范围往前拓宽 ===
    # 设为 105 或 110，给右侧留出空间
    ax.set_xlim(0, 105)
    ax.set_xlabel('训练轮次', fontsize=16)
    
    # Y 轴标签
    ax.set_ylabel('累积值', fontsize=16)
    
    # === 修改 3: 强制 Y 轴使用 10^-2 次方 ===
    # 实例化我们要强制 -2 次方的 Formatter
    formatter = FixedOrderFormatter(order=-2, useMathText=False)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((0, 0)) # 配合 FixedOrderFormatter 强制显示
    
    ax.yaxis.set_major_formatter(formatter)
    
    # 调整字体大小
    ax.yaxis.get_offset_text().set_fontsize(16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 图例
    ax.legend(fontsize=16, loc='upper left').set_zorder(10)
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, 'V-D_Cumulative_Transfer_Final_v3.png')
    eps_path = os.path.join(OUTPUT_DIR, 'V-D_Cumulative_Transfer_Final_v3.eps')
    plt.savefig(png_path, dpi=300)
    plt.savefig(eps_path, format='eps')
    print(f"图表已保存 (v3修正版): {png_path}")
    print(f"图表已保存 (v3修正版): {eps_path}")

def plot_payoff_comparison():
    """对比 exp_E 下不同方案的总 pay-off。"""
    setup_tifs_style()
    scheme_dirs = ["class", "greedy", "random", "pure", "lazy"]
    total_payoffs = {}

    for scheme in scheme_dirs:
        base_dir = os.path.join(LOG_DIR_EXP_E, scheme)
        payoff_files = glob.glob(os.path.join(base_dir, "pay_off", "pay_off*.txt"))
        if not payoff_files:
            print(f"Warning: 未找到 {scheme} 的 pay_off 文件")
            total_payoffs[scheme] = 0.0
            continue

        participant_ids = []
        for path in payoff_files:
            match = re.search(r"pay_off_?(\d+)\.txt$", os.path.basename(path))
            if match:
                participant_ids.append(match.group(1))

        total_payoffs[scheme] = sum(
            compute_pay_off(participant_id, base_dir, False)
            for participant_id in participant_ids
        )

    # labels = [scheme.upper() for scheme in scheme_dirs]
    labels = ["Our Scheme", "Greedy", "Lazy", "Pure", "Random"]
    values = [total_payoffs[scheme] for scheme in scheme_dirs]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(
        labels,
        values,
        color=['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b'],
        edgecolor='black',
        linewidth=1.0,
    )

    ax.set_ylabel('总收益', fontsize=16)
    ax.set_xlabel('方案', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=16,
        )

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, 'payoff_comparison_expE_total_cifar.png')
    eps_path = os.path.join(OUTPUT_DIR, 'payoff_comparison_expE_total_cifar.eps')
    plt.savefig(png_path, dpi=300)
    plt.savefig(eps_path, format='eps')
    print(f"Payoff 对比图已保存: {png_path}")
    print(f"Payoff 对比图已保存: {eps_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # plot_final_v2()
    plot_payoff_comparison()