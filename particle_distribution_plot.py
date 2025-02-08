import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MultipleLocator
import os

# Define ProbitScale class
class ProbitScale(mscale.ScaleBase):
    name = 'probit'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)

    def get_transform(self):
        return self.ProbitTransform()

    def set_default_locators_and_formatters(self, axis):
        pass  # 可在此设置默认的刻度定位器和格式器

    class ProbitTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return norm.ppf(a)

        def inverted(self):
            return ProbitScale.InvertedProbitTransform()

    class InvertedProbitTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return norm.cdf(a)

        def inverted(self):
            return ProbitScale.ProbitTransform()

# 注册自定义 ProbitScale
mscale.register_scale(ProbitScale)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_particle_distribution_plot(ListFreqData=None, ListCumFreqData=None, psi_data=None):
    """
    绘制粒度分布图，显示频率、累积频率和正态概率累积曲线，
    并在 X 轴添加四分位次刻度（不显示标签）。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # 默认数据
    if ListFreqData is None:
        ListFreqData = [0.00, 5.84, 11.21, 25.69, 9.72, 14.26, 9.59, 9.94, 5.26, 3.30,
                        1.68, 1.18, 0.59, 0.49, 0.73, 0.32, 0.18, 0.01, 0.01, 0.00,
                        0.00, 0.00, 0.00, 0.00, 0.00]
    if ListCumFreqData is None:
        ListCumFreqData = [0.00, 5.84, 17.05, 42.73, 52.46, 66.72, 76.31, 86.25, 91.52, 94.82,
                           96.50, 97.68, 98.27, 98.75, 99.48, 99.80, 99.98, 99.99, 100.00, 100.00,
                           100.00, 100.00, 100.00, 100.00, 100.00]
    if psi_data is None:
        psi_data = [-1.00, -0.75, -0.50, -0.25, -0.00, 0.25, 0.50, 0.75, 1.00, 1.25,
                    1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75,
                    4.00, 4.25, 4.50, 4.75, 5.00]

    psi_data = np.array(psi_data)

    fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax2 = ax1.twinx()  # 右侧 Y 轴

    # 绘制频率条形图
    ax1.bar(psi_data, ListFreqData, width=0.25, alpha=0.7,
            color='lightgray', edgecolor='black', label='频率')
    # 绘制频率折线
    ax1.plot(psi_data, ListFreqData, 'k--', linewidth=1.5, label='频率')
    # 绘制累积频率折线
    ax1.plot(psi_data, ListCumFreqData, 'k-', linewidth=2, label='累积频率')

    # 计算正态概率累积值
    cum_prob = np.clip(np.array(ListCumFreqData) / 100.0, 0.0001, 0.9999)
    normal_prob = norm.ppf(cum_prob)
    ax2.plot(psi_data, normal_prob, 'ko', markersize=5, label='正态概率累积')

    # 设置 X 轴刻度
    ax1.set_xlim(psi_data.min() - 0.25, psi_data.max() + 0.25)
    major_ticks = np.arange(np.floor(psi_data.min()), np.ceil(psi_data.max()) + 1, 1)
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels([str(t) for t in major_ticks])
    minor_ticks = np.arange(np.floor(psi_data.min()), psi_data.max() + 0.26, 0.25)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.tick_params(axis='x', which='minor', length=4)

    # 左侧 Y 轴 (频率) 范围
    ax1.set_ylim(0, 100)
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.set_ylabel('%')

    # 右侧 Y 轴 (正态概率) 刻度
    tick_probs = [0.0001, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999]
    tick_positions = norm.ppf(tick_probs)
    tick_labels = ['0.01%', '0.1%', '1%', '5%', '10%', '20%', '30%', '40%', '50%',
                   '60%', '70%', '80%', '90%', '95%', '99%', '99.90%', '99.99%']
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)
    ax2.set_ylim(norm.ppf(0.0001), norm.ppf(0.9999))

    # 隐藏顶部 spine（双 Y 轴上方的那条线）
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.set_xlabel('粒径(φ)')
    plt.title('粒度分布参数统计图', pad=15, fontsize=12)

    # 自定义图例
    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='累积频率'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='频率'),
        plt.Line2D([0], [0], color='black', marker='o', linestyle='', markersize=5, label='正态概率累积')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    # 仅在 ax1 上显示网格
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.grid(False)

    # 对齐左右 Y 轴标签
    fig.align_ylabels([ax1, ax2])

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制粒度分布参数统计图（含 Probit 坐标）")
    parser.add_argument(
        '--ListFreqData',
        type=lambda s: [float(x) for x in s.strip('[]').split(',')],
        help='频率数据列表，例如： --ListFreqData [0.1,0.5,1.2,1.3]'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=25,
        help='直方图分箱数，默认为 25'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.png',
        help='输出图像文件路径和文件名 (默认为当前路径下的 output.png)'
    )
    parser.add_argument(
        '--ListCumFreqData',
        type=lambda s: [float(x) for x in s.strip('[]').split(',')],
        help='累积频率数据列表，例如： --ListCumFreqData [0.1,0.2,0.3,...]'
    )
    parser.add_argument(
        '--psi_data',
        type=lambda s: [float(x) for x in s.strip('[]').split(',')],
        help='粒径 (∅) 数据列表，例如： --psi_data [-1.0,-0.75,-0.5,...]'
    )
    args = parser.parse_args()

    # 默认频率数据
    if not args.ListFreqData:
        print("未提供 --ListFreqData 参数，使用默认数据...")
        freq_data = [0.00, 5.84, 11.21, 25.69, 9.72, 14.26, 9.59, 9.94, 5.26, 3.30,
                     1.68, 1.18, 0.59, 0.49, 0.73, 0.32, 0.18, 0.01, 0.01, 0.00,
                     0.00, 0.00, 0.00, 0.00, 0.00]
    else:
        freq_data = np.array(args.ListFreqData, dtype=float)

    # 默认累积频率数据
    if args.ListCumFreqData is not None:
        cum_freq_data = np.array(args.ListCumFreqData, dtype=float)
    else:
        print("未提供 --ListCumFreqData 参数，使用默认数据...")
        cum_freq_data = [0.00, 5.84, 17.05, 42.73, 52.46, 66.72, 76.31, 86.25, 91.52, 94.82,
                         96.50, 97.68, 98.27, 98.75, 99.48, 99.80, 99.98, 99.99, 100.00, 100.00,
                         100.00, 100.00, 100.00, 100.00, 100.00]

    # 默认粒径数据
    if args.psi_data is not None:
        psi_data = np.array(args.psi_data, dtype=float)
    else:
        print("未提供 --psi_data 参数，使用默认数据...")
        psi_data = [-1.00, -0.75, -0.50, -0.25, -0.00, 0.25, 0.50, 0.75, 1.00, 1.25,
                    1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75,
                    4.00, 4.25, 4.50, 4.75, 5.00]

    # 生成图像
    fig = create_particle_distribution_plot(freq_data, cum_freq_data, psi_data)

    # 保存图像
    save_path = args.output  # 例如：'./my_figure.png' 或 'output.png'
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    print(f"图像已保存到: {save_path}")

    plt.show()
