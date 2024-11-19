#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/19/24
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : liduDemo.py
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms


# 定义 ProbitScale 类
class ProbitScale(mscale.ScaleBase):
    name = 'probit'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)

    def get_transform(self):
        return self.ProbitTransform()

    def set_default_locators_and_formatters(self, axis):
        pass  # 可以根据需要设置默认的定位器和格式化器

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


# 注册自定义的 ProbitScale
mscale.register_scale(ProbitScale)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成示例数据
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 0.5, 1000),  # 主要分布在0附近
    np.random.normal(1.2, 0.4, 200)  # 增加一些偏移以确保累积频率和正态概率不同但部分重合
])


def create_particle_distribution_plot(data, bins=20):
    # 创建图形和坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # 创建双Y轴

    # 计算直方图数据
    counts, bins, _ = ax1.hist(data, bins=bins, density=True, alpha=0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 计算频率和累积频率
    freq = counts / counts.sum()
    cum_freq = np.cumsum(freq)

    # 模拟正态概率累积数据，使其与累积频率部分重合
    norm_cum_freq = cum_freq * 0.95 + 0.05 * np.random.random(len(cum_freq))

    # 绘制直方图
    ax1.bar(bin_centers, freq * 100, width=np.diff(bins), alpha=0.7,
            color='lightgray', edgecolor='black', label='频率')

    # 绘制累积频率曲线 (实线)
    ax2.plot(bin_centers, cum_freq, 'k-', linewidth=2, label='累积频率')

    # 绘制正态概率累积 (数据点)
    ax2.plot(bin_centers, norm_cum_freq, 'ko', markersize=5, label='正态概率累积')

    # 绘制频率 (虚线)
    ax1.plot(bin_centers, freq * 100, 'k--', linewidth=1.5, label='频率')

    # 设置坐标轴范围和标签
    ax1.set_xlim(-1, 5)

    # 设置左侧Y轴的等距刻度（0-100%）
    ax1.set_ylim(0, 100)
    ax1.set_yticks(np.arange(0, 101, 10))

    # 设置右侧Y轴为 Probit 坐标
    ax2.set_yscale('probit')

    # 设置右侧Y轴的刻度和标签
    yticks = [0.0001, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999]
    yticklabels = ['0.01%', '0.1%', '1%', '5%', '10%', '20%', '30%', '40%', '50%',
                   '60%', '70%', '80%', '90%', '95%', '99%', '99.9%', '99.99%']
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)

    # 设置右侧Y轴的范围
    ax2.set_ylim(0.0001, 0.9999)

    # 添加标题和标签
    plt.title('粒度分布参数统计图', pad=15, fontsize=12)
    ax1.set_xlabel('粒径 (ψ)')
    ax1.set_ylabel('频率 (%)')
    ax2.set_ylabel('累积百分比 (%)')

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='累积频率'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='频率'),
        plt.Line2D([0], [0], color='black', marker='o', linestyle='', markersize=5, label='正态概率累积')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    # 设置网格
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig


# 创建并显示图形
fig = create_particle_distribution_plot(data)
plt.show()


