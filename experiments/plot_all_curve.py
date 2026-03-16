import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.font_manager as fm

# 路径设置
current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

# 字体设置
# font_prop2 = fm.FontProperties(fname='./simsun.ttc')
# font_prop2.set_size(20)

font_prop = fm.FontProperties(fname='./times.ttf')
font_prop.set_size(20)
font = {'size': 30, 'family': 'Times New Roman'}


def parse_args():
    parser = ArgumentParser(description='Neural Network acc/loss plot')
    # 【修正2】: 如果 default 里有自定义的名字，必须把它们也加到 choices 里，或者直接删掉 choices 参数
    parser.add_argument('--optimizers', type=str, nargs='+',
                        # 方案A：删掉 choices 限制，允许任意文件名
                        # choices=['SGD', 'adam', 'TJU_v1', 'TJU_v3', 'TJU_v4', '111', '222', '333'],
                        default=["ATJU", "SGD", "AdamW", "Adam"],
                        help='optimizers to use')
    return parser.parse_args()


def main():
    args = parse_args()

    colors = ['y', 'k', 'g', 'r', 'b', 'm', 'c']  # 增加一些颜色防止不够用

    legend_map = {
        'ATJU': 'ATJU',
        'SGD': "SGD",  # 示例：可以把文件名 111 映射成更好听的名字
        'AdamW': 'AdamW',
        'Adam': 'Adam',
    }

    directory = r"./txt_save/resnet18_valid_acc"

    plt.figure(figsize=(10, 8))

    for idx, opt in enumerate(args.optimizers):
        filename = f'{opt}.txt'
        file_path = os.path.join(directory, filename)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            plot_data = []
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line:
                    try:
                        val = float(line)
                        plot_data.append(val)
                    except ValueError:
                        continue

        # 【修正1】：plt.plot 必须在读取完文件的所有行之后（with open 块外面或最后），才能进行绘制！

        # 简单的数据保护：如果没有读到数据，跳过
        if not plot_data:
            print(f"Warning: {filename} is empty.")
            continue

        # 数据是否需要归一化 (0-100 -> 0-1)
        # 如果你的数据确认已经是 0.x，这里 y_data = plot_data 即可
        # 如果不确定，保留这个判断比较安全

        if max(plot_data) > 1.0:
            y_data = [x / 100.0 for x in plot_data]
        else:
            y_data = plot_data
        """
        y_data = plot_data
        """
        display_label = legend_map.get(opt, opt)

        # 绘图
        plt.plot(y_data,
                 label=display_label,
                 color=colors[idx % len(colors)],
                 linestyle='-',
                 linewidth=2)

    ax1 = plt.gca()

    # 坐标轴设置
    plt.xlim(0, 100)
    plt.ylim(0, 1)

    x_major_locator = MultipleLocator(20)
    y_major_locator = MultipleLocator(0.1)

    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.yaxis.set_major_locator(y_major_locator)

    # 图例
    plt.legend(loc='lower right', prop=font)

    # 标题和标签
    plt.title('Cifar10', fontdict=font)  # 建议加上 fontdict
    plt.xlabel('Epochs', fontdict=font)
    plt.ylabel('Accuracy', fontdict=font)

    plt.tick_params(labelsize=25)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./data.png')
    plt.show()


"""

111: lr:0.01  weight_decay:5e-4
222: lr:0.005  weight_decay:5e-4
333: lr:0.005  weight_decay:1e-4
444: lr:0.01  weight_decay:1e-4
555: 同222
"""

if __name__ == "__main__":
    main()