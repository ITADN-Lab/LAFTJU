import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.font_manager as fm

# Path settings
current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

# Font settings
# font_prop2 = fm.FontProperties(fname='./simsun.ttc')
# font_prop2.set_size(20)

font_prop = fm.FontProperties(fname='./times.ttf')
font_prop.set_size(20)
font = {'size': 30, 'family': 'Times New Roman'}


def parse_args():
    parser = ArgumentParser(description='Neural Network acc/loss plot')
    # [Fix 2]: If default contains custom names, they must also be added to choices, or simply remove the choices parameter
    parser.add_argument('--optimizers', type=str, nargs='+',
                        # Option A: remove choices restriction to allow any filename
                        # choices=['SGD', 'adam', 'TJU_v1', 'TJU_v3', 'TJU_v4', '111', '222', '333'],
                        default=["ATJU", "SGD", "AdamW", "Adam"],
                        help='optimizers to use')
    return parser.parse_args()


def main():
    args = parse_args()

    colors = ['y', 'k', 'g', 'r', 'b', 'm', 'c']  # add more colors to avoid running out

    legend_map = {
        'ATJU': 'ATJU',
        'SGD': "SGD",  # example: map filename 111 to a more readable name
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

        # [Fix 1]: plt.plot must be called after reading all lines of the file (outside or at the end of the with open block)!

        # Simple data guard: skip if no data was read
        if not plot_data:
            print(f"Warning: {filename} is empty.")
            continue

        # Whether data needs normalization (0-100 -> 0-1)
        # If your data is already in 0.x format, use y_data = plot_data directly
        # If unsure, keeping this check is safer

        if max(plot_data) > 1.0:
            y_data = [x / 100.0 for x in plot_data]
        else:
            y_data = plot_data
        """
        y_data = plot_data
        """
        display_label = legend_map.get(opt, opt)

        # Plot
        plt.plot(y_data,
                 label=display_label,
                 color=colors[idx % len(colors)],
                 linestyle='-',
                 linewidth=2)

    ax1 = plt.gca()

    # Axis settings
    plt.xlim(0, 100)
    plt.ylim(0, 1)

    x_major_locator = MultipleLocator(20)
    y_major_locator = MultipleLocator(0.1)

    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.yaxis.set_major_locator(y_major_locator)

    # Legend
    plt.legend(loc='lower right', prop=font)

    # Title and labels
    plt.title('Cifar10', fontdict=font)  # Recommended to add fontdict
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
555: same as 222
"""

if __name__ == "__main__":
    main()