#!/usr/bin/env python3

import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tools import csv2numpy, find_all_files, group_files


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''Smooth signal y, where radius is determines the size of the window.

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / \
            np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / \
            np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)

# COLORS=(
#     [
#         'dodgerblue',
#         'darkorange',  # YELLOW
#         'hotpink',  # PINK
#         'forestgreen',  # GREEN
#     ]
# )

COLORS=(
    [
"dodgerblue", "darkorange", "hotpink"
    ]
)

def plot_ax(
    ax,
    file_lists,
    legend_pattern=".*",
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    xkey='env_step',
    ykey='rew',
    smooth_radius=0,
    shaded_std=True,
    legend_outside=False,
):

    def legend_fn(x):
        # return os.path.split(os.path.join(
        #     args.root_dir, x))[0].replace('/', '_') + " (10)"
        return re.search(legend_pattern, x).group(0)

    legneds = map(legend_fn, file_lists)
    # sort filelist according to legends
    file_lists = [f for _, f in sorted(zip(legneds, file_lists))]
    # file_lists = ['ppo_8env/test_rew_2seeds.csv', 'ppo_16env/test_rew_2seeds.csv', 'ppo_32env/test_rew_2seeds.csv', 'ppo_64env/test_rew_2seeds.csv', 'ppo_128env/test_rew_2seeds.csv', 'ppo_256env/test_rew_2seeds.csv', 'ppo_512env/test_rew_2seeds.csv', 'ppo_1024env/test_rew_2seeds.csv', 'ppo_2048env/test_rew_2seeds.csv']
    # file_lists = ['sac_8env/test_rew_2seeds.csv', 'sac_16env/test_rew_2seeds.csv', 'sac_32env/test_rew_2seeds.csv', 'sac_64env/test_rew_3seeds.csv', 'sac_128env/test_rew_2seeds.csv', 'sac_256env/test_rew_2seeds.csv', 'sac_512env/test_rew_2seeds.csv', 'sac_1024env/test_rew_2seeds.csv', 'sac_2048env/test_rew_2seeds.csv']
    # file_lists = ['DAPG', 'PPO + PC']
    legneds = list(map(legend_fn, file_lists))
    legneds = ['DAPG', "PPO", 'PPO + PC']

    for index, csv_file in enumerate(file_lists):
        csv_dict = csv2numpy(csv_file)
        x, y = csv_dict[xkey], csv_dict[ykey]
        if index in [0, 2]:
            x = x * 8 * 2048
        y = smooth(y, radius=smooth_radius)
        color = COLORS[index % len(COLORS)]
        ax.plot(x, y, color=color, linewidth=3)

    ax.legend(
        legneds,
        loc=2 if legend_outside else "upper left",
        bbox_to_anchor=(1, 1) if legend_outside else None,
        fontsize = 13,
    )

    for index, csv_file in enumerate(file_lists):
        csv_dict = csv2numpy(csv_file)
        x, y = csv_dict[xkey], csv_dict[ykey]
        if index in [0, 2]:
            x = x * 8 * 2048
        y = smooth(y, radius=smooth_radius)
        color = COLORS[index % len(COLORS)]
        if shaded_std and ykey + ':shaded' in csv_dict:
            y_shaded = smooth(csv_dict[ykey + ':shaded'], radius=smooth_radius)
            ax.fill_between(x, y - y_shaded, y + y_shaded, color=color, alpha=.35)

    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    if xlim is not None:
        ax.set_xlim(xmin=0, xmax=xlim)
    # add title
    ax.set_title(title)
    # add labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize = 17)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize = 17)


def plot_figure(
    file_lists,
    group_pattern=None,
    fig_length=6,
    fig_width=6,
    sharex=False,
    sharey=False,
    title=None,
    **kwargs,
):
    if not group_pattern:
        fig, ax = plt.subplots(figsize=(fig_length, fig_width))
        plot_ax(ax, file_lists, title=title, **kwargs)
    else:
        res = group_files(file_lists, group_pattern)
        row_n = int(np.ceil(len(res) / 3))
        col_n = min(len(res), 3)
        fig, axes = plt.subplots(
            row_n,
            col_n,
            sharex=sharex,
            sharey=sharey,
            figsize=(fig_length * col_n, fig_width * row_n),
            squeeze=False
        )
        axes = axes.flatten()
        for i, (k, v) in enumerate(res.items()):
            plot_ax(axes[i], v, title=k, **kwargs)

    if title:  # add title
        fig.suptitle(title, y = 0.93, fontsize=23)

    bwith = 1.5 #边框宽度设置为2
    TK = plt.gca()#获取边框
    TK.spines['bottom'].set_linewidth(bwith)#图框下边
    TK.spines['left'].set_linewidth(bwith)#图框左边
    TK.spines['top'].set_linewidth(bwith)#图框上边
    TK.spines['right'].set_linewidth(bwith)#图框右边
    plt.tick_params(labelsize=14)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument(
        '--fig-length',
        type=int,
        default=6,
        help='matplotlib figure length (default: 6)'
    )
    parser.add_argument(
        '--fig-width',
        type=int,
        default=6,
        help='matplotlib figure width (default: 6)'
    )
    parser.add_argument(
        '--style',
        default='seaborn-paper',
        help='matplotlib figure style (default: seaborn)'
    )
    parser.add_argument(
        '--title', default=None, help='matplotlib figure title (default: None)'
    )
    parser.add_argument(
        '--xkey',
        default='env_step',
        help='x-axis key in csv file (default: env_step)'
    )
    parser.add_argument(
        '--ykey', default='rew', help='y-axis key in csv file (default: rew)'
    )
    parser.add_argument(
        '--smooth', type=int, default=0, help='smooth radius of y axis (default: 0)'
    )
    parser.add_argument(
        '--xlabel', default='Environment Steps', help='matplotlib figure xlabel'
    )
    parser.add_argument(
        '--ylabel', default='Average Episode Reward', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--shaded-std',
        action='store_true',
        help='shaded region corresponding to standard deviation of the group'
    )
    parser.add_argument(
        '--sharex',
        action='store_true',
        help='whether to share x axis within multiple sub-figures'
    )
    parser.add_argument(
        '--sharey',
        action='store_true',
        help='whether to share y axis within multiple sub-figures'
    )
    parser.add_argument(
        '--legend-outside',
        action='store_true',
        help='place the legend outside of the figure'
    )
    parser.add_argument(
        '--xlim', type=int, default=None, help='x-axis limitation (default: None)'
    )
    parser.add_argument('--root-dir', default='./', help='root dir (default: ./)')
    parser.add_argument(
        '--file-pattern',
        type=str,
        default=r".*/test_rew_\d+seeds.csv$",
        help='regular expression to determine whether or not to include target csv '
        'file, default to including all test_rew_{num}seeds.csv file under rootdir'
    )
    parser.add_argument(
        '--group-pattern',
        type=str,
        default=r"(/|^)\w*?\-v(\d|$)",
        help='regular expression to group files in sub-figure, default to grouping '
        'according to env_name dir, "" means no grouping'
    )
    parser.add_argument(
        '--legend-pattern',
        type=str,
        default=r".*",
        help='regular expression to extract legend from csv file path, default to '
        'using file path as legend name.'
    )
    parser.add_argument('--show', action='store_true', help='show figure')
    parser.add_argument(
        '--output-path', type=str, help='figure save path', default="./figure.png"
    )
    parser.add_argument(
        '--dpi', type=int, default=200, help='figure dpi (default: 200)'
    )
    args = parser.parse_args()
    input_root_dir = "/home/jmji/logs/tpami/pc/logs"
    input_output_path = "/home/jmji/logs/tpami/pc/logs"

    task_envs = [ 'ShadowHandReOrientation', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
'ShadowHandCatchOver2Underarm', 'ShadowHandTwoCatchUnderarm',
'ShadowHandDoorOpenInward', 'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward', 'ShadowHandDoorCloseOutward',
'ShadowHandPushBlock', 
'ShadowHandScissors', 'ShadowHandSwingCup', 'ShadowHandGraspAndPlace', 'ShadowHandSwitch', 'ShadowHandBottleCap', 'ShadowHandPen', 'ShadowHandPourWater', 'ShadowHandLiftUnderarm']
    # for task_name in ["pour_water", "over", "catch_underarm", "catch_over2underarm", "catch_abreast", "two_catch_underarm", "push_block", "re_orientation", "block_stack", "grasp_and_place", "door_open_inward", "door_open_outward", "door_close_inward", "door_close_outward", "switch", "lift_underarm", "pen", "swing_cup", "scissors", "bottle_cap"]:
    for task_name in task_envs:
        print("process: " + task_name)
        # args.root_dir = input_root_dir + "_" + task_name
        # args.output_path = input_output_path + "_" + task_name + "/" + task_name + ".png"
        # args.title = input_title + "_" + task_name\
        args.root_dir = input_root_dir  + "/" + task_name
        args.output_path = args.root_dir + "/" + task_name + ".png"
        args.title = task_name[10:]
        file_lists = find_all_files(args.root_dir, re.compile(args.file_pattern))
        file_lists = [os.path.relpath(f, args.root_dir) for f in file_lists]
        if args.style:
            plt.style.use(args.style)
        os.chdir(args.root_dir)
        plot_figure(
            file_lists,
            group_pattern=args.group_pattern,
            legend_pattern=args.legend_pattern,
            fig_length=args.fig_length,
            fig_width=args.fig_width,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            xkey=args.xkey,
            ykey=args.ykey,
            xlim=args.xlim,
            sharex=args.sharex,
            sharey=args.sharey,
            smooth_radius=args.smooth,
            shaded_std=args.shaded_std,
            legend_outside=args.legend_outside
        )
        plt.grid()
        if args.output_path:
            plt.savefig(args.output_path, dpi=args.dpi, bbox_inches='tight')
        if args.show:
            plt.show()