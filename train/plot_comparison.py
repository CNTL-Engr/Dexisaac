"""
训练曲线绘制脚本

用法:
    # 对比模式：绘制 CNN vs Equivariant 对比曲线
    python plot_comparison.py --num_objects 4 --compare
    
    # 单独模式：只绘制指定模型的曲线
    python plot_comparison.py --num_objects 4 --model cnn
    python plot_comparison.py --num_objects 4 --model equi

输出:
    对比图保存到 results/comparison_plots/
    单独图保存到对应模型文件夹的 plots/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 默认结果目录：脚本所在目录的上一级目录下的 model_results
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_BASE_DIR = os.path.join(PARENT_DIR, "model_results")


def load_training_log(model_type: str, num_objects: int, base_dir: str = DEFAULT_BASE_DIR) -> pd.DataFrame:
    """
    加载训练日志CSV文件
    
    Args:
        model_type: 'cnn' 或 'equi'
        num_objects: 物体数量
        base_dir: 结果根目录
        
    Returns:
        DataFrame with columns: episode, step, loss, reward
    """
    folder_name = f"{model_type}_obj_{num_objects}"
    csv_path = os.path.join(base_dir, folder_name, "training_log.csv")
    
    if not os.path.exists(csv_path):
        print(f"⚠ 警告: 找不到文件 {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✓ 已加载 {csv_path} ({len(df)} 条记录)")
    return df


def smooth_data(data: list, window: int = 10) -> np.array:
    """计算滑动平均"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_single(model_type: str, num_objects: int, base_dir: str = DEFAULT_BASE_DIR,
                window: int = 10, x_axis: str = "episode"):
    """
    绘制单个模型的训练曲线
    
    Args:
        model_type: 'cnn' 或 'equi'
        num_objects: 物体数量
        base_dir: 结果根目录
        window: 滑动平均窗口大小
    """
    data = load_training_log(model_type, num_objects, base_dir)
    
    if data is None:
        print(f"✗ 没有找到 {model_type} 的训练数据")
        return
    
    # 创建保存目录（放在 cnn_obj_{num_objects}/ 或 equi_obj_{num_objects}/ 目录下）
    folder_name = f"{model_type}_obj_{num_objects}"
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    fig_size = (10, 6)
    color = 'blue' if model_type == 'cnn' else 'red'
    label = 'CNN' if model_type == 'cnn' else 'Equivariant'
    
    # 选择横轴：episode 或 step
    if x_axis == "episode" and "episode" in data.columns:
        x = data["episode"].values
        x_label = "Episode"
    else:
        x = data["step"].values
        x_label = "Training Step"
    
    # ============ Loss 与 Reward 同一张图（上下两个子图） ============
    fig, (ax_loss, ax_reward) = plt.subplots(2, 1, figsize=fig_size, sharex=True)

    # Loss 子图
    loss = data['loss'].values
    loss_smooth = smooth_data(loss, window)
    ax_loss.plot(x, loss, alpha=0.3, color=color, label='Loss Raw')
    ax_loss.plot(x[:len(loss_smooth)], loss_smooth,
                 color=color, linewidth=2, label=f'Loss MA{window}')
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title(f'{label} Training - {num_objects} Objects', fontsize=14)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # Reward 子图
    reward = data['reward'].values
    reward_smooth = smooth_data(reward, window)
    ax_reward.plot(x, reward, alpha=0.3, color=color, label='Reward Raw')
    ax_reward.plot(x[:len(reward_smooth)], reward_smooth,
                   color=color, linewidth=2, label=f'Reward MA{window}')
    ax_reward.set_xlabel(x_label, fontsize=12)
    ax_reward.set_ylabel('Reward', fontsize=12)
    ax_reward.legend(fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss 与 Reward 曲线已保存到同一张图: {combined_plot_path}")


def plot_comparison(num_objects: int, base_dir: str = DEFAULT_BASE_DIR,
                    window: int = 20, x_axis: str = "episode"):
    """
    绘制 CNN vs Equivariant 对比曲线
    
    Args:
        num_objects: 物体数量
        base_dir: 结果根目录
        window: 滑动平均窗口大小
    """
    # 加载数据
    cnn_data = load_training_log('cnn', num_objects, base_dir)
    equi_data = load_training_log('equi', num_objects, base_dir)
    
    if cnn_data is None and equi_data is None:
        print("✗ 没有找到任何训练数据，请先运行训练")
        return
    
    # 创建保存目录
    output_dir = os.path.join(base_dir, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    fig_size = (12, 6)

    # 选择横轴：episode 或 step
    def get_x_and_label(df):
        if x_axis == "episode" and "episode" in df.columns:
            return df["episode"].values, "Episode"
        else:
            return df["step"].values, "Training Step"
    
    # ============ Loss 与 Reward 对比，同一张图（上下两个子图） ============
    fig, (ax_loss, ax_reward) = plt.subplots(2, 1, figsize=fig_size, sharex=True)

    # Loss 子图
    x_label = "Training Step"
    if cnn_data is not None:
        steps_cnn, x_label = get_x_and_label(cnn_data)
        loss_cnn = cnn_data['loss'].values
        loss_cnn_smooth = smooth_data(loss_cnn, window)
        
        ax_loss.plot(steps_cnn, loss_cnn, alpha=0.2, color='blue', label='CNN Loss Raw')
        ax_loss.plot(steps_cnn[:len(loss_cnn_smooth)], loss_cnn_smooth, 
                     color='blue', linewidth=2, label=f'CNN Loss MA{window}')
    
    if equi_data is not None:
        steps_equi, x_label = get_x_and_label(equi_data)
        loss_equi = equi_data['loss'].values
        loss_equi_smooth = smooth_data(loss_equi, window)
        
        ax_loss.plot(steps_equi, loss_equi, alpha=0.2, color='red', label='Equivariant Loss Raw')
        ax_loss.plot(steps_equi[:len(loss_equi_smooth)], loss_equi_smooth, 
                     color='red', linewidth=2, label=f'Equivariant Loss MA{window}')
    
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title(f'Loss & Reward Comparison - {num_objects} Objects', fontsize=14)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    
    # Reward 子图
    if cnn_data is not None:
        steps_cnn, x_label = get_x_and_label(cnn_data)
        reward_cnn = cnn_data['reward'].values
        reward_cnn_smooth = smooth_data(reward_cnn, window)
        
        ax_reward.plot(steps_cnn, reward_cnn, alpha=0.2, color='blue', label='CNN Reward Raw')
        ax_reward.plot(steps_cnn[:len(reward_cnn_smooth)], reward_cnn_smooth, 
                       color='blue', linewidth=2, label=f'CNN Reward MA{window}')
    
    if equi_data is not None:
        steps_equi, x_label = get_x_and_label(equi_data)
        reward_equi = equi_data['reward'].values
        reward_equi_smooth = smooth_data(reward_equi, window)
        
        ax_reward.plot(steps_equi, reward_equi, alpha=0.2, color='red', label='Equivariant Reward Raw')
        ax_reward.plot(steps_equi[:len(reward_equi_smooth)], reward_equi_smooth, 
                       color='red', linewidth=2, label=f'Equivariant Reward MA{window}')
    
    ax_reward.set_xlabel(x_label, fontsize=12)
    ax_reward.set_ylabel('Reward', fontsize=12)
    ax_reward.legend(fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, f'training_comparison_obj{num_objects}.png')
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss 与 Reward 对比曲线已保存到同一张图: {combined_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--num_objects', '-n', type=int, required=True,
                        help='物体数量')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='对比模式：绘制 CNN vs Equivariant 对比曲线')
    parser.add_argument('--model', '-m', type=str, choices=['cnn', 'equi'],
                        help='单独模式：只绘制指定模型 (cnn 或 equi)')
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR,
                        help='训练结果根目录（默认: 脚本上一级目录下的 model_results）')
    parser.add_argument('--window', '-w', type=int, default=10,
                        help='滑动平均窗口大小')
    parser.add_argument('--x_axis', type=str, choices=['episode', 'step'], default='episode',
                        help='横轴类型: episode 或 step (默认: episode)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    
    if args.compare:
        # 对比模式
        print(f"  CNN vs Equivariant 训练曲线对比")
        print(f"  物体数量: {args.num_objects}")
        print("=" * 60)
        plot_comparison(args.num_objects, args.base_dir, args.window, args.x_axis)
    elif args.model:
        # 单独模式
        model_name = 'CNN' if args.model == 'cnn' else 'Equivariant'
        print(f"  {model_name} 训练曲线绘制")
        print(f"  物体数量: {args.num_objects}")
        print("=" * 60)
        plot_single(args.model, args.num_objects, args.base_dir, args.window, args.x_axis)
    else:
        # 默认对比模式
        print(f"  CNN vs Equivariant 训练曲线对比 (默认)")
        print(f"  物体数量: {args.num_objects}")
        print("=" * 60)
        plot_comparison(args.num_objects, args.base_dir, args.window, args.x_axis)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
