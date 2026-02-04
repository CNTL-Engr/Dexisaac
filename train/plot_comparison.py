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


def load_training_log(model_type: str, num_objects: int, base_dir: str = "results") -> pd.DataFrame:
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


def plot_single(model_type: str, num_objects: int, base_dir: str = "results", window: int = 10):
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
    
    # 创建保存目录
    folder_name = f"{model_type}_obj_{num_objects}"
    output_dir = os.path.join(base_dir, folder_name, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    fig_size = (10, 5)
    color = 'blue' if model_type == 'cnn' else 'red'
    label = 'CNN' if model_type == 'cnn' else 'Equivariant'
    
    steps = data['step'].values
    
    # ============ 1. Loss 曲线 ============
    plt.figure(figsize=fig_size)
    loss = data['loss'].values
    loss_smooth = smooth_data(loss, window)
    
    plt.plot(steps, loss, alpha=0.3, color=color, label='Raw')
    plt.plot(steps[:len(loss_smooth)], loss_smooth, 
             color=color, linewidth=2, label=f'MA{window}')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{label} Loss - {num_objects} Objects', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    loss_plot_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss 曲线已保存: {loss_plot_path}")
    
    # ============ 2. Reward 曲线 ============
    plt.figure(figsize=fig_size)
    reward = data['reward'].values
    reward_smooth = smooth_data(reward, window)
    
    plt.plot(steps, reward, alpha=0.3, color=color, label='Raw')
    plt.plot(steps[:len(reward_smooth)], reward_smooth, 
             color=color, linewidth=2, label=f'MA{window}')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(f'{label} Reward - {num_objects} Objects', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    reward_plot_path = os.path.join(output_dir, 'reward_curve.png')
    plt.savefig(reward_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Reward 曲线已保存: {reward_plot_path}")


def plot_comparison(num_objects: int, base_dir: str = "results", window: int = 20):
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
    fig_size = (12, 5)
    
    # ============ 1. Loss 对比图 ============
    plt.figure(figsize=fig_size)
    
    if cnn_data is not None:
        steps_cnn = cnn_data['step'].values
        loss_cnn = cnn_data['loss'].values
        loss_cnn_smooth = smooth_data(loss_cnn, window)
        
        plt.plot(steps_cnn, loss_cnn, alpha=0.2, color='blue', label='_nolegend_')
        plt.plot(steps_cnn[:len(loss_cnn_smooth)], loss_cnn_smooth, 
                 color='blue', linewidth=2, label=f'CNN (MA{window})')
    
    if equi_data is not None:
        steps_equi = equi_data['step'].values
        loss_equi = equi_data['loss'].values
        loss_equi_smooth = smooth_data(loss_equi, window)
        
        plt.plot(steps_equi, loss_equi, alpha=0.2, color='red', label='_nolegend_')
        plt.plot(steps_equi[:len(loss_equi_smooth)], loss_equi_smooth, 
                 color='red', linewidth=2, label=f'Equivariant (MA{window})')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss Comparison - {num_objects} Objects', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    loss_plot_path = os.path.join(output_dir, f'loss_comparison_obj{num_objects}.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss 对比图已保存: {loss_plot_path}")
    
    # ============ 2. Reward 对比图 ============
    plt.figure(figsize=fig_size)
    
    if cnn_data is not None:
        steps_cnn = cnn_data['step'].values
        reward_cnn = cnn_data['reward'].values
        reward_cnn_smooth = smooth_data(reward_cnn, window)
        
        plt.plot(steps_cnn, reward_cnn, alpha=0.2, color='blue', label='_nolegend_')
        plt.plot(steps_cnn[:len(reward_cnn_smooth)], reward_cnn_smooth, 
                 color='blue', linewidth=2, label=f'CNN (MA{window})')
    
    if equi_data is not None:
        steps_equi = equi_data['step'].values
        reward_equi = equi_data['reward'].values
        reward_equi_smooth = smooth_data(reward_equi, window)
        
        plt.plot(steps_equi, reward_equi, alpha=0.2, color='red', label='_nolegend_')
        plt.plot(steps_equi[:len(reward_equi_smooth)], reward_equi_smooth, 
                 color='red', linewidth=2, label=f'Equivariant (MA{window})')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(f'Reward Comparison - {num_objects} Objects', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    reward_plot_path = os.path.join(output_dir, f'reward_comparison_obj{num_objects}.png')
    plt.savefig(reward_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Reward 对比图已保存: {reward_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--num_objects', '-n', type=int, required=True,
                        help='物体数量')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='对比模式：绘制 CNN vs Equivariant 对比曲线')
    parser.add_argument('--model', '-m', type=str, choices=['cnn', 'equi'],
                        help='单独模式：只绘制指定模型 (cnn 或 equi)')
    parser.add_argument('--base_dir', type=str, default='results',
                        help='训练结果根目录')
    parser.add_argument('--window', '-w', type=int, default=10,
                        help='滑动平均窗口大小')
    
    args = parser.parse_args()
    
    print("=" * 60)
    
    if args.compare:
        # 对比模式
        print(f"  CNN vs Equivariant 训练曲线对比")
        print(f"  物体数量: {args.num_objects}")
        print("=" * 60)
        plot_comparison(args.num_objects, args.base_dir, args.window)
    elif args.model:
        # 单独模式
        model_name = 'CNN' if args.model == 'cnn' else 'Equivariant'
        print(f"  {model_name} 训练曲线绘制")
        print(f"  物体数量: {args.num_objects}")
        print("=" * 60)
        plot_single(args.model, args.num_objects, args.base_dir, args.window)
    else:
        # 默认对比模式
        print(f"  CNN vs Equivariant 训练曲线对比 (默认)")
        print(f"  物体数量: {args.num_objects}")
        print("=" * 60)
        plot_comparison(args.num_objects, args.base_dir, args.window)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
