"""
DQN Agent for PushNet

支持等变和非等变两种网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import sys
import os
import copy  # [新增] 用于deepcopy

# 添加 src 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, src_path)

from pushnet import EquivariantPushNet
from pushnet_cnn import CNNPushNet


class DQNAgent:
    """
    [功能]: DQN Agent，支持等变和非等变两种Q网络
    
    [描述]: 网络输出统一的 Q 值图，通过 (u, v, θ) 的 Q 值决定推动策略
    """
    def __init__(self, device='cuda', lr=1e-4, gamma=0.99, buffer_capacity=50000, use_equivariant=True):
        """
        [输入]: device: 'cuda' or 'cpu'
                lr: 学习率
                gamma: 折扣因子
                buffer_capacity: 经验回放缓冲区容量
                use_equivariant: 是否使用等变网络
        """
        self.device = device
        self.gamma = gamma
        self.use_equivariant = use_equivariant
        
        # 根据参数选择网络类型
        if use_equivariant:
            # Policy Network 初始化（等变）
            self.policy_net = EquivariantPushNet().to(device)
        else:
            # Policy Network 初始化（非等变CNN）
            self.policy_net = CNNPushNet().to(device)
        
        # Target Network 初始化
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()  # 设为评估模式
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 混合精度训练 Scaling
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
      
        
        print(f"  - Q 网络: {'EquivariantPushNet (C4等变)' if use_equivariant else 'CNNPushNet (非等变)'}")
        print(f"  - 设备: {device}, AMP混合精度: 开启")
        print(f"  - 学习率: {lr}")
        print(f"  - Gamma: {gamma}")
        
    def select_action(self, state, epsilon, invalid_actions=None, env_idx=0, debug=False):
        """
        [功能]: 选择离散动作 (Epsilon-Greedy)
        [输入]: state: (1, 3, 320, 320) tensor (uint8)
                epsilon: 探索率
                invalid_actions: 无效动作列表
        [输出]: (action_idx, strategy_type)
                action_idx: int (0-7)
                strategy_type: str ('exploit' 或 'explore')
                - 0-3: 推目标物体
                - 4-7: 推障碍物
        """
        # 确保 state 有 batch 维度
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        # 统一处理状态转换
        with torch.no_grad():
            if state.dtype == torch.uint8:
                state_float = state.float().to(self.device) / 255.0
            else:
                state_float = state.to(self.device)
        
        # Epsilon-Greedy 策略
        if random.random() > epsilon:
            # 利用：选择Q值最大的动作
            strategy_type = 'exploit'
            with torch.no_grad():
                # [利用时] 选择Q值最大的动作
                q_values = self.policy_net(state_float)  # (1, 8)
                
                # [调试] 打印Q值
                if 'debug' in locals() or env_idx == 0:  # 简化：只打印env 0
                    pass  # 先不打印，避免混乱
                
                # [IAS] Mask invalid actions
                if invalid_actions:
                    for action_idx in invalid_actions:
                        q_values[0, action_idx] = -float('inf')
                
                action_idx = q_values.argmax(dim=1).item()  # 0-7
                
                # [激进内存优化] 立即释放q_values和state_float
                del q_values, state_float
                torch.cuda.empty_cache()
        else:
            # 探索：随机选择，但避免无效动作
            strategy_type = 'explore'
            
            # [内存优化] 探索时不调用网络，直接随机选择
            # 立即释放 state_float
            del state_float
            torch.cuda.empty_cache()
            
            available_actions = [i for i in range(8)]
            if invalid_actions:
                available_actions = [i for i in available_actions if i not in invalid_actions]
            
            if not available_actions:
                # 如果所有动作都无效（极端情况），随机选一个
                action_idx = np.random.randint(0, 8)
            else:
                # [修复] 使用np.random.choice确保真正的均匀随机
                action_idx = int(np.random.choice(available_actions))
        
        return action_idx, strategy_type

    def store_transition(self, state, action, reward, next_state, done):
        """
        [功能]: 存储经验到回放缓冲区
        [输入]: state: (1, 3, 320, 320) 或 (3, 320, 320)
                action: int (0-7)
                reward: float
                next_state: (1, 3, 320, 320) 或 (3, 320, 320)
                done: bool
        """
        # 确保 state 有 batch 维度
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)
        
        # [内存优化] 如果已经是CPU uint8，直接使用；否则转换
        if state.device.type != 'cpu':
            state = state.cpu()
        if next_state.device.type != 'cpu':
            next_state = next_state.cpu()
            
        if state.dtype != torch.uint8:
            state = (state * 255).to(torch.uint8)
        if next_state.dtype != torch.uint8:
            next_state = (next_state * 255).to(torch.uint8)
        
        # 存储到缓冲区
        self.replay_buffer.push(
            state.contiguous(), 
            action, 
            reward, 
            next_state.contiguous(), 
            done
        )
    
    def train_step(self, batch_size=32):
        """
        [功能]: 执行一次训练步骤
        [输入]: batch_size: int
        [输出]: loss: float or None (如果缓冲区样本不足)
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # 采样 mini-batch (uint8 on CPU)
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # 转换为 tensor 并移至 GPU
        states = torch.cat(states).to(self.device).float() / 255.0           # (B, 3, 320, 320)
        next_states = torch.cat(next_states).to(self.device).float() / 255.0
        
        # actions现在是整数索引(0-7)，不再是(u,v,direction)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)  # (B,)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # 混合精度上下文
        with torch.cuda.amp.autocast():
            # 1. 计算当前动作的 Q 值
            q_pred = self.policy_net(states)  # (B, 8)
            
            # 提取对应动作的 Q 值
            q_values = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
            
            # 2. 计算目标 Q 值（Double DQN）
            with torch.no_grad():
                # Double DQN: 用policy net选择action，target net评估Q值
                q_next_policy = self.policy_net(next_states)  # (B, 8)
                next_actions = q_next_policy.max(dim=1)[1]  # (B,) 选择最大Q值的action
                
                # [内存优化] 立即释放 q_next_policy
                del q_next_policy
                
                q_next_target = self.target_net(next_states)  # (B, 8)
                
                # [内存优化] next_states 不再需要，立即释放
                del next_states
                torch.cuda.empty_cache()
                
                # 用target net评估policy net选择的action
                max_q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
                
                # [内存优化] 保存统计用的值到CPU
                if hasattr(self, 'train_step_count') and (self.train_step_count + 1) % 50 == 0:
                    target_sample_cpu = q_next_target[0].cpu().numpy().copy()
                    target_stats = (q_next_target.mean().item(), q_next_target.std().item(), 
                                    q_next_target.min().item(), q_next_target.max().item())
                else:
                    target_sample_cpu = None
                    target_stats = None
                
                # [内存优化] 释放 next_actions 和 q_next_target
                del next_actions, q_next_target
                
                targets = rewards + (1 - dones) * self.gamma * max_q_next
                del max_q_next
            
            # 3. 计算损失
            loss = F.mse_loss(q_values, targets)
            del targets
            
            # [激进内存优化] 立即保存loss值
            loss_value = loss.item()
        
        # [Debug] Q值分布统计 - 使用已保存的CPU数据
        if hasattr(self, 'train_step_count'):
            self.train_step_count += 1
        else:
            self.train_step_count = 0
        
        if self.train_step_count % 50 == 0 and target_sample_cpu is not None:
            with torch.no_grad():
                # 使用已保存的CPU数据，避免额外GPU访问
                q_sample = q_pred[0].cpu().numpy()
                q_values_list = [f"{q:.2f}" for q in q_sample]
                
                policy_mean = q_pred.mean().item()
                policy_std = q_pred.std().item()
                policy_min = q_pred.min().item()
                policy_max = q_pred.max().item()
                
                target_values_list = [f"{q:.2f}" for q in target_sample_cpu]
                target_net_mean, target_net_std, target_net_min, target_net_max = target_stats
                
                # 打印统计
                print(f"\n{'='*70}")
                print(f"[Q值统计 Step {self.train_step_count}]")
                print(f"{'='*70}")
                print(f"\n【在线网络 Policy Net】")
                print(f"  Q值表: [{', '.join(q_values_list)}]")
                print(f"  Mean: {policy_mean:+.3f} | Std: {policy_std:.3f} | Range: [{policy_min:+.3f}, {policy_max:+.3f}]")
                print(f"\n【目标网络 Target Net】")
                print(f"  Q值表: [{', '.join(target_values_list)}]")
                print(f"  Mean: {target_net_mean:+.3f} | Std: {target_net_std:.3f} | Range: [{target_net_min:+.3f}, {target_net_max:+.3f}]")
                print(f"\n【训练Loss】 {loss_value:.4f}")
                print(f"{'='*70}\n")
        
        # 4. 反向传播 (使用 Scaler)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪 (Unscale first)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # [内存优化] 清理剩余tensor
        del q_pred, q_values, states, actions, rewards, dones, loss
        torch.cuda.empty_cache()
        
        return loss_value
    
    def update_target_network(self):
        """
        [功能]: 硬更新：完全复制 policy network 到 target network
        """
        with torch.no_grad():
            # 使用deepcopy完整复制网络
            old_target = self.target_net
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.eval()  # 保持评估模式
            del old_target  # 删除旧的target_net释放内存
    
    def save(self, path):
        """
        [功能]: 保存模型
        [输入]: path: 保存路径
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"[DQN Agent] 模型已保存到: {path}")
    
    def load(self, path):
        """
        [功能]: 加载模型
        [输入]: path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[DQN Agent] 模型已从 {path} 加载")


class ReplayBuffer:
    """
    [功能]: 经验回放缓冲区
    """
    def __init__(self, capacity=50000):
        """
        [输入]: capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        [功能]: 存储一条经验
        [输入]: state: tensor
                action: (u, v, direction) tuple
                reward: float
                next_state: tensor
                done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        [功能]: 随机采样 batch
        [输入]: batch_size: 批大小
        [输出]: tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

 
