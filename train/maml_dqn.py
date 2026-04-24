"""
MAML-DQN Agent for PushNet
实现 MAML 双循环机制与象限密度任务生成器
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque
import os
import sys

# 尝试导入 functorch / torch.func 用于 fast_weights 的函数式前向传播
try:
    from torch.func import functional_call
except ImportError:
    # 兼容老版本 PyTorch
    from functorch import make_functional_with_buffers
    def functional_call(module, params_dict, args):
        params = tuple(params_dict.values())
        buffers = tuple(module.buffers())
        fmodel, _, _ = make_functional_with_buffers(module)
        return fmodel(params, buffers, *args)

# 确保能找到 src 目录中的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pushnet import EquivariantPushNet
from pushnet_cnn import CNNPushNet


class QuadrantTaskGenerator:
    """
    [功能]: 基于象限障碍物密度的 MAML 子任务生成器。

    [坐标系定义]:
      - 初始局部坐标系: 原点 = P_target, X轴 = 从 P_base 指向 P_target 的方向 (远离基座),
        Y轴 = X轴逆时针 90° (右手定则)。
      - 实际坐标系: 在初始局部坐标系的基础上，X轴和Y轴同时绕 Z 轴逆时针旋转 135° (3π/4 rad)。

    [子任务定义] (5 种拓扑分布):
      Task 0 (Q1 Blocked): Q1 稠密
      Task 1 (Q2 Blocked): Q2 稠密
      Task 2 (Q3 Blocked): Q3 稠密
      Task 3 (Q4 Blocked): Q4 稠密
      Task 4 (Dense Clutter): 四个象限均匀稠密
    """

    NUM_TASKS = 5  # 固定 5 种子任务

    # 各 Task 的名称（用于打印/日志）
    TASK_NAMES = {
        0: "Q1 Blocked",
        1: "Q2 Blocked",
        2: "Q3 Blocked",
        3: "Q4 Blocked",
        4: "Dense Clutter",
    }

    def __init__(self, radius=0.21, min_dist=0.06, max_attempts=1000):
        self.radius = radius
        self.min_dist = min_dist  # 障碍物中心之间的最小距离 (防穿模)
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------ #
    #  任务分配逻辑
    # ------------------------------------------------------------------ #
    def get_task_distribution(self, task_id, total_obstacles):
        """
        [功能]: 根据 task_id 计算各象限的障碍物数量四元组 (n1, n2, n3, n4)
        [规则]:
          - Task 0-3: 稠密象限获得 ~60% 的障碍物, 其余 3 个象限均分剩余。
          - Task 4: 四个象限均匀分配。
        [输入]: task_id (0-4), total_obstacles (int)
        [输出]: tuple (n1, n2, n3, n4)
        """
        if task_id == 4:
            # Dense Clutter: 均匀分配
            base = total_obstacles // 4
            remainder = total_obstacles % 4
            counts = [base] * 4
            # 将余数随机散布
            indices = list(range(4))
            random.shuffle(indices)
            for i in range(remainder):
                counts[indices[i]] += 1
            return tuple(counts)
        else:
            # Task 0-3: 指定象限稠密
            dense_q = task_id  # 稠密象限索引 (0-3)
            n_dense = max(1, round(total_obstacles * 0.6))
            n_rest = total_obstacles - n_dense

            counts = [0, 0, 0, 0]
            counts[dense_q] = n_dense

            # 其余 3 个象限均分剩余
            other_qs = [q for q in range(4) if q != dense_q]
            base = n_rest // 3
            remainder = n_rest % 3
            random.shuffle(other_qs)
            for i, q in enumerate(other_qs):
                counts[q] = base + (1 if i < remainder else 0)

            return tuple(counts)

    def sample_task(self, total_obstacles=4):
        """
        [功能]: 随机采样一个 task_id (0-4) 并返回其四元组分布
        [输出]: (task_id, task_tuple, task_name)
        """
        task_id = random.randint(0, self.NUM_TASKS - 1)
        task_tuple = self.get_task_distribution(task_id, total_obstacles)
        return task_id, task_tuple, self.TASK_NAMES[task_id]

    def get_task_by_id(self, task_id, total_obstacles=4):
        """
        [功能]: 获取指定 task_id 的四元组分布
        [输出]: (task_tuple, task_name)
        """
        task_tuple = self.get_task_distribution(task_id, total_obstacles)
        return task_tuple, self.TASK_NAMES[task_id]

    # ------------------------------------------------------------------ #
    #  坐标生成
    # ------------------------------------------------------------------ #
    def generate_positions_for_task(self, target_pos, robot_pos, task_tuple):
        """
        [功能]: 根据任务四元组生成具体的障碍物坐标
        [输入]: target_pos: 目标物体中心 [x_t, y_t, z_t]
                robot_pos: 机器人基座中心 [x_r, y_r, z_r]
                task_tuple: (n1, n2, n3, n4)
        [输出]: List[[x, y, z]] 障碍物的位置列表
        """
        xt, yt = target_pos[0], target_pos[1]
        xr, yr = robot_pos[0], robot_pos[1]

        # 1. 计算初始 X 轴向量 (基座 -> 目标物体, 即远离基座的方向)
        vx, vy = xt - xr, yt - yr
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            vx, vy = 1.0, 0.0
            norm = 1.0

        ux, uy = vx / norm, vy / norm  # 初始 X 轴

        # 2. 坐标系旋转 135° (逆时针)
        angle = math.radians(135)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # 任务坐标系 X 轴 (X_task)
        x_task_dx = ux * cos_a - uy * sin_a
        x_task_dy = ux * sin_a + uy * cos_a

        # 任务坐标系 Y 轴 (Y_task) = X_task 逆时针 90°
        y_task_dx = -x_task_dy
        y_task_dy = x_task_dx

        # ---- 辅助函数: 在象限扇形内均匀采样 ---- #
        def sample_point_in_quadrant(q_idx):
            # 面积均匀: r = sqrt(U) * R, 留出 0.05m 内圈
            r = math.sqrt(random.uniform((0.05 / self.radius) ** 2, 1.0)) * self.radius

            # 象限角度范围 (每象限 90°)
            base_angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
            local_angle = base_angles[q_idx] + random.uniform(0, math.pi / 2)

            local_x = r * math.cos(local_angle)
            local_y = r * math.sin(local_angle)

            global_x = xt + local_x * x_task_dx + local_y * y_task_dx
            global_y = yt + local_x * x_task_dy + local_y * y_task_dy

            return [global_x, global_y, target_pos[2]]

        # 3. 按四元组生成所有点，带防碰撞检测
        obstacle_positions = []
        existing_pts = [[xt, yt]]  # 目标物体本身也算已占用

        for q_idx, count in enumerate(task_tuple):
            for _ in range(count):
                placed = False
                for _ in range(self.max_attempts):
                    pt = sample_point_in_quadrant(q_idx)

                    collision = any(
                        math.hypot(pt[0] - ex[0], pt[1] - ex[1]) < self.min_dist
                        for ex in existing_pts
                    )
                    if not collision:
                        obstacle_positions.append(pt)
                        existing_pts.append([pt[0], pt[1]])
                        placed = True
                        break

                if not placed:
                    print(f"[TaskGenerator] 警告：在象限 Q{q_idx+1} 找不到防碰撞的安放点！")
                    obstacle_positions.append(pt)
                    existing_pts.append([pt[0], pt[1]])

        return obstacle_positions


class TaskReplayBuffer:
    """
    [功能]: 存储特定 Task 的经验，区分 Support Set 和 Query Set
    """
    def __init__(self, capacity=1000):
        # 结构: { task_id: { 'support': [], 'query': [] } }
        self.buffer = {}
        self.capacity = capacity
        self.task_ids = deque(maxlen=capacity)

    def push(self, task_id, is_support, state, action, reward, next_state, done):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)
            
        if task_id not in self.buffer:
            if len(self.task_ids) >= self.capacity:
                oldest_task = self.task_ids.popleft()
                del self.buffer[oldest_task]
            self.buffer[task_id] = {'support': [], 'query': []}
            self.task_ids.append(task_id)
            
        dataset_type = 'support' if is_support else 'query'
        self.buffer[task_id][dataset_type].append((state, action, reward, next_state, done))

    def sample_tasks(self, batch_size):
        """
        采样多个 task 的数据
        [输出]: List[Dict] 每个元素包含一个 task 的 support 和 query 数据集
        """
        valid_tasks = [tid for tid in self.task_ids 
                       if len(self.buffer[tid]['support']) > 0 and len(self.buffer[tid]['query']) > 0]
        
        if len(valid_tasks) < batch_size:
            sampled_tids = valid_tasks
        else:
            sampled_tids = random.sample(valid_tasks, batch_size)
            
        sampled_data = []
        for tid in sampled_tids:
            sampled_data.append(self.buffer[tid])
        return sampled_data
    
    def __len__(self):
        return len(self.task_ids)


class MAMLDQNAgent:
    """
    [功能]: MAML DQN Agent，支持等变网络，使用 torch.autograd.grad 手动更新 fast_weights。
    """
    def __init__(self, device='cuda', lr=1e-4, inner_lr=1e-3, gamma=0.99, use_equivariant=True):
        self.device = device
        self.gamma = gamma
        self.inner_lr = inner_lr
        self.use_equivariant = use_equivariant
        
        if use_equivariant:
            self.policy_net = EquivariantPushNet().to(device)
        else:
            self.policy_net = CNNPushNet().to(device)
            
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = TaskReplayBuffer(capacity=5000)
        
    def select_action(self, state, epsilon, invalid_actions=None, fast_weights=None):
        """
        [功能]: MAML 阶段的环境交互。如果有 fast_weights，则用适应后的权重推理。
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            if state.dtype == torch.uint8:
                state_float = state.float().to(self.device) / 255.0
            else:
                state_float = state.to(self.device)
                
        if random.random() > epsilon:
            strategy_type = 'exploit'
            with torch.no_grad():
                # 判断是否使用 task-specific fast weights
                if fast_weights is not None:
                    # 使用 torch.func 进行无状态前向传播
                    q_values = functional_call(self.policy_net, fast_weights, (state_float,))
                else:
                    q_values = self.policy_net(state_float)
                
                if invalid_actions:
                    for action_idx in invalid_actions:
                        q_values[0, action_idx] = -float('inf')
                
                action_idx = q_values.argmax(dim=1).item()
        else:
            strategy_type = 'explore'
            available_actions = [i for i in range(8)]
            if invalid_actions:
                available_actions = [i for i in available_actions if i not in invalid_actions]
            if not available_actions:
                action_idx = np.random.randint(0, 8)
            else:
                action_idx = int(np.random.choice(available_actions))
                
        return action_idx, strategy_type

    def adapt(self, support_transitions, inner_steps=1, first_order=False):
        """
        [功能]: Inner-loop Adaptation
        [输入]: support_transitions: Support Set 经验
                first_order: 如果 True，不计算二阶导数 (FOMAML)
        [输出]: fast_weights: 字典形式的模型参数
        """
        if not support_transitions:
            # 如果没收集到Support数据，返回初始权重
            return {name: param for name, param in self.policy_net.named_parameters()}
            
        states, actions, rewards, next_states, dones = zip(*support_transitions)
        
        states = torch.cat(states).to(self.device).float() / 255.0
        next_states = torch.cat(next_states).to(self.device).float() / 255.0
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # 初始化 fast_weights 为当前模型的参数
        fast_weights = {name: param for name, param in self.policy_net.named_parameters()}
        
        for _ in range(inner_steps):
            # 1. 计算 Q(s, a; fast_weights)
            q_pred = functional_call(self.policy_net, fast_weights, (states,))
            q_values = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 2. 计算 Target Q
            # Target Network 保持冻结 (不进行 Inner Loop 的梯度流追踪，以防不稳定)
            with torch.no_grad():
                # Double DQN 机制：用当前的 fast_weights 选择最优动作
                q_next_policy = functional_call(self.policy_net, fast_weights, (next_states,))
                next_actions = q_next_policy.max(dim=1)[1]
                
                # 用冻结的 Target Net 评估该动作
                q_next_target = self.target_net(next_states)
                max_q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                targets = rewards + (1 - dones) * self.gamma * max_q_next
            
            # 3. Inner Loss
            inner_loss = F.mse_loss(q_values, targets)
            
            # 4. 手动求导更新 fast_weights
            # create_graph=True 开启二阶导数用于 Outer-loop backprop
            grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=not first_order, allow_unused=True)
            
            new_fast_weights = {}
            for (name, param), grad in zip(fast_weights.items(), grads):
                if grad is not None:
                    new_fast_weights[name] = param - self.inner_lr * grad
                else:
                    new_fast_weights[name] = param
            fast_weights = new_fast_weights
            
        return fast_weights

    def meta_update(self, task_batch, inner_steps=1, first_order=False):
        """
        [功能]: Outer-loop Meta Update
        [返回]: meta_loss 的均值
        """
        if not task_batch:
            return 0.0
            
        meta_loss = 0.0
        self.optimizer.zero_grad()
        
        for task_data in task_batch:
            support_set = task_data['support']
            query_set = task_data['query']
            
            # --- 1. Inner Loop: 得到 Task-specific 的权重 ---
            fast_weights = self.adapt(support_set, inner_steps=inner_steps, first_order=first_order)
            
            # --- 2. Outer Loop: 在 Query Set 上评估 ---
            if not query_set:
                continue
                
            q_states, q_actions, q_rewards, q_next_states, q_dones = zip(*query_set)
            
            q_states = torch.cat(q_states).to(self.device).float() / 255.0
            q_next_states = torch.cat(q_next_states).to(self.device).float() / 255.0
            q_actions = torch.tensor(q_actions, dtype=torch.long, device=self.device)
            q_rewards = torch.tensor(q_rewards, dtype=torch.float32, device=self.device)
            q_dones = torch.tensor(q_dones, dtype=torch.float32, device=self.device)
            
            # 前向传播使用适应后的 fast_weights
            q_pred = functional_call(self.policy_net, fast_weights, (q_states,))
            q_values = q_pred.gather(1, q_actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                q_next_policy = functional_call(self.policy_net, fast_weights, (q_next_states,))
                next_actions = q_next_policy.max(dim=1)[1]
                q_next_target = self.target_net(q_next_states)
                max_q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = q_rewards + (1 - q_dones) * self.gamma * max_q_next
                
            task_meta_loss = F.mse_loss(q_values, targets)
            
            # [显存优化] 每次 Task 直接计算梯度并累加，释放中间激活张量的显存
            task_meta_loss_scaled = task_meta_loss / len(task_batch)
            task_meta_loss_scaled.backward()
            
            meta_loss += task_meta_loss.item()
            
            # 释放 Query Set 推理产生的张量
            del q_states, q_next_states, q_actions, q_rewards, q_dones
            del q_pred, q_values, q_next_policy, next_actions, q_next_target, max_q_next, targets
            del fast_weights
            torch.cuda.empty_cache()
            
        if meta_loss == 0.0:
            return 0.0
            
        # 聚合所有 task 的 meta loss，对初始 Meta-Policy 参数进行梯度下降
        meta_loss_avg = meta_loss / len(task_batch)
        
        # 反向传播已经在循环中完成，直接裁剪梯度和步进
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return meta_loss_avg
        
    def update_target_network(self):
        with torch.no_grad():
            old_target = self.target_net
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.eval()
            del old_target

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
