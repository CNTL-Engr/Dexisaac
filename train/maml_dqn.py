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


class ObstacleCountTaskGenerator:
    """
    [功能]: 基于"障碍物总数量"递增的 MAML 子任务生成器。

    通过逐步增加场上障碍物的数量来划分子任务难度，
    让 Agent 从简单场景（少量障碍物）逐步过渡到复杂场景（多量障碍物）。

    [子任务定义] (4 种难度等级):
      Task 0: 生成 4 个障碍物 (最简单)
      Task 1: 生成 5 个障碍物
      Task 2: 生成 6 个障碍物
      Task 3: 生成 7 个障碍物 (最复杂)

    [模型文件夹]:
      每个子任务可以指定独立的障碍物模型文件夹和目标物体模型文件夹。
      通过构造函数的 model_dirs 参数或 set_task_model_dirs 方法设置。

    [坐标生成]:
      使用纯 Numpy + math 库，以目标物体为圆心，在半径范围内
      均匀随机采样障碍物坐标，并带有防碰撞检测。
    """

    NUM_TASKS = 4  # 固定 4 种子任务（Task 0-3）

    # 障碍物数量基数：Task 0 对应 4 个障碍物
    BASE_OBSTACLE_COUNT = 4

    # 各 Task 的名称（用于打印/日志）
    TASK_NAMES = {
        0: "4 Obstacles",
        1: "5 Obstacles",
        2: "6 Obstacles",
        3: "7 Obstacles",
    }

    # 默认模型文件夹路径
    _DEFAULT_OBSTACLE_DIR = "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_CH"
    _DEFAULT_TARGET_DIR = "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata_target"

    def __init__(self, radius=0.21, min_dist=0.06, max_attempts=1000, model_dirs=None):
        """
        [参数]:
          radius       : 障碍物分布的最大半径（以目标物体为圆心）
          min_dist     : 障碍物中心之间的最小间距（防穿模）
          max_attempts : 单个障碍物放置时的最大尝试次数
          model_dirs   : 每个子任务使用的模型文件夹配置 (dict 或 None)
                         格式: { task_id: { 'obstacle_dir': '路径', 'target_dir': '路径' }, ... }
                         未指定的 task_id 将使用默认路径。
                         也可以只指定其中一个 key（obstacle_dir 或 target_dir），另一个使用默认值。

        [示例]:
          model_dirs = {
              0: {'obstacle_dir': '/path/to/simple_models',  'target_dir': '/path/to/targets'},
              1: {'obstacle_dir': '/path/to/medium_models', 'target_dir': '/path/to/targets'},
              2: {'obstacle_dir': '/path/to/complex_models'},  # target_dir 使用默认
          }
        """
        self.radius = radius
        self.min_dist = min_dist
        self.max_attempts = max_attempts

        # 初始化每个 Task 的模型文件夹配置
        self._task_model_dirs = {}
        for tid in range(self.NUM_TASKS):
            self._task_model_dirs[tid] = {
                'obstacle_dir': self._DEFAULT_OBSTACLE_DIR,
                'target_dir': self._DEFAULT_TARGET_DIR,
            }

        # 如果用户传入了自定义配置，合并覆盖默认值
        if model_dirs is not None:
            self.set_task_model_dirs(model_dirs)

    # ------------------------------------------------------------------ #
    #  核心方法：根据 task_id 计算障碍物数量
    # ------------------------------------------------------------------ #
    def get_obstacle_count(self, task_id):
        """
        [功能]: 根据 task_id 计算需要生成的障碍物数量 N
        [公式]: N = BASE_OBSTACLE_COUNT + task_id
        [输入]: task_id (int, 0-3)
        [输出]: int, 障碍物数量

        [边界情况]:
          - task_id 超出 [0, NUM_TASKS-1] 范围时，自动裁剪到合法区间
        """
        # 边界检查：将 task_id 限制在有效范围内
        clamped_id = max(0, min(task_id, self.NUM_TASKS - 1))
        if clamped_id != task_id:
            print(f"[ObstacleCountTaskGenerator] 警告: task_id={task_id} 超出范围 [0, {self.NUM_TASKS - 1}]，"
                  f"已裁剪为 {clamped_id}")

        # 障碍物数量 = 基数 + task_id
        return self.BASE_OBSTACLE_COUNT + clamped_id

    # ------------------------------------------------------------------ #
    #  任务采样接口（兼容原有 MAML 训练循环）
    # ------------------------------------------------------------------ #
    def sample_task(self, **kwargs):
        """
        [功能]: 随机采样一个 task_id (0-3) 并返回障碍物数量
        [输出]: (task_id, obstacle_count, task_name)
                - task_id: 子任务编号 (0-3)
                - obstacle_count: 对应的障碍物数量 (4-7)
                - task_name: 任务名称字符串
        [说明]: **kwargs 用于兼容旧接口（如 total_obstacles 参数），实际不使用
        """
        task_id = random.randint(0, self.NUM_TASKS - 1)
        obstacle_count = self.get_obstacle_count(task_id)
        return task_id, obstacle_count, self.TASK_NAMES[task_id]

    def get_task_by_id(self, task_id):
        """
        [功能]: 获取指定 task_id 的障碍物数量和名称
        [输入]: task_id (int, 0-3)
        [输出]: (obstacle_count, task_name)
        """
        obstacle_count = self.get_obstacle_count(task_id)
        return obstacle_count, self.TASK_NAMES[task_id]

    # ------------------------------------------------------------------ #
    #  模型文件夹管理
    # ------------------------------------------------------------------ #
    def set_task_model_dirs(self, model_dirs):
        """
        [功能]: 设置/更新各子任务使用的模型文件夹
        [输入]: model_dirs (dict)
                格式: { task_id: { 'obstacle_dir': '路径', 'target_dir': '路径' }, ... }
                只需传入需要修改的 task_id，未传入的保持不变。
                每个 task_id 的 dict 中也只需传入需要修改的 key。
        """
        for tid, dirs in model_dirs.items():
            tid = int(tid)
            if tid < 0 or tid >= self.NUM_TASKS:
                print(f"[ObstacleCountTaskGenerator] 警告: model_dirs 中的 task_id={tid} "
                      f"超出范围 [0, {self.NUM_TASKS - 1}]，已忽略")
                continue
            if 'obstacle_dir' in dirs:
                self._task_model_dirs[tid]['obstacle_dir'] = dirs['obstacle_dir']
            if 'target_dir' in dirs:
                self._task_model_dirs[tid]['target_dir'] = dirs['target_dir']

    def get_model_dirs(self, task_id):
        """
        [功能]: 获取指定 task_id 使用的模型文件夹路径
        [输入]: task_id (int, 0-3)
        [输出]: dict, {'obstacle_dir': '路径', 'target_dir': '路径'}
        """
        clamped_id = max(0, min(task_id, self.NUM_TASKS - 1))
        return self._task_model_dirs[clamped_id].copy()

    def print_model_dirs_config(self):
        """
        [功能]: 打印所有子任务的模型文件夹配置（用于调试/日志）
        """
        print("[ObstacleCountTaskGenerator] 各子任务模型文件夹配置:")
        for tid in range(self.NUM_TASKS):
            dirs = self._task_model_dirs[tid]
            print(f"  Task {tid} ({self.TASK_NAMES[tid]}):")
            print(f"    障碍物模型: {dirs['obstacle_dir']}")
            print(f"    目标物体模型: {dirs['target_dir']}")

    # ------------------------------------------------------------------ #
    #  坐标生成（纯 Numpy + math 实现）
    # ------------------------------------------------------------------ #
    def generate_positions_for_task(self, target_pos, robot_pos, obstacle_count):
        """
        [功能]: 根据障碍物数量，以目标物体为圆心，在环形区域内
                均匀随机生成障碍物的 2D 坐标，并附带防碰撞检测。

        [输入]:
          target_pos     : 目标物体中心位置 [x_t, y_t, z_t] (list 或 np.ndarray)
          robot_pos      : 机器人基座中心位置 [x_r, y_r, z_r] (list 或 np.ndarray)
          obstacle_count : 需要生成的障碍物数量 N (int)

        [输出]:
          List[ [x, y, z] ]  —— 标准 Python 列表，内含 list 形式的 2D+Z 坐标

        [算法细节]:
          1. 以目标物体 (x_t, y_t) 为圆心
          2. 在 [r_inner, r_outer] 的环形区域内面积均匀采样
          3. 角度在 [0, 2π) 上均匀分布
          4. 防碰撞：新点与已有点的欧氏距离 >= min_dist
          5. 若单点放置失败（超过 max_attempts 次），打印警告并强制放置

        [边界情况]:
          - obstacle_count <= 0 时，返回空列表
          - obstacle_count 极大导致空间不足时，会打印警告
        """
        # ---- 边界检查：障碍物数量非正时直接返回空列表 ---- #
        if obstacle_count <= 0:
            print("[ObstacleCountTaskGenerator] 警告: obstacle_count <= 0，返回空列表")
            return []

        # ---- 提取目标物体的 XY 坐标和 Z 高度 ---- #
        xt = float(target_pos[0])
        yt = float(target_pos[1])
        zt = float(target_pos[2])  # 障碍物的 Z 高度与目标物体保持一致

        # ---- 采样参数 ---- #
        r_inner = 0.05   # 内圈安全距离 (米), 避免与目标物体重叠
        r_outer = self.radius  # 外圈最大半径

        # ---- 辅助函数: 在环形区域内面积均匀采样单个点 ---- #
        def _sample_one_point():
            """
            [功能]: 在以 (xt, yt) 为圆心的环形区域内采样一个点
            [算法]: 面积均匀采样 r = sqrt(U * (R^2 - r_inner^2) + r_inner^2)
                    角度 θ ~ Uniform(0, 2π)
            [输出]: numpy.ndarray, shape=(2,), 全局 XY 坐标
            """
            # 面积均匀采样半径
            u = np.random.uniform(0.0, 1.0)
            r = math.sqrt(u * (r_outer ** 2 - r_inner ** 2) + r_inner ** 2)

            # 均匀采样角度
            theta = np.random.uniform(0.0, 2.0 * math.pi)

            # 转换为直角坐标（全局坐标系）
            px = xt + r * math.cos(theta)
            py = yt + r * math.sin(theta)

            return np.array([px, py])

        # ---- 防碰撞检测函数 ---- #
        def _has_collision(new_pt_xy, existing_pts_xy):
            """
            [功能]: 检查新点是否与已有点集中的任何点碰撞
            [输入]: new_pt_xy   : numpy.ndarray, shape=(2,)
                    existing_pts_xy : list of numpy.ndarray, 每个 shape=(2,)
            [输出]: bool, True 表示存在碰撞
            """
            for ex_pt in existing_pts_xy:
                dist = np.linalg.norm(new_pt_xy - ex_pt)
                if dist < self.min_dist:
                    return True
            return False

        # ---- 主循环：逐个放置障碍物 ---- #
        obstacle_positions = []         # 最终输出：[x, y, z] 列表
        existing_xy = [np.array([xt, yt])]  # 已占用的 XY 坐标列表（包含目标物体）

        for i in range(obstacle_count):
            placed = False
            last_pt_xy = None

            for _ in range(self.max_attempts):
                pt_xy = _sample_one_point()
                last_pt_xy = pt_xy

                # 防碰撞检测
                if not _has_collision(pt_xy, existing_xy):
                    # 无碰撞，成功放置
                    obstacle_positions.append([float(pt_xy[0]), float(pt_xy[1]), zt])
                    existing_xy.append(pt_xy)
                    placed = True
                    break

            # 如果超过最大尝试次数仍未找到合法位置，强制放置最后一次采样的点
            if not placed:
                print(f"[ObstacleCountTaskGenerator] 警告: 第 {i+1}/{obstacle_count} 个障碍物"
                      f"在 {self.max_attempts} 次尝试后仍未找到无碰撞位置，强制放置！")
                if last_pt_xy is not None:
                    obstacle_positions.append([float(last_pt_xy[0]), float(last_pt_xy[1]), zt])
                    existing_xy.append(last_pt_xy)
                else:
                    # 极端兜底：放在目标物体正上方偏移位置
                    fallback_xy = np.array([xt + r_inner * 2, yt])
                    obstacle_positions.append([float(fallback_xy[0]), float(fallback_xy[1]), zt])
                    existing_xy.append(fallback_xy)

        return obstacle_positions


# ---- 保留旧名称兼容性别名 ---- #
QuadrantTaskGenerator = ObstacleCountTaskGenerator


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
