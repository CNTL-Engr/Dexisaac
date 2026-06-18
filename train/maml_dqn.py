"""
MAML-DQN Agent for PushNet
实现 MAML 双循环机制与象限密度任务生成器
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from contextlib import contextmanager
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
from escnn import nn as enn


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

    [布局生成]:
      物体位置由 Scene.create_clutter_environment() 统一生成。
      本类只保留任务采样、障碍物数量和模型文件夹配置。
    """

    NUM_TASKS = 4  # 固定 4 种子任务（Task 0-3）

    # 障碍物数量基数：Task 0 对应 4 个障碍物
    BASE_OBSTACLE_COUNT = 8

    # 各 Task 的名称（用于打印/日志）
    TASK_NAMES = {
        0: "8 Obstacles",
        1: "9 Obstacles",
        2: "10 Obstacles",
        3: "11 Obstacles",
    }

    # 默认模型文件夹路径
    _DEFAULT_OBSTACLE_DIR = "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata/meshdata_CH"
    _DEFAULT_TARGET_DIR = "/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/meshdata/meshdata_target"

    def __init__(self, num_tasks=None, base_obstacle_count=None, radius=0.21, min_dist=0.07, max_attempts=1000, model_dirs=None):
        """
        [参数]:
          num_tasks          : 子任务数量，默认使用类属性 NUM_TASKS
          base_obstacle_count: 基础障碍物数量（Task 0 的障碍物数），默认使用类属性 BASE_OBSTACLE_COUNT
          radius             : 旧布局参数，保留用于兼容配置记录
          min_dist           : 旧布局参数，保留用于兼容配置记录
          max_attempts       : 旧布局参数，保留用于兼容配置记录
          model_dirs         : 每个子任务使用的模型文件夹配置 (dict 或 None)
        """
        if num_tasks is not None:
            self.NUM_TASKS = num_tasks
        if base_obstacle_count is not None:
            self.BASE_OBSTACLE_COUNT = base_obstacle_count
        self.TASK_NAMES = {
            i: f"{self.BASE_OBSTACLE_COUNT + i} Obstacles" for i in range(self.NUM_TASKS)
        }

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

    def sample_task_batch(self, batch_size, mode='random', curriculum_level=None):
        """
        [功能]: 采样一批 task_id，支持多种采样模式 (标准 MAML 接口)
        [输入]:
          batch_size       : 采样数量
          mode             : 'random' | 'curriculum' | 'curriculum_random' | 'balanced'
          curriculum_level : 当前课程等级 (仅 curriculum/curriculum_random 模式需要)
        [输出]: List[int], task_id 列表
        """
        if mode == 'random':
            return [random.randint(0, self.NUM_TASKS - 1) for _ in range(batch_size)]
        elif mode == 'curriculum':
            level = max(0, min(curriculum_level or 0, self.NUM_TASKS - 1))
            return [level] * batch_size
        elif mode == 'curriculum_random':
            level = max(0, min(curriculum_level or 0, self.NUM_TASKS - 1))
            return [random.randint(0, level) for _ in range(batch_size)]
        elif mode == 'balanced':
            task_ids = []
            while len(task_ids) < batch_size:
                shuffled_ids = list(range(self.NUM_TASKS))
                random.shuffle(shuffled_ids)
                task_ids.extend(shuffled_ids)
            return task_ids[:batch_size]
        else:
            raise ValueError(f"Unknown task sampling mode: {mode}")

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
    def __init__(self, device='cuda', lr=1e-4, inner_lr=1e-3, gamma=0.99,
                 use_equivariant=True, first_order=True, inner_steps=1):
        self.device = device
        self.gamma = gamma
        self.inner_lr = inner_lr
        self.first_order = first_order
        self.inner_steps = inner_steps
        self.use_equivariant = use_equivariant

        if use_equivariant:
            self.policy_net = EquivariantPushNet().to(device)
        else:
            self.policy_net = CNNPushNet().to(device)

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    @contextmanager
    def _dynamic_r2conv_filters_for_eval(self):
        """
        Temporarily bypass escnn R2Conv eval-mode filter caches.
        Only R2Conv modules are toggled; BatchNorm and all other modules keep
        their original training/eval state.
        """
        r2conv_states = []
        for module in self.policy_net.modules():
            if isinstance(module, enn.R2Conv) and not module.training:
                r2conv_states.append((module, module.training))
                module.train(True)
        try:
            yield
        finally:
            for module, was_training in r2conv_states:
                module.train(was_training)

    def _eval_functional_call(self, fast_weights, args):
        if self.policy_net.training:
            return functional_call(self.policy_net, fast_weights, args)
        with self._dynamic_r2conv_filters_for_eval():
            return functional_call(self.policy_net, fast_weights, args)
        
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
                    if self.policy_net.training:
                        q_values = functional_call(self.policy_net, fast_weights, (state_float,))
                    else:
                        q_values = self._eval_functional_call(fast_weights, (state_float,))
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

    def adapt(self, support_transitions, inner_steps=None, first_order=None, adapt_batch_size=None):
        """
        [功能]: Inner-loop Adaptation
        [输入]: support_transitions: Support Set 经验
                inner_steps: 内循环步数 (None 则使用实例默认值)
                first_order: 如果 True，不计算二阶导数 (FOMAML)
                adapt_batch_size: 适应阶段微批大小 (None 则保持原全量batch行为)
        [输出]: fast_weights: 字典形式的模型参数
        """
        if inner_steps is None:
            inner_steps = self.inner_steps
        if first_order is None:
            first_order = self.first_order
        if not support_transitions:
            return {name: param for name, param in self.policy_net.named_parameters()}

        states, actions, rewards, next_states, dones = zip(*support_transitions)
        support_size = len(support_transitions)

        if adapt_batch_size is not None and adapt_batch_size <= 0:
            adapt_batch_size = None

        if adapt_batch_size is None or adapt_batch_size >= support_size:
            return self._adapt_full_batch(
                states, actions, rewards, next_states, dones, inner_steps, first_order
            )

        return self._adapt_micro_batch(
            states,
            actions,
            rewards,
            next_states,
            dones,
            inner_steps,
            first_order,
            adapt_batch_size,
        )

    def _adapt_full_batch(self, states, actions, rewards, next_states, dones, inner_steps, first_order):
        states = torch.stack(states).to(self.device).float() / 255.0
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 计算 Target Q (不需要梯度，逐样本处理以节省显存)
        with torch.no_grad():
            next_states_t = torch.stack(next_states).to(self.device).float() / 255.0
            q_next_target = self.target_net(next_states_t)
            # Double DQN: 用 policy_net 选动作 (此处用初始权重即可)
            q_next_policy = self.policy_net(next_states_t)
            next_actions = q_next_policy.max(dim=1)[1]
            max_q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1 - dones) * self.gamma * max_q_next
            del next_states_t, q_next_target, q_next_policy, next_actions, max_q_next
            torch.cuda.empty_cache()

        # 初始化 fast_weights 为当前模型的参数
        fast_weights = {name: param for name, param in self.policy_net.named_parameters()}

        for _ in range(inner_steps):
            q_pred = functional_call(self.policy_net, fast_weights, (states,))
            q_values = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

            inner_loss = F.mse_loss(q_values, targets)

            grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=not first_order, allow_unused=True)

            new_fast_weights = {}
            for (name, param), grad in zip(fast_weights.items(), grads):
                if grad is not None:
                    new_fast_weights[name] = param - self.inner_lr * grad
                else:
                    new_fast_weights[name] = param
            fast_weights = new_fast_weights

        return fast_weights

    def _adapt_micro_batch(self, states, actions, rewards, next_states, dones, inner_steps, first_order, adapt_batch_size):
        support_size = len(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        targets = torch.empty(support_size, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for start in range(0, support_size, adapt_batch_size):
                end = min(start + adapt_batch_size, support_size)
                next_states_t = torch.stack(next_states[start:end]).to(self.device).float() / 255.0
                q_next_target = self.target_net(next_states_t)
                q_next_policy = self.policy_net(next_states_t)
                next_actions = q_next_policy.max(dim=1)[1]
                max_q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets[start:end] = rewards[start:end] + (1 - dones[start:end]) * self.gamma * max_q_next
                del next_states_t, q_next_target, q_next_policy, next_actions, max_q_next
                torch.cuda.empty_cache()

        fast_weights = {name: param for name, param in self.policy_net.named_parameters()}

        for _ in range(inner_steps):
            fast_items = list(fast_weights.items())
            fast_params = [param for _, param in fast_items]
            grad_sums = [None for _ in fast_params]

            for start in range(0, support_size, adapt_batch_size):
                end = min(start + adapt_batch_size, support_size)
                states_t = torch.stack(states[start:end]).to(self.device).float() / 255.0
                q_pred = self._eval_functional_call(fast_weights, (states_t,))
                q_values = q_pred.gather(1, actions[start:end].unsqueeze(1)).squeeze(1)
                inner_loss = F.mse_loss(q_values, targets[start:end], reduction="sum") / support_size

                grads = torch.autograd.grad(
                    inner_loss,
                    fast_params,
                    create_graph=not first_order,
                    allow_unused=True,
                )

                for idx, grad in enumerate(grads):
                    if grad is None:
                        continue
                    if grad_sums[idx] is None:
                        grad_sums[idx] = grad
                    else:
                        grad_sums[idx] = grad_sums[idx] + grad

                del states_t, q_pred, q_values, inner_loss, grads
                torch.cuda.empty_cache()

            new_fast_weights = {}
            for (name, param), grad in zip(fast_items, grad_sums):
                if grad is not None:
                    updated = param - self.inner_lr * grad
                else:
                    updated = param
                if first_order:
                    updated = updated.detach().requires_grad_(True)
                new_fast_weights[name] = updated
            fast_weights = new_fast_weights

        return fast_weights

    def meta_update(self, task_batch, inner_steps=None, first_order=None):
        """
        [功能]: Outer-loop Meta Update
        [返回]: meta_loss 的均值
        """
        if inner_steps is None:
            inner_steps = self.inner_steps
        if first_order is None:
            first_order = self.first_order

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

            q_actions = torch.tensor(q_actions, dtype=torch.long, device=self.device)
            q_rewards = torch.tensor(q_rewards, dtype=torch.float32, device=self.device)
            q_dones = torch.tensor(q_dones, dtype=torch.float32, device=self.device)

            # 先计算 targets (不需要梯度), 然后释放 next_states 显存
            with torch.no_grad():
                q_next_states_t = torch.stack(q_next_states).to(self.device).float() / 255.0
                q_next_policy = functional_call(self.policy_net, fast_weights, (q_next_states_t,))
                next_actions = q_next_policy.max(dim=1)[1]
                q_next_target = self.target_net(q_next_states_t)
                max_q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = q_rewards + (1 - q_dones) * self.gamma * max_q_next
                del q_next_states_t, q_next_policy, next_actions, q_next_target, max_q_next
                torch.cuda.empty_cache()

            # 前向传播使用适应后的 fast_weights (需要梯度)
            q_states_t = torch.stack(q_states).to(self.device).float() / 255.0
            q_pred = functional_call(self.policy_net, fast_weights, (q_states_t,))
            q_values = q_pred.gather(1, q_actions.unsqueeze(1)).squeeze(1)

            task_meta_loss = F.mse_loss(q_values, targets)

            task_meta_loss_scaled = task_meta_loss / len(task_batch)
            task_meta_loss_scaled.backward()

            meta_loss += task_meta_loss.item()

            del q_states_t, q_actions, q_rewards, q_dones
            del q_pred, q_values, targets
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
