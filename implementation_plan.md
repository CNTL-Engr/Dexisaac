目标物体分离成功判定逻辑修改方案

修改文件

[MODIFY] env_wrapper.py (file:/home/disk_18T/user/kjy/equi/IsaacLab/scripts/Dexisaac_MAML/train/env_wrapper.py)
重写 `_check_successful_separation` 方法。

核心逻辑步骤：
1. 提取目标物体的外接矩形 (Bounding Box)：
   通过 `target_mask` 计算物体像素的最小和最大坐标（`min_x, max_x, min_y, max_y`）。
   计算目标物体的原始宽度 `W = max_x - min_x` 和高度 `H = max_y - min_y`。
   计算物体的中心点 `cx = (min_x + max_x) / 2` 和 `cy = (min_y + max_y) / 2`。

2. 构建向外扩张 3cm 的检测区域：
   计算 3cm 对应的像素数（约 21 像素，设为 `margin`）。
   包围盒向四个方向分别扩展 `margin` 的距离，得到扩张后的检测矩形区域。扩张后的宽为 `W_exp = W + 2*margin`，高为 `H_exp = H + 2*margin`。

3. 利用对角线划分为四个检测区 (上下左右)：
   利用矩形的对角线将该检测区域精确地切分为 4 个三角形区域。对于区域内任意像素 `(x, y)`，计算其相对于中心的归一化坐标 `nx = (x - cx) / W_exp`, `ny = (y - cy) / H_exp`：
   - 上区域 (Top)：`ny < -abs(nx)`
   - 下区域 (Bottom)：`ny > abs(nx)`
   - 左区域 (Left)：`nx < -abs(ny)`
   - 右区域 (Right)：`nx > abs(ny)`

4. 提取杂物掩膜：
   通过 `spawned_objects` 获取环境中所有物体 ID，并在分割图中生成 `other_objects_mask`（所有交互物体，排除目标物体自身）。

5. 统计各区域的杂物像素：
   分别计算上下左右四个区域内，落在 `other_objects_mask` 上的像素总数（`top_obstacles`, `bottom_obstacles`, `left_obstacles`, `right_obstacles`）。

6. 判定分离成功的标准 (结合长宽比例)：
   设置容差阈值 `threshold_pixels = 10`。
   - 当 `W >= H` 时（水平长条或正方形）：
     上、下两侧是长边，对应上下区域。要求上下区域必须无障碍物（`top_obstacles < 10` 且 `bottom_obstacles < 10`）。左右区域（短边）允许有障碍物。
   - 当 `H > W` 时（垂直长条）：
     左、右两侧是长边，对应左右区域。要求左右区域必须无障碍物（`left_obstacles < 10` 且 `right_obstacles < 10`）。上下区域（短边）允许有障碍物。

---

## User Review Required & Open Questions

>
> 核心逻辑已确认为：
> - 提取外接矩形包围盒并向外扩张 3cm。
> - 利用这个扩展矩形的对角线划分四个三角区。
> - 比较原始（或扩展后）长宽：
>   - W >= H 优先要求上下两块区域干净（<10像素）。
>   - W < H 优先要求左右两块区域干净（<10像素）。

