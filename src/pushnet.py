"""
等变Push网络 - 基于论文完整架构
双U-Net + 图注意力 + 特征合并
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces, nn as enn
import cv2

class EquivariantDoubleConv(enn.EquivariantModule):
    """等变双卷积模块"""
    def __init__(self, in_type, out_type, stride=1):
        super(EquivariantDoubleConv, self).__init__()
        
        # First conv
        if stride == 1:
            self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type)
        
        # Second conv
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
    def evaluate_output_shape(self, input_shape):
        return input_shape


class EquivariantGraphAttention(nn.Module):
    """
    等变图注意力模块 - 完整实现
    1. 从掩码提取节点特征
    2. 构建图（空间距离定义边）
    3. 等变图注意力计算物体交互
    4. 广播增强特征回特征图
    """
    def __init__(self, feature_type):
        super(EquivariantGraphAttention, self).__init__()
        self.feature_type = feature_type
        
        # 特征维度（128通道）
        self.feature_dim = 128
        
        # 节点特征投影（等变）
        # 从空间特征图提取节点特征
        self.node_pooling = nn.AdaptiveAvgPool2d(1)  # 池化到单值
        
        # 多头注意力参数
        self.num_heads = 4
        self.head_dim = self.feature_dim // self.num_heads  # 32
        
        # Query, Key, Value投影（保持等变性）
        self.query_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.key_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_proj = nn.Linear(self.feature_dim, self.feature_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )
        
    def extract_nodes_from_masks(self, feature_map, masks):
        """
        从掩码提取节点特征
        Args:
            feature_map: (B, C, H, W) 特征图
            masks: (B, 2, H, W) - [Target, Obstacle]
        Returns:
            nodes: (B, num_nodes, C) 节点特征
            node_positions: (B, num_nodes, 2) 节点位置(归一化)
        """
        B, C, H, W = feature_map.shape
        
        nodes_list = []
        positions_list = []
        
        for b in range(B):
            batch_nodes = []
            batch_positions = []
            
            # 处理每个掩码（Target, Obstacle）
            for mask_idx in range(2):  # 0:Target, 1:Obstacle
                mask = masks[b, mask_idx]  # (H, W)
                
                # 找到掩码区域
                coords = torch.nonzero(mask > 0.5, as_tuple=False)  # (N, 2)
                
                if len(coords) > 0:
                    # 计算掩码中心
                    center_y = coords[:, 0].float().mean()
                    center_x = coords[:, 1].float().mean()
                    
                    # 提取该区域的特征（使用掩码加权平均）
                    mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    masked_features = feature_map[b:b+1] * mask_expanded  # (1, C, H, W)
                    
                    # 池化得到节点特征
                    node_feature = masked_features.sum(dim=[2, 3]) / (mask.sum() + 1e-6)  # (1, C)
                    node_feature = node_feature.squeeze(0)  # (C,)

                    
                    batch_nodes.append(node_feature)
                    
                    # 归一化位置到[-1, 1]
                    pos_y = (center_y / H) * 2 - 1
                    pos_x = (center_x / W) * 2 - 1
                    batch_positions.append(torch.tensor([pos_x, pos_y], device=feature_map.device))
            
            # 如果没有找到任何节点，使用全局特征
            if len(batch_nodes) == 0:
                global_feat = feature_map[b].mean(dim=[1, 2])  # (C,)
                batch_nodes.append(global_feat)
                batch_positions.append(torch.tensor([0.0, 0.0], device=feature_map.device))
            
            nodes_list.append(torch.stack(batch_nodes))  # (num_nodes, C)
            positions_list.append(torch.stack(batch_positions))  # (num_nodes, 2)
        
        # Pad到相同节点数
        max_nodes = max(n.shape[0] for n in nodes_list)
        max_nodes = min(max_nodes, 4)  # 限制最大节点数
        max_nodes = max(max_nodes, 1)  # 至少1个节点
        
        nodes_padded = []
        positions_padded = []
        
        for nodes, positions in zip(nodes_list, positions_list):
            curr_num = nodes.shape[0]
            if curr_num < max_nodes:
                # Padding
                pad_size = max_nodes - curr_num
                nodes = torch.cat([nodes, nodes[-1:].repeat(pad_size, 1)], dim=0)
                positions = torch.cat([positions, positions[-1:].repeat(pad_size, 1)], dim=0)
            elif curr_num > max_nodes:
                nodes = nodes[:max_nodes]
                positions = positions[:max_nodes]
            
            nodes_padded.append(nodes)
            positions_padded.append(positions)
        
        
        return torch.stack(nodes_padded), torch.stack(positions_padded)  # (B, max_nodes, C), (B, max_nodes, 2)

    
    def graph_attention(self, nodes, positions):
        """
        图注意力机制
        Args:
            nodes: (B, N, C) 节点特征
            positions: (B, N, 2) 节点位置
        Returns:
            updated_nodes: (B, N, C)
        """
        B, N, C = nodes.shape
        
        # Multi-head attention
        Q = self.query_proj(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.key_proj(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        
        # 距离bias（空间距离影响注意力）
        dist_matrix = torch.cdist(positions, positions)  # (B, N, N)
        dist_bias = -dist_matrix.unsqueeze(1)  # (B, 1, N, N)
        scores = scores + dist_bias
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)  # (B, H, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        # 残差连接 + LayerNorm
        nodes = self.norm1(nodes + output)
        
        # FFN
        ffn_output = self.ffn(nodes)
        nodes = self.norm2(nodes + ffn_output)
        
        return nodes
    
    def broadcast_nodes_to_feature_map(self, nodes, positions, feature_map_shape):
        """
        将节点特征广播回特征图
        Args:
            nodes: (B, N, C)
            positions: (B, N, 2)
            feature_map_shape: (B, C, H, W)
        Returns:
            feature_map: (B, C, H, W)
        """
        B, N, C = nodes.shape
        _, _, H, W = feature_map_shape
        
        # 创建位置网格
        y_grid = torch.linspace(-1, 1, H, device=nodes.device)
        x_grid = torch.linspace(-1, 1, W, device=nodes.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # 计算每个像素到每个节点的距离权重
        feature_map = torch.zeros(B, C, H, W, device=nodes.device)
        
        for b in range(B):
            for n in range(N):
                pos = positions[b, n]  # (2,)
                node_feat = nodes[b, n]  # (C,)
                
                # 计算距离
                dist = torch.norm(grid[b] - pos.view(1, 1, 2), dim=-1)  # (H, W)
                
                # Gaussian权重（sigma=0.3）
                weight = torch.exp(- (dist ** 2) / (2 * 0.3 ** 2))  # (H, W)
                
                # 广播特征
                feature_map[b] += node_feat.view(C, 1, 1) * weight.unsqueeze(0)  # (C, H, W)
        
        return feature_map
        
    def forward(self, x, masks, return_debug_info=False):
        """
        Args:
            x: GeometricTensor (B, 128, H, W) - 注意H,W是下采样后的尺寸(如40x40)
            masks: Tensor (B, 2, H_orig, W_orig) - 原始尺寸的掩码(如320x320)
        Returns:
            enhanced_x: GeometricTensor
        """
        feature_map = x.tensor  # (B, 128, H, W), H=W=40
        B, C, H, W = feature_map.shape
        
        # [修复] 下采样masks到与feature_map相同的空间尺寸
        masks_resized = F.interpolate(masks, size=(H, W), mode='nearest')  # (B, 2, 40, 40)
        
        # 1. 提取节点（使用resize后的masks）
        nodes, positions = self.extract_nodes_from_masks(feature_map, masks_resized)  # (B, N, 128), (B, N, 2)
        
        # 2. 图注意力
        enhanced_nodes = self.graph_attention(nodes, positions)  # (B, N, 128)
        
        # 3. 广播回特征图
        graph_features = self.broadcast_nodes_to_feature_map(
            enhanced_nodes, positions, feature_map.shape
        )  # (B, 128, H, W)
        
        # 4. 融合原始特征和图特征
        enhanced_features = feature_map + graph_features  # 残差连接
        
        # 5. 转回GeometricTensor
        enhanced_x = enn.GeometricTensor(enhanced_features, x.type)
        
        if return_debug_info:
            debug_info = {
                'nodes': nodes,
                'positions': positions,
                'graph_features': graph_features
            }
            return enhanced_x, debug_info
        
        return enhanced_x


class EquivariantPushNet(nn.Module):
    """
    论文完整架构：双U-Net + 图注意力 + 特征合并
    """
    def __init__(self):
        super(EquivariantPushNet, self).__init__()
        
        # --- 1. C4对称群 ---
        self.gspace = gspaces.rot2dOnR2(N=4)
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        
        activation = self.gspace.regular_repr
        
        # Encoder Config
        self.enc1_type = enn.FieldType(self.gspace, 4 * [activation])   # 16
        self.enc2_type = enn.FieldType(self.gspace, 8 * [activation])   # 32
        self.enc3_type = enn.FieldType(self.gspace, 16 * [activation])  # 64
        self.enc4_type = enn.FieldType(self.gspace, 32 * [activation])  # 128
        
        # Encoder Layers
        self.enc1 = EquivariantDoubleConv(self.in_type, self.enc1_type, stride=1)
        self.enc2 = EquivariantDoubleConv(self.enc1_type, self.enc2_type, stride=2)
        self.enc3 = EquivariantDoubleConv(self.enc2_type, self.enc3_type, stride=2)
        self.enc4 = EquivariantDoubleConv(self.enc3_type, self.enc4_type, stride=2)
        
        # --- 2. 图注意力 ---
        self.reasoning = EquivariantGraphAttention(self.enc4_type)
        
        # --- 3. 第一个U-Net的Decoder ---
        self.up1 = enn.R2Upsampling(self.enc4_type, scale_factor=2)
        self.dec1_in_type = self.enc4_type + self.enc3_type
        self.dec1 = EquivariantDoubleConv(self.dec1_in_type, self.enc3_type, stride=1)
        
        self.up2 = enn.R2Upsampling(self.enc3_type, scale_factor=2)
        self.dec2_in_type = self.enc3_type + self.enc2_type
        self.dec2 = EquivariantDoubleConv(self.dec2_in_type, self.enc2_type, stride=1)
        
        self.up3 = enn.R2Upsampling(self.enc2_type, scale_factor=2)
        self.dec3_in_type = self.enc2_type + self.enc1_type
        self.dec3 = EquivariantDoubleConv(self.dec3_in_type, self.enc1_type, stride=1)  # 16通道
        
        # --- 4. 图特征投影（广播到空间）---
        # 128 -> 32*10*10，然后上采样到320x320
        self.graph_proj = nn.Sequential(
            nn.Linear(128, 32 * 10 * 10),  # 参数量: 128 * 3200 = 409,600
            nn.ReLU()
        )
        
        # --- 5. 特征合并层 ---
        # U-Net1: 16通道 + Graph: 32通道 = 48通道
        self.merge_conv = nn.Conv2d(48, 48, kernel_size=1)
        self.merged_type = enn.FieldType(self.gspace, 12 * [activation])  # 48通道
        
        # --- 6. 第二个U-Net（轻量精炼器）---
        self.unet2_enc1_type = enn.FieldType(self.gspace, 16 * [activation])  # 64
        self.unet2_enc2_type = enn.FieldType(self.gspace, 8 * [activation])   # 32
        
        self.unet2_enc1 = EquivariantDoubleConv(self.merged_type, self.unet2_enc1_type, stride=2)
        self.unet2_enc2 = EquivariantDoubleConv(self.unet2_enc1_type, self.unet2_enc2_type, stride=2)
        
        self.unet2_up1 = enn.R2Upsampling(self.unet2_enc2_type, scale_factor=2)
        self.unet2_dec1_in = self.unet2_enc2_type + self.unet2_enc1_type
        self.unet2_dec1 = EquivariantDoubleConv(self.unet2_dec1_in, self.unet2_enc1_type, stride=1)
        
        self.unet2_up2 = enn.R2Upsampling(self.unet2_enc1_type, scale_factor=2)
        self.unet2_dec2_in = self.unet2_enc1_type + self.merged_type
        self.unet2_dec2 = EquivariantDoubleConv(self.unet2_dec2_in, self.merged_type, stride=1)
        
        # --- 7. 输出头：8通道Q值图 ---
        self.out_type = enn.FieldType(self.gspace, 2 * [activation])  # 8通道
        self.head_q = enn.R2Conv(self.merged_type, self.out_type, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        
        # ===== 第一个U-Net =====
        x_geo = enn.GeometricTensor(x, self.in_type)
        
        x1 = self.enc1(x_geo)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        masks = x[:, 1:3, :, :]
        x_neck = self.reasoning(x4, masks)
        
        x_up1 = self.up1(x_neck)
        d1 = self.dec1(enn.tensor_directsum([x_up1, x3]))
        
        x_up2 = self.up2(d1)
        d2 = self.dec2(enn.tensor_directsum([x_up2, x2]))
        
        x_up3 = self.up3(d2)
        d3 = self.dec3(enn.tensor_directsum([x_up3, x1]))
        
        unet1_features = d3.tensor  # (B, 16, 320, 320)
        
        # ===== 图特征提取 =====
        # [关键] 在删除x_neck之前先提取bottleneck
        bottleneck = x_neck.tensor  # (B, 128, H, W)
        graph_feat = F.adaptive_avg_pool2d(bottleneck, 1).view(batch_size, -1)  # (B, 128)
        
        # [内存优化] 现在可以安全删除UNet1中间特征
        del x_geo, x1, x2, x3, x4, x_neck, x_up1, x_up2, x_up3, d1, d2, d3, bottleneck
        torch.cuda.empty_cache()
        
        # 图特征投影到空间
        # 先投影到 32*10*10，然后上采样到 320x320
        graph_feat_proj = self.graph_proj(graph_feat)  # (B, 32*10*10)
        graph_spatial_small = graph_feat_proj.view(batch_size, 32, 10, 10)  # (B, 32, 10, 10)
        del graph_feat, graph_feat_proj  # 立即释放
        
        # 上采样到 320x320
        graph_spatial = F.interpolate(graph_spatial_small, size=(320, 320), mode='bilinear', align_corners=False)  # (B, 32, 320, 320)
        del graph_spatial_small  # 释放中间结果
        
        # ===== 特征合并 =====
        merged = torch.cat([unet1_features, graph_spatial], dim=1)  # (B, 48, 320, 320)
        
        # 释放不再需要的特征
        del unet1_features, graph_spatial
        
        merged = self.merge_conv(merged)
        merged_geo = enn.GeometricTensor(merged, self.merged_type)
        
        #===== 第二个U-Net =====
        u2_e1 = self.unet2_enc1(merged_geo)
        u2_e2 = self.unet2_enc2(u2_e1)
        
        # [内存优化] 释放merged，保留merged_geo用于后续skip connection
        del merged
        
        u2_up1 = self.unet2_up1(u2_e2)
        u2_d1 = self.unet2_dec1(enn.tensor_directsum([u2_up1, u2_e1]))
        
        # [内存优化] 释放encoder特征
        del u2_e2, u2_up1
        
        u2_up2 = self.unet2_up2(u2_d1)
        u2_d2 = self.unet2_dec2(enn.tensor_directsum([u2_up2, merged_geo]))
        
        # [内存优化] 释放decoder中间特征
        del u2_e1, u2_d1, u2_up2, merged_geo
        
        # ===== 输出 =====
        q_map = self.head_q(u2_d2).tensor  # (B, 8, 320, 320)
        
        # [内存优化] 释放最后的decoder输出
        del u2_d2
        
        q_values = F.adaptive_max_pool2d(q_map, 1).view(batch_size, -1)  # (B, 8)
        
        # [内存优化] 释放Q地图
        del q_map
        torch.cuda.empty_cache()
        
        return q_values
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    @staticmethod
    def load(path, device='cuda'):
        model = EquivariantPushNet()
        model.to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model
