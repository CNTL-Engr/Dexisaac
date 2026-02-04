"""
等变Push网络（FC精炼器版本）- 用于UNet消融实验
U-Net1 + 图注意力 + FC精炼器（无第二个UNet）
用于验证第二个UNet的作用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class DoubleConv(nn.Module):
    """普通双卷积模块（非等变）"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DoubleConv, self).__init__()
        
        # First conv
        if stride == 1:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class GraphAttention(nn.Module):
    """
    图注意力模块（非等变）
    1. 从掩码提取节点特征
    2. 构建图（空间距离定义边）
    3. 图注意力计算物体交互
    4. 广播增强特征回特征图
    """
    def __init__(self, feature_dim):
        super(GraphAttention, self).__init__()
        self.feature_dim = feature_dim
        
        # 多头注意力参数
        self.num_heads = 4
        self.head_dim = self.feature_dim // self.num_heads
        
        # Query, Key, Value投影
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
        """从掩码提取节点特征"""
        B, C, H, W = feature_map.shape
        
        nodes_list = []
        positions_list = []
        
        for b in range(B):
            batch_nodes = []
            batch_positions = []
            
            for mask_idx in range(2):
                mask = masks[b, mask_idx]
                coords = torch.nonzero(mask > 0.5, as_tuple=False)
                
                if len(coords) > 0:
                    center_y = coords[:, 0].float().mean()
                    center_x = coords[:, 1].float().mean()
                    
                    mask_expanded = mask.unsqueeze(0).unsqueeze(0)
                    masked_features = feature_map[b:b+1] * mask_expanded
                    node_feature = masked_features.sum(dim=[2, 3]) / (mask.sum() + 1e-6)
                    node_feature = node_feature.squeeze(0)
                    
                    batch_nodes.append(node_feature)
                    pos_y = (center_y / H) * 2 - 1
                    pos_x = (center_x / W) * 2 - 1
                    batch_positions.append(torch.tensor([pos_x, pos_y], device=feature_map.device))
            
            if len(batch_nodes) == 0:
                global_feat = feature_map[b].mean(dim=[1, 2])
                batch_nodes.append(global_feat)
                batch_positions.append(torch.tensor([0.0, 0.0], device=feature_map.device))
            
            nodes_list.append(torch.stack(batch_nodes))
            positions_list.append(torch.stack(batch_positions))
        
        # Pad到相同节点数
        max_nodes = max(n.shape[0] for n in nodes_list)
        max_nodes = min(max_nodes, 4)
        max_nodes = max(max_nodes, 1)
        
        nodes_padded = []
        positions_padded = []
        
        for nodes, positions in zip(nodes_list, positions_list):
            curr_num = nodes.shape[0]
            if curr_num < max_nodes:
                pad_size = max_nodes - curr_num
                nodes = torch.cat([nodes, nodes[-1:].repeat(pad_size, 1)], dim=0)
                positions = torch.cat([positions, positions[-1:].repeat(pad_size, 1)], dim=0)
            elif curr_num > max_nodes:
                nodes = nodes[:max_nodes]
                positions = positions[:max_nodes]
            
            nodes_padded.append(nodes)
            positions_padded.append(positions)
        
        return torch.stack(nodes_padded), torch.stack(positions_padded)
    
    def graph_attention(self, nodes, positions):
        """图注意力机制"""
        B, N, C = nodes.shape
        
        Q = self.query_proj(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        dist_matrix = torch.cdist(positions, positions)
        dist_bias = -dist_matrix.unsqueeze(1)
        scores = scores + dist_bias
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        
        output = self.out_proj(attn_output)
        nodes = self.norm1(nodes + output)
        
        ffn_output = self.ffn(nodes)
        nodes = self.norm2(nodes + ffn_output)
        
        return nodes
    
    def broadcast_nodes_to_feature_map(self, nodes, positions, feature_map_shape):
        """将节点特征广播回特征图"""
        B, N, C = nodes.shape
        _, _, H, W = feature_map_shape
        
        y_grid = torch.linspace(-1, 1, H, device=nodes.device)
        x_grid = torch.linspace(-1, 1, W, device=nodes.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        feature_map = torch.zeros(B, C, H, W, device=nodes.device)
        
        for b in range(B):
            for n in range(N):
                pos = positions[b, n]
                node_feat = nodes[b, n]
                
                dist = torch.norm(grid[b] - pos.view(1, 1, 2), dim=-1)
                weight = torch.exp(- (dist ** 2) / (2 * 0.3 ** 2))
                
                feature_map[b] += node_feat.view(C, 1, 1) * weight.unsqueeze(0)
        
        return feature_map
    
    def forward(self, x, masks):
        """
        Args:
            x: Tensor (B, 128, H, W)
            masks: Tensor (B, 2, H_orig, W_orig)
        """
        B, C, H, W = x.shape
        
        masks_resized = F.interpolate(masks, size=(H, W), mode='nearest')
        
        nodes, positions = self.extract_nodes_from_masks(x, masks_resized)
        enhanced_nodes = self.graph_attention(nodes, positions)
        
        graph_features = self.broadcast_nodes_to_feature_map(
            enhanced_nodes, positions, x.shape
        )
        
        enhanced_features = x + graph_features
        
        return enhanced_features


class CNNPushNet(nn.Module):
    """
    非等变CNN架构：双U-Net + 图注意力 + 特征合并
    用于对照实验，验证C4等变性的优势
    """
    def __init__(self):
        super(CNNPushNet, self).__init__()
        
        # --- 1. 编码器配置 ---
        self.enc1 = DoubleConv(3, 16, stride=1)
        self.enc2 = DoubleConv(16, 32, stride=2)
        self.enc3 = DoubleConv(32, 64, stride=2)
        self.enc4 = DoubleConv(64, 128, stride=2)
        
        # --- 2. 图注意力 ---
        self.reasoning = GraphAttention(feature_dim=128)
        
        # --- 3. 第一个U-Net的解码器 ---
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv(128 + 64, 64, stride=1)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv(64 + 32, 32, stride=1)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(32 + 16, 16, stride=1)  # 16通道
        
        # --- 4. 图特征投影 ---
        # 128 -> 32*10*10
        self.graph_proj = nn.Sequential(
            nn.Linear(128, 32 * 10 * 10),
            nn.ReLU()
        )
        
        # --- 5. 特征合并层 ---
        # U-Net1: 16通道 + Graph: 32通道 = 48通道
        self.merge_conv = nn.Conv2d(48, 48, kernel_size=1)
        
        # --- 6. FC精炼器（替代第二个UNet）---
        # 使用全连接层代替UNet，用于消融实验验证UNet的作用
        # 层结构：48 → 256 → 128 → 64 → 8
        self.fc_refiner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化 (B, 48, 1, 1)
        )
        self.fc_head = nn.Sequential(
            nn.Linear(48, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # 输出8个Q值
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # ===== 第一个U-Net =====
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        masks = x[:, 1:3, :, :]
        x_neck = self.reasoning(x4, masks)
        
        x_up1 = self.up1(x_neck)
        d1 = self.dec1(torch.cat([x_up1, x3], dim=1))
        
        x_up2 = self.up2(d1)
        d2 = self.dec2(torch.cat([x_up2, x2], dim=1))
        
        x_up3 = self.up3(d2)
        d3 = self.dec3(torch.cat([x_up3, x1], dim=1))
        
        unet1_features = d3
        
        # ===== 图特征提取 =====
        bottleneck = x_neck
        graph_feat = F.adaptive_avg_pool2d(bottleneck, 1).view(batch_size, -1)
        
        del x1, x2, x3, x4, x_neck, x_up1, x_up2, x_up3, d1, d2, d3, bottleneck
        torch.cuda.empty_cache()
        
        # 图特征投影到空间
        graph_feat_proj = self.graph_proj(graph_feat)  # (B, 32*10*10)
        graph_spatial_small = graph_feat_proj.view(batch_size, 32, 10, 10)  # (B, 32, 10, 10)
        del graph_feat, graph_feat_proj
        
        graph_spatial = F.interpolate(graph_spatial_small, size=(320, 320), mode='bilinear', align_corners=False)  # (B, 32, 320, 320)
        del graph_spatial_small
        
        # ===== 特征合并 =====
        merged = torch.cat([unet1_features, graph_spatial], dim=1)  # (B, 48, 320, 320)
        
        del unet1_features, graph_spatial
        
        merged = self.merge_conv(merged)
        
        # ===== FC精炼器（替代第二个UNet）=====
        fc_feat = self.fc_refiner(merged)  # (B, 48, 1, 1)
        fc_feat = fc_feat.view(batch_size, -1)  # (B, 48)
        del merged
        
        # ===== 输出 =====
        q_values = self.fc_head(fc_feat)  # (B, 8)
        del fc_feat
        torch.cuda.empty_cache()
        
        return q_values
    
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    @staticmethod
    def load(path, device='cuda'):
        model = CNNPushNet()
        model.to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model
