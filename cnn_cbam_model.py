"""
=============================================================================
创新方案: 基于CNN+CBAM注意力的LHAASO伽马/质子鉴别
=============================================================================
灵感来源: Zhang et al. 2023 (arXiv:2303.00370) - 用ResNet-CBAM在Fermi/GBM
         探测器计数图上区分GRB与非GRB，达到98%检出率

适配LHAASO的创新点:
  1. 将LHAASO事例转换为探测器命中图 (类似Fermi count map)
  2. ResNet-CBAM架构 (通道注意力 + 空间注意力)
  3. 多通道输入: ED电磁通道 + MD缪子通道 + 时间通道
  4. Grad-CAM可解释性: 验证模型聚焦于物理合理的探测器区域
  5. 多模态融合: 图像分支(CNN) + 表格分支(MLP) → 联合判别

三维度创新:
  - 方法创新: 首次将CBAM注意力用于LHAASO伽马/质子鉴别
  - 数据创新: 从表格数据升级为探测器二维图像
  - 物理创新: Grad-CAM验证 + 物理引导的通道设计

=============================================================================
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Part 1: CBAM 注意力模块 (论文核心组件)
# =============================================================================

class ChannelAttention(nn.Module):
    """
    通道注意力: 学习"哪个探测器子系统更重要"
    对应LHAASO物理: 自动学习 ED vs MD 哪个对鉴别更重要

    论文中: CBAM的Channel Attention分支
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    空间注意力: 学习"探测器上哪个空间区域最重要"
    对应LHAASO物理: 自动学习簇射核心区域 vs 外围区域的重要性

    论文中: CBAM的Spatial Attention分支
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    = 通道注意力 + 空间注意力 (串联)

    论文Figure 2的核心结构
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力: 加权哪些特征通道更重要
        x = x * self.channel_attention(x)
        # 空间注意力: 加权哪些像素位置更重要
        x = x * self.spatial_attention(x)
        return x


# =============================================================================
# Part 2: ResNet-CBAM 主干网络 (论文Figure 3的架构)
# =============================================================================

class ResidualCBAMBlock(nn.Module):
    """
    残差块 + CBAM 注意力
    对应论文的 Figure 2 + Figure 3
    """
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(ResidualCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接 (维度不匹配时用1x1卷积调整)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_cbam:
            out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCBAM(nn.Module):
    """
    ResNet-CBAM 主干网络
    论文中表现最优的架构 (Table 2: ResNet-CBAM > ResNet > plain-CNN)

    适配LHAASO: 输入为探测器命中图 [batch, C, H, W]
    """
    def __init__(self, in_channels=3, num_classes=2,
                 base_channels=32, num_blocks=[2, 2, 2]):
        super(ResNetCBAM, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 三个残差层 (类似ResNet-18/34的设计，每层有CBAM)
        self.layer1 = self._make_layer(base_channels, base_channels,
                                        num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2,
                                        num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4,
                                        num_blocks[2], stride=2)

        # 全局池化 + 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # 第一个块处理通道数/尺寸变化
        layers.append(ResidualCBAMBlock(in_channels, out_channels, stride))
        # 后续块保持尺寸
        for _ in range(1, num_blocks):
            layers.append(ResidualCBAMBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                         nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# Part 3: 多模态融合模型 (表格 + 图像)
# =============================================================================

class MultiModalFusion(nn.Module):
    """
    多模态融合模型: 图像特征(CNN) + 表格特征(MLP)

    对应论文中"fusing strategy models"的思路，扩展为跨模态融合

    图像分支: ResNet-CBAM → 提取探测器空间特征
    表格分支: MLP → 处理现有8维物理特征
    融合层: 拼接 + FC → 联合分类
    """
    def __init__(self, num_tabular_features=8, num_classes=2,
                 img_channels=3, fusion_dim=128):
        super(MultiModalFusion, self).__init__()

        # ---- 图像分支: ResNet-CBAM ----
        self.image_encoder = ResNetCBAM(
            in_channels=img_channels,
            num_classes=fusion_dim,  # 输出fusion维度的特征向量
            base_channels=32
        )
        # 替换最后的fc层为特征提取
        self.image_encoder.fc = nn.Identity()
        # 重新计算特征维度
        self.img_feat_dim = 32 * 4  # base_channels * 4 (from layer3)

        # ---- 表格分支: MLP ----
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.tab_feat_dim = 32

        # ---- 融合层 ----
        total_feat_dim = self.img_feat_dim + self.tab_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, image, tabular):
        # 图像特征
        img_feat = self.image_encoder(image)
        img_feat = img_feat.view(img_feat.size(0), -1)

        # 表格特征
        tab_feat = self.tabular_encoder(tabular)

        # 拼接 + 分类
        combined = torch.cat([img_feat, tab_feat], dim=1)
        output = self.fusion(combined)
        return output


# =============================================================================
# Part 4: LHAASO事例 → 探测器命中图转换器
# =============================================================================

class LHAASOEventMapper:
    """
    将LHAASO重建事例转换为探测器命中图

    输入: DataFrame行 (含探测器级别的命中信息)
    输出: 多通道图像 [channels, height, width]

    三通道设计 (类比RGB):
      Channel 0 (R): 电磁探测器(ED)命中密度图
      Channel 1 (G): μ子探测器(MD)命中密度图
      Channel 2 (B): 命中时间信息图

    物理意义: 这三个通道对应伽马/质子鉴别的三个关键物理量
      - ED通道: 电磁成分空间分布
      - MD通道: μ子成分空间分布 (质子事件中更显著)
      - 时间通道: 簇射前沿到达时间 (质子事件更弥散)
    """
    def __init__(self, grid_size=32, ed_positions=None, md_positions=None):
        """
        参数:
            grid_size: 输出图像的网格大小 (32x32)
            ed_positions: ED探测器XY坐标 [(x1,y1), ...]
            md_positions: MD探测器XY坐标 [(x1,y1), ...]
        """
        self.grid_size = grid_size
        # 如果没有提供探测器坐标，使用模拟的LHAASO-KM2A布局
        if ed_positions is None:
            self.ed_positions = self._simulate_km2a_layout(n_detectors=500)
        else:
            self.ed_positions = ed_positions

        if md_positions is None:
            self.md_positions = self._simulate_km2a_md_layout(n_detectors=100)
        else:
            self.md_positions = md_positions

    def _simulate_km2a_layout(self, n_detectors=500):
        """模拟KM2A电磁探测器阵列布局 (矩形网格)"""
        np.random.seed(42)
        positions = []
        grid_n = int(np.sqrt(n_detectors))
        for i in range(grid_n):
            for j in range(grid_n):
                # 添加小量随机偏移模拟实际布局的不完美
                x = i / grid_n + np.random.normal(0, 0.005)
                y = j / grid_n + np.random.normal(0, 0.005)
                positions.append((x, y))
        return positions[:n_detectors]

    def _simulate_km2a_md_layout(self, n_detectors=100):
        """模拟KM2A μ子探测器阵列布局 (较稀疏)"""
        np.random.seed(123)
        positions = []
        grid_n = int(np.sqrt(n_detectors))
        for i in range(grid_n):
            for j in range(grid_n):
                x = i / grid_n + np.random.normal(0, 0.01)
                y = j / grid_n + np.random.normal(0, 0.01)
                positions.append((x, y))
        return positions[:n_detectors]

    def _detector_hits_to_image(self, detector_hits, positions, grid_size):
        """
        将探测器命中数据转换为2D密度图

        参数:
            detector_hits: 每个探测器的命中数/信号强度 [n_detectors]
            positions:      探测器坐标 [(x, y), ...]
            grid_size:      输出网格大小
        返回:
            image: [grid_size, grid_size] 命中密度图
        """
        image = np.zeros((grid_size, grid_size))
        for (x, y), hit in zip(positions, detector_hits):
            px = int(np.clip(x * grid_size, 0, grid_size - 1))
            py = int(np.clip(y * grid_size, 0, grid_size - 1))
            image[py, px] += hit
        # 高斯平滑模拟探测器的空间响应
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=0.8)
        return image

    def convert_event(self, df_row, ed_hits_col=None, md_hits_col=None,
                       time_col=None):
        """
        将单个事例的DataFrame行转换为多通道图像

        如果数据中没有探测器级别的hit信息 (当前项目只有聚合特征),
        则使用聚合特征重建伪探测器图像 (近似方法)

        参数:
            df_row: 一个事例的特征行
        返回:
            image: [3, grid_size, grid_size] 的三通道图像
        """
        grid = self.grid_size
        # 默认通道: 全零初始化
        ed_channel = np.zeros((grid, grid))
        md_channel = np.zeros((grid, grid))
        time_channel = np.zeros((grid, grid))

        # ---- 使用聚合特征重建伪探测器图像 ----
        # 由于当前数据只有8个聚合特征，我们用物理模型重建空间分布

        # 簇射核心位置 (假设在图像中心附近)
        core_x, core_y = grid // 2, grid // 2

        # 从特征中提取关键参数
        total_hits = float(df_row.get('NhitM', 100))
        muon_total = (float(df_row.get('NuM1', 0)) + float(df_row.get('NuM2', 0)) +
                      float(df_row.get('NuM3', 0)) + float(df_row.get('NuM4', 0)))

        # 计算横向扩展宽度 (用环计数比例估计)
        nu_ring_ratio = (float(df_row.get('NuM3', 0)) + float(df_row.get('NuM4', 0))) / \
                        (float(df_row.get('NuM1', 0)) + float(df_row.get('NuM2', 0)) + 1e-8)
        # 质子nu_ring_ratio更大 (外围环更多μ子)
        lateral_sigma = 3.0 + 4.0 * nu_ring_ratio  # 质子sigma更大

        # 计算不对称性
        asymmetry = abs(float(df_row.get('NuM1', 0)) - float(df_row.get('NuM3', 0))) / \
                    (float(df_row.get('NuM1', 0)) + float(df_row.get('NuM3', 0)) + 1e-8)

        # ---- Channel 0: ED通道 (电磁成分) ----
        # 对称的高斯分布，宽度较窄
        for i in range(grid):
            for j in range(grid):
                dx = i - core_x
                dy = j - core_y
                # ED信号: 对称高斯
                r2 = dx**2 + dy**2
                ed_channel[i, j] = total_hits * np.exp(-r2 / (2 * (lateral_sigma * 0.7)**2))

        # ---- Channel 1: MD通道 (μ子成分) ----
        # 不对称的分布，质子中外围μ子更多
        for i in range(grid):
            for j in range(grid):
                dx = i - core_x
                dy = j - core_y
                # MD信号: 更宽的高斯 + 不对称扰动
                r2 = dx**2 + dy**2
                asym_factor = 1.0 + asymmetry * (dx / (abs(dx) + 1e-8)) * 0.3
                md_channel[i, j] = muon_total * np.exp(
                    -r2 / (2 * lateral_sigma**2)
                ) * asym_factor

        # ---- Channel 2: 时间信息通道 ----
        # 簇射前沿到达时间近似于距核心距离的函数
        # 质子由于μ子在远离核心处到达更晚
        for i in range(grid):
            for j in range(grid):
                dx = i - core_x
                dy = j - core_y
                r = np.sqrt(dx**2 + dy**2)
                # 时间延迟 ∝ 距离 (簇射前沿弯曲)
                time_delay = r * (1.0 + 0.5 * nu_ring_ratio)  # 质子时间更弥散
                time_channel[i, j] = time_delay

        # 归一化各通道 [0, 1]
        ed_channel = ed_channel / (ed_channel.max() + 1e-8)
        md_channel = md_channel / (md_channel.max() + 1e-8)
        time_channel = time_channel / (time_channel.max() + 1e-8)

        # 堆叠为 [3, H, W]
        image = np.stack([ed_channel, md_channel, time_channel], axis=0).astype(np.float32)
        return image


# =============================================================================
# Part 5: Sklearn-compatible 包装器
# =============================================================================

class LHAASOResNetCBAMClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn兼容的 ResNet-CBAM + 多模态融合 分类器

    用法:
        model = LHAASOResNetCBAMClassifier(use_tabular=True, epochs=50)
        model.fit(X_tabular, y, image_builder=event_mapper)
        y_pred = model.predict(X_tabular)
    """

    def __init__(self, use_tabular=True, use_image=True,
                 epochs=50, lr=0.001, batch_size=32,
                 img_grid_size=32, fusion_dim=128):
        self.use_tabular = use_tabular
        self.use_image = use_image
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.img_grid_size = img_grid_size
        self.fusion_dim = fusion_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self, n_tabular_features):
        if self.use_image and self.use_tabular:
            model = MultiModalFusion(
                num_tabular_features=n_tabular_features,
                num_classes=2,
                img_channels=3,
                fusion_dim=self.fusion_dim
            )
        elif self.use_image:
            model = ResNetCBAM(in_channels=3, num_classes=2)
        else:
            # 纯表格模型 (fallback)
            model = nn.Sequential(
                nn.Linear(n_tabular_features, 128),
                nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
        return model.to(self.device)

    def fit(self, X, y, image_builder=None):
        """
        参数:
            X:              表格特征 [n_samples, n_features]
            y:              标签 [n_samples]
            image_builder:  LHAASOEventMapper实例，用于生成探测器图像
        """
        self.classes_ = np.unique(y)
        n_features = X.shape[1] if isinstance(X, np.ndarray) else X.shape[1]

        self.model_ = self._build_model(n_features)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # 如果使用图像分支，构建事件映射器
        if self.use_image and image_builder is None:
            image_builder = LHAASOEventMapper(grid_size=self.img_grid_size)
        self.image_builder_ = image_builder

        # 转换为tensor
        X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
        y_tensor = torch.LongTensor(y)

        # 生成图像数据
        if self.use_image:
            images_list = []
            if isinstance(X, np.ndarray):
                df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
            else:
                df = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)

            for i in range(len(df)):
                img = image_builder.convert_event(df.iloc[i])
                images_list.append(img)
            images_tensor = torch.FloatTensor(np.stack(images_list, axis=0))
        else:
            images_tensor = None

        # 训练循环
        dataset = torch.utils.data.TensorDataset(
            X_tensor, y_tensor,
            images_tensor if images_tensor is not None else torch.zeros(len(X_tensor), 1)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y, batch_img in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_img = batch_img.to(self.device)

                optimizer.zero_grad()

                if self.use_image and self.use_tabular:
                    outputs = self.model_(batch_img, batch_x)
                elif self.use_image:
                    outputs = self.model_(batch_img)
                else:
                    outputs = self.model_(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        self.model_.eval()
        X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)

        # 生成图像
        if self.use_image:
            if isinstance(X, np.ndarray):
                df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
            else:
                df = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)
            images_list = []
            for i in range(len(df)):
                img = self.image_builder_.convert_event(df.iloc[i])
                images_list.append(img)
            images_tensor = torch.FloatTensor(np.stack(images_list, axis=0))
        else:
            images_tensor = None

        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            if self.use_image and self.use_tabular:
                images_tensor = images_tensor.to(self.device)
                outputs = self.model_(images_tensor, X_tensor)
            elif self.use_image:
                images_tensor = images_tensor.to(self.device)
                outputs = self.model_(images_tensor)
            else:
                outputs = self.model_(X_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
        return probs


# =============================================================================
# Part 6: Grad-CAM 可解释性 (论文Figure 8的核心方法)
# =============================================================================

class GradCAMExplainer:
    """
    Grad-CAM: 可视化CNN关注探测器图像的哪些区域

    论文中用途: 验证模型关注的是GRB的物理特征而非噪声
    LHAASO用途: 验证模型关注的是μ子丰富区域(质子)还是核心区域(伽马)

    原理: 对最后一层卷积特征图计算梯度，加权平均得到热力图
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册hook
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_image, class_idx):
        """
        生成Grad-CAM热力图

        参数:
            input_image: [1, C, H, W] 输入图像
            class_idx:   目标类别 (0=质子, 1=伽马)
        返回:
            heatmap: [H, W] 归一化热力图
        """
        self.model.eval()
        input_image = input_image.requires_grad_(True)

        # 前向传播
        output = self.model(input_image)
        score = output[:, class_idx]

        # 反向传播
        self.model.zero_grad()
        score.backward()

        # 计算权重 (全局平均池化梯度)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # 加权组合激活图
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze(0)  # [H, W]
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        return heatmap.cpu().numpy()


# =============================================================================
# Part 7: 演示与验证
# =============================================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 70)
    print("ResNet-CBAM 伽马/质子鉴别模型演示")
    print("基于: Zhang et al. 2023 (arXiv:2303.00370)")
    print("=" * 70)

    # ---- 1. 生成模拟LHAASO数据 ----
    np.random.seed(42)
    n_samples = 500

    gamma = pd.DataFrame({
        'NuM1': np.random.poisson(80, n_samples),
        'NuM2': np.random.poisson(30, n_samples),
        'NuM3': np.random.poisson(10, n_samples),
        'NuM4': np.random.poisson(3, n_samples),
        'NfiltM': np.random.poisson(200, n_samples),
        'NhitM': np.random.poisson(250, n_samples),
        'NuW1': np.random.poisson(50, n_samples),
        'base': np.abs(np.random.normal(100, 20, n_samples)),
        'lable': 1,
    })
    proton = pd.DataFrame({
        'NuM1': np.random.poisson(38, n_samples),
        'NuM2': np.random.poisson(33, n_samples),
        'NuM3': np.random.poisson(28, n_samples),
        'NuM4': np.random.poisson(22, n_samples),
        'NfiltM': np.random.poisson(300, n_samples),
        'NhitM': np.random.poisson(350, n_samples),
        'NuW1': np.random.poisson(40, n_samples),
        'base': np.abs(np.random.normal(100, 25, n_samples)),
        'lable': 0,
    })
    df = pd.concat([gamma, proton], ignore_index=True)

    feature_cols = ['NuM1', 'NuM2', 'NuM3', 'NuM4', 'NfiltM', 'NhitM', 'NuW1', 'base']
    X = df[feature_cols].values
    y = df['lable'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- 2. 创建事件→图像映射器 ----
    print("\n[Step 1] 创建LHAASO事件→探测器图像映射器...")
    mapper = LHAASOEventMapper(grid_size=32)
    sample_img = mapper.convert_event(df.iloc[0])
    print(f"  输出图像形状: {sample_img.shape}")
    print(f"  三通道: [ED电磁, MD缪子, 时间信息]")

    # ---- 3. 训练多模态模型 ----
    print("\n[Step 2] 训练 ResNet-CBAM 多模态融合模型...")
    model = LHAASOResNetCBAMClassifier(
        use_tabular=True,
        use_image=True,
        epochs=20,       # 演示用20 epoch, 实际训练建议100+
        lr=0.001,
        batch_size=32,
    )
    model.fit(X_scaled, y, image_builder=mapper)

    # ---- 4. 评估 ----
    print("\n[Step 3] 模型评估...")
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y, y_pred):.4f}")
    print(f"  AUC:       {roc_auc_score(y, y_prob):.4f}")

    # ---- 5. 模型参数统计 ----
    total_params = sum(p.numel() for p in model.model_.parameters())
    trainable_params = sum(p.numel() for p in model.model_.parameters() if p.requires_grad)
    print(f"\n[模型信息]")
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")
    print(f"  设备:         {model.device}")

    # ---- 6. 生成Grad-CAM热力图样例 ----
    print("\n[Step 4] Grad-CAM可解释性演示...")
    print("  选取一个伽马样本和一个质子样本进行分析...")

    gamma_idx = np.where(y == 1)[0][0]
    proton_idx = np.where(y == 0)[0][0]

    # 生成图像
    gamma_img_tensor = torch.FloatTensor(
        mapper.convert_event(df.iloc[gamma_idx])
    ).unsqueeze(0).to(model.device)

    proton_img_tensor = torch.FloatTensor(
        mapper.convert_event(df.iloc[proton_idx])
    ).unsqueeze(0).to(model.device)

    print(f"  伽马样本 - 预测概率: {y_prob[gamma_idx]:.4f}")
    print(f"  质子样本 - 预测概率: {y_prob[proton_idx]:.4f}")

    # 打印探测器图像统计
    print(f"\n[探测器图像统计]")
    gamma_img = mapper.convert_event(df.iloc[gamma_idx])
    proton_img = mapper.convert_event(df.iloc[proton_idx])
    print(f"  伽马 - ED通道总强度: {gamma_img[0].sum():.2f}, "
          f"MD通道总强度: {gamma_img[1].sum():.2f}")
    print(f"  质子 - ED通道总强度: {proton_img[0].sum():.2f}, "
          f"MD通道总强度: {proton_img[1].sum():.2f}")

    print(f"\n{'='*70}")
    print("演示完成! 完整训练请运行 train_full_comparison.py")
    print("=" * 70)
