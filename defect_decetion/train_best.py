import os
import numpy as np
import random
import cv2
from collections import defaultdict
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import albumentations as A
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv


# ================================
# 增强的数据集类 - 专注缺陷区域
# ================================

class DefectFocusedDataset(Dataset):
    """专注缺陷区域的数据集"""

    def __init__(self, image_paths, transform=None, focus_on_defects=True):
        self.image_paths = image_paths
        self.transform = transform
        self.focus_on_defects = focus_on_defects
        self.categories = ['oil', 'scratch', 'stain']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # 获取标签
        label = self._get_label_from_folder(image_path)

        # 根据缺陷类型进行特定的预处理
        if self.focus_on_defects:
            image_np = self._enhance_defect_visibility(image_np, label)

        # 应用数据增强
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image_np)
                image_np = augmented['image']

        # 转换为张量
        image = transforms.ToTensor()(image_np)

        return image, label, image_path

    def _get_label_from_folder(self, image_path):
        folder_name = os.path.basename(os.path.dirname(image_path))
        if folder_name in self.categories:
            return self.categories.index(folder_name)
        else:
            raise ValueError(f"Unexpected folder name: {folder_name}")

    def _enhance_defect_visibility(self, image, label):
        """根据缺陷类型增强特定特征的可见性"""

        if label == 0:  # oil - 油污：增强透明度差异
            return self._enhance_oil_defects(image)
        elif label == 1:  # scratch - 划痕：增强细长浅白痕
            return self._enhance_scratch_defects(image)
        elif label == 2:  # stain - 斑点：增强白色小点
            return self._enhance_stain_defects(image)
        else:
            return image

    def _enhance_oil_defects(self, image):
        """增强油污特征：较大区域透明状"""
        # 转换为LAB色彩空间，L通道对亮度敏感
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # 增强亮度对比度，突出透明区域
        l_enhanced = cv2.equalizeHist(l_channel)
        lab[:, :, 0] = l_enhanced

        # 转回RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 额外的对比度增强
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)

        return enhanced

    def _enhance_scratch_defects(self, image):
        """增强划痕特征：细长浅白痕"""
        # 转换为灰度图进行边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 使用定向滤波器检测线性特征
        kernel_horizontal = np.array([[-1, -1, -1],
                                      [2, 2, 2],
                                      [-1, -1, -1]], dtype=np.float32)

        kernel_vertical = np.array([[-1, 2, -1],
                                    [-1, 2, -1],
                                    [-1, 2, -1]], dtype=np.float32)

        # 检测水平和垂直线性特征
        horizontal_edges = cv2.filter2D(gray, -1, kernel_horizontal)
        vertical_edges = cv2.filter2D(gray, -1, kernel_vertical)

        # 合并边缘检测结果
        edges = np.maximum(horizontal_edges, vertical_edges)
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        # 将边缘信息融合回原图像
        enhanced = image.copy()
        for i in range(3):  # RGB三个通道
            channel = enhanced[:, :, i]
            # 增强有边缘的区域
            mask = edges > 30
            channel[mask] = np.clip(channel[mask] * 1.3 + 20, 0, 255)
            enhanced[:, :, i] = channel

        return enhanced

    def _enhance_stain_defects(self, image):
        """增强斑点特征：白色小点"""
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 检测白色区域（高亮度，低饱和度）
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 形态学操作去除噪声，保留小的白色斑点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # 增强白色斑点区域
        enhanced = image.copy()
        enhanced[white_mask > 0] = np.clip(enhanced[white_mask > 0] * 1.2 + 15, 0, 255)

        # 整体轻微增强对比度
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=5)

        return enhanced


# ================================
# 注意力机制模块
# ================================

class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        out = x * self.channel_attention(x)
        # 空间注意力
        out = out * self.spatial_attention(out)
        return out


# ================================
# 增强的ResNet分类器
# ================================

class EnhancedResNetClassifier(nn.Module):
    """增强的ResNet分类器，集成注意力机制"""

    def __init__(self, num_classes=3, pretrained=True, use_attention=True):
        super(EnhancedResNetClassifier, self).__init__()

        # 使用ResNet50作为主干网络
        self.backbone = models.resnet50(pretrained=pretrained)

        # 移除原始的全连接层和平均池化层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # 获取特征图的通道数
        self.feature_channels = 2048

        # 添加注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(self.feature_channels)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # 特征图可视化钩子（用于调试）
        self.feature_maps = None
        self.attention_maps = None

    def forward(self, x):
        # 提取特征
        features = self.backbone(x)  # [B, 2048, H, W]

        # 保存特征图（用于可视化）
        self.feature_maps = features.detach()

        # 应用注意力机制
        if self.use_attention:
            attended_features = self.attention(features)
            self.attention_maps = attended_features.detach()
        else:
            attended_features = features

        # 全局平均池化
        pooled_features = self.global_avg_pool(attended_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # 分类
        output = self.classifier(pooled_features)

        return output


# ================================
# 焦点损失函数
# ================================

class FocalLoss(nn.Module):
    """
    焦点损失函数，用于处理困难样本和类别不平衡
    着重关注难以分类的样本
    """

    def __init__(self, alpha=1.0, gamma=2.0, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ================================
# 数据增强策略 - 专注缺陷区域
# ================================

def get_defect_focused_augmentations(is_training=True):
    """获取专注缺陷区域的数据增强策略"""

    if is_training:
        return A.Compose([
            # 基础变换
            A.Resize(224, 224),

            # 几何变换 - 保持缺陷形状特征
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=45, p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),

            # 强度和对比度变换 - 增强缺陷可见性
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.3,
                p=0.7
            ),
            A.CLAHE(clip_limit=2.0, p=0.5),  # 限制对比度自适应直方图均衡
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),

            # 噪声和模糊 - 提高鲁棒性
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),

            # 颜色变换 - 适应不同光照条件
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4
            ),

            # 细节增强变换
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.2), p=0.3),
            A.UnsharpMask(blur_limit=3, p=0.2),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            # 验证和测试时可以添加轻微的TTA（测试时数据增强）
            # A.CLAHE(clip_limit=1.0, p=1.0),
        ])


# ================================
# 改进的训练函数
# ================================

def train_enhanced_classification(model, train_loader, val_loader, device, num_epochs=50):
    """训练增强的分类模型"""

    # 使用焦点损失
    criterion = FocalLoss(alpha=1.0, gamma=2.0, num_classes=3)

    # 使用AdamW优化器，更好的权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    patience_counter = 0
    patience = 10

    print("🚀 开始增强分类模型训练...")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 使用tqdm显示进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 更新进度条
            current_acc = train_correct / train_total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)

        # 学习率调度
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
            }, "best_enhanced_model.pth")
            print(f"  💾 保存最佳模型 (验证准确率: {val_accuracy:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= patience:
            print(f"  ⏰ 验证准确率连续 {patience} 个epoch未改善，提前停止训练")
            break

        print()

    return train_losses, val_accuracies, best_accuracy


# ================================
# 特征可视化函数
# ================================

def visualize_attention_maps(model, dataloader, device, save_dir="attention_visualizations"):
    """可视化注意力图"""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    categories = ['oil', 'scratch', 'stain']
    category_names = ['油污', '划痕', '斑点']

    with torch.no_grad():
        for batch_idx, (images, labels, image_paths) in enumerate(dataloader):
            if batch_idx >= 3:  # 只可视化前几个batch
                break

            images = images.to(device)
            outputs = model(images)

            # 获取注意力图
            if hasattr(model, 'attention_maps') and model.attention_maps is not None:
                attention_maps = model.attention_maps.cpu()

                for i in range(min(4, images.size(0))):  # 每个batch最多4张图
                    # 原始图像
                    original_img = images[i].cpu().permute(1, 2, 0).numpy()
                    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

                    # 注意力图 (取通道平均)
                    attention_map = torch.mean(attention_maps[i], dim=0).numpy()
                    attention_map = cv2.resize(attention_map, (224, 224))
                    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

                    # 创建热力图
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.imshow(original_img)
                    plt.title('原始图像')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(attention_map, cmap='hot')
                    plt.title('注意力图')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(original_img)
                    plt.imshow(attention_map, alpha=0.6, cmap='hot')
                    plt.title('叠加显示')
                    plt.axis('off')

                    # 保存图像
                    label = labels[i].item()
                    filename = f"attention_batch{batch_idx}_img{i}_{category_names[label]}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
                    plt.close()

    print(f"📊 注意力可视化图已保存到: {save_dir}")


# ================================
# 数据集划分函数（直接整合）
# ================================

def create_balanced_dataset_split(base_dir, test_samples_per_class=30, train_ratio=0.8, random_seed=42):
    """
    创建平衡的数据集划分，确保测试集每个类别有固定数量的样本

    Args:
        base_dir: 数据集根目录
        test_samples_per_class: 每个类别在测试集中的样本数量
        train_ratio: 剩余数据中训练集的比例
        random_seed: 随机种子

    Returns:
        train_paths, val_paths, test_paths: 三个数据集的路径列表
    """

    # 设置随机种子
    random.seed(random_seed)

    # 按类别收集图像路径
    categories = ['oil', 'scratch', 'stain']
    category_paths = defaultdict(list)

    print("📊 收集各类别数据...")
    for category in categories:
        category_folder = os.path.join(base_dir, category)
        if os.path.exists(category_folder):
            for image_file in os.listdir(category_folder):
                if image_file.lower().endswith(('.jpg', '.jpeg')):
                    category_paths[category].append(os.path.join(category_folder, image_file))

    # 显示各类别数据统计
    for category in categories:
        print(f"  {category}: {len(category_paths[category])} 张图片")

    # 检查每个类别是否有足够的数据
    for category in categories:
        if len(category_paths[category]) < test_samples_per_class:
            raise ValueError(f"类别 '{category}' 只有 {len(category_paths[category])} 张图片，"
                             f"无法提供 {test_samples_per_class} 张测试样本")

    # 为每个类别分别划分数据
    train_paths = []
    val_paths = []
    test_paths = []

    print(f"\n🎯 按类别划分数据 (测试集每类 {test_samples_per_class} 张)...")

    for category in categories:
        category_images = category_paths[category].copy()
        random.shuffle(category_images)  # 随机打乱

        # 先取出测试集
        test_images = category_images[:test_samples_per_class]
        remaining_images = category_images[test_samples_per_class:]

        # 剩余数据按比例划分训练集和验证集
        remaining_count = len(remaining_images)
        train_count = int(remaining_count * train_ratio)

        train_images = remaining_images[:train_count]
        val_images = remaining_images[train_count:]

        # 添加到总列表
        train_paths.extend(train_images)
        val_paths.extend(val_images)
        test_paths.extend(test_images)

        print(f"  {category}:")
        print(f"    训练集: {len(train_images)} 张")
        print(f"    验证集: {len(val_images)} 张")
        print(f"    测试集: {len(test_images)} 张")

    # 最终打乱各数据集
    random.shuffle(train_paths)
    random.shuffle(val_paths)
    random.shuffle(test_paths)

    # 统计信息
    total_samples = len(train_paths) + len(val_paths) + len(test_paths)
    print(f"\n📈 最终数据集统计:")
    print(f"  训练集: {len(train_paths)} 张 ({len(train_paths) / total_samples * 100:.1f}%)")
    print(f"  验证集: {len(val_paths)} 张 ({len(val_paths) / total_samples * 100:.1f}%)")
    print(f"  测试集: {len(test_paths)} 张 ({len(test_paths) / total_samples * 100:.1f}%)")
    print(f"  总计: {total_samples} 张")

    # 验证测试集的类别分布
    print(f"\n✅ 测试集类别分布验证:")
    test_category_count = defaultdict(int)
    for path in test_paths:
        category = os.path.basename(os.path.dirname(path))
        test_category_count[category] += 1

    for category in categories:
        count = test_category_count[category]
        print(f"  {category}: {count} 张 {'✓' if count == test_samples_per_class else '✗'}")

    return train_paths, val_paths, test_paths


# ================================
# 评估和分析函数（直接整合）
# ================================

def evaluate_classification_model(model, test_loader, device):
    """评估分类模型"""
    model.eval()

    true_labels = []
    predicted_labels = []
    all_image_paths = []
    prediction_probs = []

    categories = ['oil', 'scratch', 'stain']
    category_names = ['油污', '划痕', '斑点']

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            images, labels, image_paths = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 获取概率和预测结果
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            all_image_paths.extend(image_paths)
            prediction_probs.extend(probs.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)

    print("\n" + "=" * 60)
    print("🎯 图像分类评估结果")
    print("=" * 60)
    print(f"总体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 详细报告
    print("\n分类报告:")
    print(classification_report(true_labels, predicted_labels, target_names=category_names))

    # 混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    print("\n混淆矩阵:")
    print("真实\\预测", "  ".join([f"{name:>6}" for name in category_names]))
    for i, true_name in enumerate(category_names):
        row_str = f"{true_name:>8}: "
        row_str += "  ".join([f"{cm[i, j]:>6}" for j in range(len(category_names))])
        print(row_str)

    return accuracy, true_labels, predicted_labels, all_image_paths, prediction_probs


def analyze_test_results(test_paths, true_labels, predicted_labels):
    """分析测试集结果的详细情况"""

    categories = ['oil', 'scratch', 'stain']
    category_names = ['油污', '划痕', '斑点']

    print("\n" + "=" * 60)
    print("📊 测试集详细分析 (每类30张)")
    print("=" * 60)

    # 按类别分析准确率
    for i, (category, category_name) in enumerate(zip(categories, category_names)):
        class_true = [j for j, label in enumerate(true_labels) if label == i]
        class_predicted = [predicted_labels[j] for j in class_true]
        class_correct = sum(1 for pred in class_predicted if pred == i)

        accuracy = class_correct / len(class_true) if len(class_true) > 0 else 0

        print(f"\n{category_name} ({category}):")
        print(f"  测试样本: {len(class_true)} 张")
        print(f"  预测正确: {class_correct} 张")
        print(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # 显示错误预测的详情
        if class_correct < len(class_true):
            error_count = len(class_true) - class_correct
            print(f"  错误预测: {error_count} 张")

            # 统计错误预测到哪些类别
            error_distribution = defaultdict(int)
            for pred in class_predicted:
                if pred != i:
                    error_distribution[pred] += 1

            for wrong_class, count in error_distribution.items():
                wrong_name = category_names[wrong_class]
                print(f"    误判为{wrong_name}: {count} 张")


def save_results_to_csv(image_paths, true_labels, predicted_labels, prediction_probs,
                        output_file="enhanced_classification_results.csv"):
    """保存测试结果到CSV文件"""

    categories = ['oil', 'scratch', 'stain']
    category_names = ['油污', '划痕', '斑点']

    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['图片路径', '真实类别', '预测类别', '是否正确', '置信度']
            header.extend([f'{name}_概率' for name in category_names])
            writer.writerow(header)

            # 写入数据
            for i, (img_path, true_label, pred_label, probs) in enumerate(
                    zip(image_paths, true_labels, predicted_labels, prediction_probs)
            ):
                is_correct = "正确" if true_label == pred_label else "错误"
                confidence = probs[pred_label]

                row = [
                    os.path.basename(img_path),
                    category_names[true_label],
                    category_names[pred_label],
                    is_correct,
                    f"{confidence:.4f}"
                ]

                # 添加各类别概率
                for prob in probs:
                    row.append(f"{prob:.4f}")

                writer.writerow(row)

        print(f"✅ 测试结果已保存到: {output_file}")

    except Exception as e:
        print(f"❌ 保存CSV文件失败: {e}")


def plot_training_curves(train_losses, val_accuracies, save_path="enhanced_training_curves.png"):
    """绘制训练曲线"""

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 训练损失曲线
    ax1.plot(train_losses, 'b-', label='训练损失')
    ax1.set_title('训练损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 验证准确率曲线
    ax2.plot(val_accuracies, 'r-', label='验证准确率')
    ax2.set_title('验证准确率变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 训练曲线已保存到: {save_path}")


# ================================
# 主函数 - 整合所有改进
# ================================

def main():
    """主训练函数 - 集成所有优化策略"""

    # 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 数据路径
    base_dir = "MSD-US/train_set"

    # 数据集划分
    train_paths, val_paths, test_paths = create_balanced_dataset_split(
        base_dir=base_dir,
        test_samples_per_class=30,
        train_ratio=0.8,
        random_seed=42
    )

    # 获取专门的数据增强策略
    train_transform = get_defect_focused_augmentations(is_training=True)
    val_transform = get_defect_focused_augmentations(is_training=False)

    print("🎯 创建专注缺陷区域的数据集...")

    # 创建增强的数据集
    train_dataset = DefectFocusedDataset(train_paths, train_transform, focus_on_defects=True)
    val_dataset = DefectFocusedDataset(val_paths, val_transform, focus_on_defects=True)
    test_dataset = DefectFocusedDataset(test_paths, val_transform, focus_on_defects=True)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # 减小batch size以适应更复杂的模型
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"📦 增强数据加载器创建完成:")
    print(f"  训练批次: {len(train_loader)} 批")
    print(f"  验证批次: {len(val_loader)} 批")
    print(f"  测试批次: {len(test_loader)} 批")

    # 创建增强模型
    model = EnhancedResNetClassifier(num_classes=3, pretrained=True, use_attention=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🎯 增强模型参数量: {trainable_params:,} / {total_params:,}")

    # 训练模型
    print("\n🚀 开始增强训练...")
    train_losses, val_accuracies, best_accuracy = train_enhanced_classification(
        model, train_loader, val_loader, device, num_epochs=50
    )

    # 绘制训练曲线
    plot_training_curves(train_losses, val_accuracies, "enhanced_training_curves.png")

    # 加载最佳模型并测试
    print("\n🔍 加载最佳增强模型进行测试...")
    checkpoint = torch.load("best_enhanced_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # 测试评估
    accuracy, true_labels, predicted_labels, image_paths, prediction_probs = evaluate_classification_model(
        model, test_loader, device
    )

    # 详细分析
    analyze_test_results(test_paths, true_labels, predicted_labels)

    # 保存结果
    save_results_to_csv(
        image_paths, true_labels, predicted_labels, prediction_probs,
        "enhanced_classification_results.csv"
    )

    # 可视化注意力图
    print("\n🎨 生成注意力可视化图...")
    visualize_attention_maps(model, test_loader, device)

    # 最终总结
    print("\n" + "=" * 60)
    print("🎉 增强训练完成!")
    print("=" * 60)
    print("📁 生成的文件:")

    files = [
        "best_enhanced_model.pth",
        "enhanced_training_curves.png",
        "enhanced_classification_results.csv",
        "attention_visualizations/"
    ]

    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (未生成)")

    print(f"\n🎯 最终测试准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"🏆 最佳验证准确率: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")

    print("\n🔧 主要优化策略:")
    print("  ✅ 缺陷特征增强预处理")
    print("  ✅ CBAM注意力机制")
    print("  ✅ 焦点损失函数")
    print("  ✅ 专门的数据增强策略")
    print("  ✅ 注意力可视化")
    print("  ✅ 早停和学习率调度")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()