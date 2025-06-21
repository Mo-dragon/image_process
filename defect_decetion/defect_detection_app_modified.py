#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缺陷检测用户端应用程序（修改版 - 移除oil选项和UI中的三分类显示）
支持单张图片预测、批量处理和可视化分析
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# GUI相关
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️ GUI模块不可用，仅支持命令行模式")


# ================================
# 模型结构定义（与训练代码保持一致）
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
    """卷积块注意力模块"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class EnhancedResNetClassifier(nn.Module):
    """增强的ResNet分类器（修改为2分类：划痕和斑点）"""

    def __init__(self, num_classes=2, pretrained=True, use_attention=True):
        super(EnhancedResNetClassifier, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.feature_channels = 2048
        self.use_attention = use_attention

        if use_attention:
            self.attention = CBAM(self.feature_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.feature_maps = None
        self.attention_maps = None

    def forward(self, x):
        features = self.backbone(x)
        self.feature_maps = features.detach()

        if self.use_attention:
            attended_features = self.attention(features)
            self.attention_maps = attended_features.detach()
        else:
            attended_features = features

        pooled_features = self.global_avg_pool(attended_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)

        return output


# ================================
# 图像预处理函数（移除oil相关）
# ================================

def enhance_defect_visibility(image, defect_type=None):
    """增强缺陷特征的可见性（移除oil类型）"""
    if defect_type == 'scratch':
        return enhance_scratch_defects(image)
    elif defect_type == 'stain':
        return enhance_stain_defects(image)
    else:
        # 通用增强
        return enhance_general_defects(image)


def enhance_scratch_defects(image):
    """增强划痕特征"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel_horizontal = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    kernel_vertical = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)

    horizontal_edges = cv2.filter2D(gray, -1, kernel_horizontal)
    vertical_edges = cv2.filter2D(gray, -1, kernel_vertical)
    edges = np.maximum(horizontal_edges, vertical_edges)
    edges = np.clip(edges, 0, 255).astype(np.uint8)

    enhanced = image.copy()
    for i in range(3):
        channel = enhanced[:, :, i]
        mask = edges > 30
        channel[mask] = np.clip(channel[mask] * 1.3 + 20, 0, 255)
        enhanced[:, :, i] = channel

    return enhanced


def enhance_stain_defects(image):
    """增强斑点特征"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    enhanced = image.copy()
    enhanced[white_mask > 0] = np.clip(enhanced[white_mask > 0] * 1.2 + 15, 0, 255)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=5)

    return enhanced


def enhance_general_defects(image):
    """通用缺陷增强"""
    # CLAHE增强
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 轻微锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def get_prediction_transform():
    """获取预测时的图像变换"""
    return A.Compose([
        A.Resize(224, 224),
        A.CLAHE(clip_limit=1.0, p=0.5),
    ])


# ================================
# 3分类到2分类的概率转换
# ================================

def convert_3class_to_2class(probabilities_3class):
    """
    将3分类概率转换为2分类概率
    原始：[oil, scratch, stain]
    转换后：[scratch, stain]
    oil的概率分配给scratch和stain
    """
    if len(probabilities_3class) != 3:
        raise ValueError("输入概率数组长度必须为3")

    oil_prob, scratch_prob, stain_prob = probabilities_3class

    # 将oil概率按比例分配给scratch和stain
    # 如果scratch和stain概率都为0，则平均分配
    total_non_oil = scratch_prob + stain_prob

    if total_non_oil > 0:
        scratch_ratio = scratch_prob / total_non_oil
        stain_ratio = stain_prob / total_non_oil
    else:
        scratch_ratio = 0.5
        stain_ratio = 0.5

    # 重新分配概率
    new_scratch_prob = scratch_prob + oil_prob * scratch_ratio
    new_stain_prob = stain_prob + oil_prob * stain_ratio

    # 归一化确保概率和为1
    total = new_scratch_prob + new_stain_prob
    if total > 0:
        new_scratch_prob /= total
        new_stain_prob /= total

    return np.array([new_scratch_prob, new_stain_prob])


# ================================
# 缺陷检测器类（修改版）
# ================================

class DefectDetector:
    """缺陷检测器主类（修改版 - 移除oil）"""

    def __init__(self, model_path, device=None):
        # 更新为2分类
        self.categories = ['scratch', 'stain']
        self.category_names = ['划痕', '斑点']
        self.colors = ['#4ECDC4', '#45B7D1']

        # 设备设置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"🖥️ 使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model(model_path)
        self.transform = get_prediction_transform()

        print("✅ 缺陷检测器初始化完成（2分类模式：划痕/斑点）")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        try:
            print(f"📦 加载模型: {model_path}")

            # 首先加载为3分类模型
            model_3class = EnhancedResNetClassifier(num_classes=3, pretrained=False, use_attention=True)

            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                model_3class.load_state_dict(checkpoint['model_state_dict'])
                print(f"📊 原始模型最佳验证准确率: {checkpoint.get('best_accuracy', 'N/A')}")
            else:
                model_3class.load_state_dict(checkpoint)

            model_3class.to(self.device)
            model_3class.eval()

            # 检查分类器结构，找到正确的最后一层
            print("🔍 检查模型结构...")
            classifier_layers = []
            for name, module in model_3class.classifier.named_children():
                classifier_layers.append((name, module))
                print(f"  {name}: {module}")

            # 找到最后的Linear层
            last_linear_name = None
            for name, module in reversed(classifier_layers):
                if isinstance(module, torch.nn.Linear):
                    last_linear_name = name
                    print(f"📍 找到最后的分类层: classifier.{name}")
                    break

            if last_linear_name is None:
                raise ValueError("无法找到分类器中的Linear层")

            # 创建2分类模型
            model_2class = EnhancedResNetClassifier(num_classes=2, pretrained=False, use_attention=True)

            # 复制除最后分类层外的所有权重
            model_2class_dict = model_2class.state_dict()
            model_3class_dict = model_3class.state_dict()

            last_layer_weight_key = f'classifier.{last_linear_name}.weight'
            last_layer_bias_key = f'classifier.{last_linear_name}.bias'

            for key in model_2class_dict.keys():
                if key not in [last_layer_weight_key, last_layer_bias_key]:
                    if key in model_3class_dict:
                        model_2class_dict[key] = model_3class_dict[key]
                    else:
                        print(f"⚠️ 权重键不匹配: {key}")

            # 处理最后的分类层权重
            if last_layer_weight_key in model_3class_dict:
                old_weight = model_3class_dict[last_layer_weight_key]  # [3, 512]
                old_bias = model_3class_dict[last_layer_bias_key]  # [3]

                print(f"🔧 转换分类层权重: {old_weight.shape} -> [2, {old_weight.size(1)}]")

                # 创建新的2分类权重
                new_weight = torch.zeros(2, old_weight.size(1))
                new_bias = torch.zeros(2)

                # scratch (原索引1 -> 新索引0)，添加30%的oil权重
                new_weight[0] = old_weight[1] + 0.3 * old_weight[0]
                new_bias[0] = old_bias[1] + 0.3 * old_bias[0]

                # stain (原索引2 -> 新索引1)，添加70%的oil权重
                new_weight[1] = old_weight[2] + 0.7 * old_weight[0]
                new_bias[1] = old_bias[2] + 0.7 * old_bias[0]

                model_2class_dict[last_layer_weight_key] = new_weight
                model_2class_dict[last_layer_bias_key] = new_bias

                print(f"✅ 权重转换完成")
            else:
                print(f"⚠️ 未找到分类层权重: {last_layer_weight_key}")

            # 加载权重到2分类模型
            model_2class.load_state_dict(model_2class_dict)
            model_2class.to(self.device)
            model_2class.eval()

            # 保存3分类模型用于概率转换
            self.model_3class = model_3class

            print("✅ 模型适配完成：3分类 -> 2分类")
            return model_2class

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 如果适配失败，尝试直接使用3分类模型
            print("🔄 尝试直接使用3分类模型...")
            try:
                model = EnhancedResNetClassifier(num_classes=3, pretrained=False, use_attention=True)
                checkpoint = torch.load(model_path, map_location=self.device)

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                model.to(self.device)
                model.eval()

                # 直接使用3分类模型，在预测时转换概率
                self.model_3class = model
                self.use_3class_direct = True

                print("✅ 使用3分类模型，预测时转换概率")
                return model

            except Exception as e2:
                print(f"❌ 3分类模型加载也失败: {e2}")
                raise e

    def preprocess_image(self, image_path, enhance_type=None):
        """预处理单张图像"""
        try:
            # 读取图像
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            # 缺陷增强（移除oil选项）
            if enhance_type and enhance_type != 'oil':
                image_np = enhance_defect_visibility(image_np, enhance_type)

            # 数据增强
            augmented = self.transform(image=image_np)
            image_tensor = transforms.ToTensor()(augmented['image'])

            return image_tensor.unsqueeze(0), image_np

        except Exception as e:
            print(f"❌ 图像预处理失败: {e}")
            raise

    def predict_single(self, image_path, enhance_type=None, return_attention=False):
        """预测单张图像（使用3分类模型然后转换为2分类）"""
        try:
            # 预处理
            image_tensor, original_image = self.preprocess_image(image_path, enhance_type)
            image_tensor = image_tensor.to(self.device)

            # 推理
            with torch.no_grad():
                start_time = time.time()

                # 使用3分类模型获取原始预测
                outputs_3class = self.model_3class(image_tensor)
                probabilities_3class = torch.softmax(outputs_3class, dim=1)[0].cpu().numpy()

                # 转换为2分类概率
                probabilities_2class = convert_3class_to_2class(probabilities_3class)

                # 获取2分类预测结果
                predicted_class = np.argmax(probabilities_2class)
                confidence = probabilities_2class[predicted_class]

                inference_time = time.time() - start_time

                # 构建结果
                result = {
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'predicted_label': self.categories[predicted_class],
                    'predicted_name': self.category_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities_2class,
                    'probabilities_3class_original': probabilities_3class,  # 保留原始3分类概率用于调试
                    'inference_time': inference_time,
                    'original_image': original_image
                }

                # 添加注意力图
                if return_attention and hasattr(self.model_3class, 'attention_maps'):
                    if self.model_3class.attention_maps is not None:
                        attention_map = torch.mean(self.model_3class.attention_maps[0], dim=0).cpu().numpy()
                        attention_map = cv2.resize(attention_map, (224, 224))
                        attention_map = (attention_map - attention_map.min()) / (
                                attention_map.max() - attention_map.min())
                        result['attention_map'] = attention_map

                return result

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            raise

    def predict_batch(self, image_paths, enhance_type=None, progress_callback=None):
        """批量预测"""
        results = []
        total_images = len(image_paths)

        print(f"🔍 开始批量预测 {total_images} 张图像...")

        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path, enhance_type)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total_images)

                if (i + 1) % 10 == 0:
                    print(f"  已处理: {i + 1}/{total_images}")

            except Exception as e:
                print(f"⚠️ 跳过图像 {image_path}: {e}")
                continue

        print(f"✅ 批量预测完成，成功处理 {len(results)} 张图像")
        return results

    def generate_report(self, results, save_path=None):
        """生成预测报告"""
        if not results:
            print("⚠️ 没有预测结果可生成报告")
            return None

        # 统计信息
        total_images = len(results)
        category_counts = {name: 0 for name in self.category_names}
        avg_confidence = 0
        avg_inference_time = 0

        for result in results:
            category_counts[result['predicted_name']] += 1
            avg_confidence += result['confidence']
            avg_inference_time += result['inference_time']

        avg_confidence /= total_images
        avg_inference_time /= total_images

        # 生成报告
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': '2-class (scratch/stain)',
            'total_images': total_images,
            'category_distribution': category_counts,
            'average_confidence': avg_confidence,
            'average_inference_time': avg_inference_time,
            'detailed_results': results
        }

        # 保存报告
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                print(f"📊 报告已保存到: {save_path}")
            except Exception as e:
                print(f"❌ 保存报告失败: {e}")

        return report

    def visualize_prediction(self, result, save_path=None, show_attention=True):
        """可视化预测结果"""
        try:
            fig, axes = plt.subplots(1, 3 if show_attention and 'attention_map' in result else 2,
                                     figsize=(15, 5))

            # 原始图像
            axes[0].imshow(result['original_image'])
            axes[0].set_title('原始图像')
            axes[0].axis('off')

            # 预测概率
            probs = result['probabilities']
            bars = axes[1].bar(self.category_names, probs, color=self.colors)
            axes[1].set_title(f'预测结果: {result["predicted_name"]} (置信度: {result["confidence"]:.2%})')
            axes[1].set_ylabel('概率')
            axes[1].set_ylim(0, 1)

            # 添加数值标签
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{prob:.2%}', ha='center', va='bottom')

            # 注意力图
            if show_attention and 'attention_map' in result:
                attention_map = result['attention_map']
                im = axes[2].imshow(result['original_image'])
                axes[2].imshow(attention_map, alpha=0.6, cmap='hot')
                axes[2].set_title('注意力图')
                axes[2].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 可视化结果已保存到: {save_path}")

            plt.show()

        except Exception as e:
            print(f"❌ 可视化失败: {e}")


# ================================
# GUI应用程序类（修改版 - 移除原始三分类选项）
# ================================

if GUI_AVAILABLE:
    class DefectDetectionGUI:
        """缺陷检测GUI应用（修改版 - 移除原始三分类显示选项）"""

        def __init__(self, detector):
            self.detector = detector
            self.current_image_path = None
            self.current_result = None

            # 创建主窗口
            self.root = tk.Tk()
            self.root.title("智能缺陷检测系统 (划痕/斑点)")
            self.root.geometry("1200x800")  # 增大窗口尺寸
            self.root.configure(bg='#f0f0f0')

            # 设置样式和字体
            self.style = ttk.Style()
            self.style.theme_use('clam')

            # 配置字体大小
            self.configure_fonts()

            self.create_widgets()

        def configure_fonts(self):
            """配置界面字体大小"""
            # 定义字体
            self.title_font = ('Arial', 18, 'bold')
            self.label_font = ('Arial', 12, 'bold')
            self.button_font = ('Arial', 11)
            self.text_font = ('Arial', 15)
            self.combo_font = ('Arial', 10)

            # 配置ttk样式的字体
            self.style.configure('Title.TLabel', font=self.title_font)
            self.style.configure('Heading.TLabel', font=self.label_font)
            self.style.configure('Custom.TButton', font=self.button_font, padding=(10, 5))
            self.style.configure('Custom.TLabelframe.Label', font=self.label_font)
            self.style.configure('Custom.TCombobox', font=self.combo_font)
            self.style.configure('Custom.TCheckbutton', font=self.text_font)

        def create_widgets(self):
            """创建GUI组件"""
            # 主框架
            main_frame = ttk.Frame(self.root, padding="15")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # 标题
            title_label = ttk.Label(main_frame, text="智能缺陷检测系统 (划痕/斑点分类)",
                                    style='Title.TLabel')
            title_label.grid(row=0, column=0, columnspan=3, pady=(0, 25))

            # 左侧面板 - 图像显示
            image_frame = ttk.LabelFrame(main_frame, text="图像显示", padding="15",
                                         style='Custom.TLabelframe')
            image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))

            self.image_label = ttk.Label(image_frame, text="请选择图像",
                                         background='white', relief='sunken',
                                         font=self.text_font)
            self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # 中间面板 - 控制按钮
            control_frame = ttk.LabelFrame(main_frame, text="操作控制", padding="15",
                                           style='Custom.TLabelframe')
            control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=8)

            # 选择图像按钮
            select_btn = ttk.Button(control_frame, text="选择图像",
                                    command=self.select_image, style='Custom.TButton')
            select_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

            # 预测按钮
            self.predict_btn = ttk.Button(control_frame, text="开始预测",
                                          command=self.predict_image, state='disabled',
                                          style='Custom.TButton')
            self.predict_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

            # 批量处理按钮
            batch_btn = ttk.Button(control_frame, text="批量处理",
                                   command=self.batch_process, style='Custom.TButton')
            batch_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

            # 右侧面板 - 结果显示
            result_frame = ttk.LabelFrame(main_frame, text="预测结果", padding="15",
                                          style='Custom.TLabelframe')
            result_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(15, 0))

            # 结果文本框
            self.result_text = tk.Text(result_frame, width=35, height=22, wrap=tk.WORD,
                                       font=self.text_font, bg='white', relief='sunken',
                                       borderwidth=2, padx=10, pady=10)
            scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
            self.result_text.configure(yscrollcommand=scrollbar.set)

            self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

            # 配置网格权重
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=2)
            main_frame.columnconfigure(1, weight=1)
            main_frame.columnconfigure(2, weight=1)
            main_frame.rowconfigure(1, weight=1)

            image_frame.columnconfigure(0, weight=1)
            image_frame.rowconfigure(0, weight=1)
            control_frame.columnconfigure(0, weight=1)
            result_frame.columnconfigure(0, weight=1)
            result_frame.rowconfigure(0, weight=1)

        def select_image(self):
            """选择图像文件"""
            file_path = filedialog.askopenfilename(
                title="选择图像文件",
                filetypes=[
                    ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("JPEG文件", "*.jpg *.jpeg"),
                    ("PNG文件", "*.png"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                self.current_image_path = file_path
                self.display_image(file_path)
                self.predict_btn['state'] = 'normal'

        def display_image(self, image_path):
            """显示图像"""
            try:
                # 加载并调整图像大小
                image = Image.open(image_path)
                # 计算缩放比例
                display_size = (400, 300)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)

                # 转换为PhotoImage
                photo = ImageTk.PhotoImage(image)

                # 更新标签
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # 保持引用

            except Exception as e:
                messagebox.showerror("错误", f"无法显示图像: {e}")

        def predict_image(self):
            """预测当前图像"""
            if not self.current_image_path:
                messagebox.showwarning("警告", "请先选择图像")
                return

            try:
                # 执行预测（移除缺陷增强和注意力图选项）
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "🔍 正在预测...\n")
                self.root.update()

                result = self.detector.predict_single(
                    self.current_image_path,
                    enhance_type=None,
                    return_attention=False
                )

                self.current_result = result
                self.display_result(result)

            except Exception as e:
                messagebox.showerror("错误", f"预测失败: {e}")

        def display_result(self, result):
            """显示预测结果（移除原始三分类概率显示）"""
            self.result_text.delete(1.0, tk.END)

            # 基本信息
            self.result_text.insert(tk.END, "📊 预测结果\n")
            self.result_text.insert(tk.END, "=" * 30 + "\n")
            self.result_text.insert(tk.END, f"图像: {os.path.basename(result['image_path'])}\n")
            self.result_text.insert(tk.END, f"预测类别: {result['predicted_name']}\n")
            self.result_text.insert(tk.END, f"置信度: {result['confidence']:.2%}\n")
            self.result_text.insert(tk.END, f"推理耗时: {result['inference_time']:.3f}秒\n\n")

            # 2分类概率
            self.result_text.insert(tk.END, "📈 类别概率:\n")
            for i, (category, prob) in enumerate(zip(self.detector.category_names, result['probabilities'])):
                marker = "👉 " if i == result['predicted_class'] else "   "
                self.result_text.insert(tk.END, f"{marker}{category}: {prob:.2%}\n")

        def batch_process(self):
            """批量处理"""
            folder_path = filedialog.askdirectory(title="选择图像文件夹")
            if not folder_path:
                return

            # 获取所有图像文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_paths = []

            for file_path in Path(folder_path).rglob('*'):
                if file_path.suffix.lower() in image_extensions:
                    image_paths.append(str(file_path))

            if not image_paths:
                messagebox.showwarning("警告", "文件夹中没有找到图像文件")
                return

            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("批量处理进度")
            progress_window.geometry("450x180")
            progress_window.configure(bg='#f0f0f0')

            # 进度窗口标题
            progress_title = ttk.Label(progress_window, text="批量处理进度",
                                       font=self.label_font)
            progress_title.pack(pady=15)

            progress_label = ttk.Label(progress_window, text="正在处理...",
                                       font=self.text_font)
            progress_label.pack(pady=5)

            progress_bar = ttk.Progressbar(progress_window, mode='determinate',
                                           length=350)
            progress_bar.pack(pady=15, padx=25, fill=tk.X)

            def update_progress(current, total):
                progress = (current / total) * 100
                progress_bar['value'] = progress
                progress_label.configure(text=f"正在处理: {current}/{total}")
                progress_window.update()

            try:
                # 执行批量预测（移除缺陷增强选项）
                results = self.detector.predict_batch(
                    image_paths,
                    enhance_type=None,
                    progress_callback=update_progress
                )

                # 保存结果
                save_path = filedialog.asksaveasfilename(
                    title="保存批量处理结果",
                    defaultextension=".json",
                    filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
                )

                if save_path:
                    report = self.detector.generate_report(results, save_path)

                    # 显示统计信息
                    stats_text = f"批量处理完成!\n\n"
                    stats_text += f"总图像数: {len(results)}\n"
                    stats_text += f"平均置信度: {report['average_confidence']:.2%}\n"
                    stats_text += f"平均处理时间: {report['average_inference_time']:.3f}秒\n\n"
                    stats_text += "类别分布:\n"
                    for category, count in report['category_distribution'].items():
                        stats_text += f"  {category}: {count}张\n"

                    messagebox.showinfo("批量处理完成", stats_text)

                progress_window.destroy()

            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("错误", f"批量处理失败: {e}")

        def run(self):
            """运行GUI"""
            self.root.mainloop()


# ================================
# 命令行接口函数（修改版）
# ================================

def create_cli_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="缺陷检测用户端应用程序（修改版 - 移除oil选项）")
    parser.add_argument("--model", "-m", required=True, help="模型文件路径")
    parser.add_argument("--input", "-i", help="输入图像路径或文件夹")
    parser.add_argument("--output", "-o", help="输出结果路径")
    parser.add_argument("--enhance", "-e", choices=["scratch", "stain", "none"],
                        help="缺陷增强类型（移除oil选项）")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理模式")
    parser.add_argument("--gui", "-g", action="store_true", help="启动GUI界面")
    parser.add_argument("--attention", "-a", action="store_true", help="生成注意力图")
    parser.add_argument("--visualize", "-v", action="store_true", help="可视化预测结果")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda"], help="指定设备")
    parser.add_argument("--show-3class", action="store_true", help="显示原始3分类概率")

    return parser


def run_cli_prediction(detector, args):
    """运行命令行预测"""
    if not args.input:
        print("❌ 请指定输入图像路径或文件夹")
        return

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ 输入路径不存在: {input_path}")
        return

    # 单张图像预测
    if input_path.is_file():
        print(f"🔍 预测单张图像: {input_path}")

        try:
            result = detector.predict_single(
                str(input_path),
                enhance_type=args.enhance,
                return_attention=args.attention
            )

            # 显示结果
            print("\n📊 预测结果:")
            print("=" * 50)
            print(f"图像: {result['image_path']}")
            print(f"预测类别: {result['predicted_name']} ({result['predicted_label']})")
            print(f"置信度: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
            print(f"推理时间: {result['inference_time']:.3f}秒")

            print(f"\n各类别概率 (2分类):")
            for i, (category, prob) in enumerate(zip(detector.category_names, result['probabilities'])):
                marker = "👉" if i == result['predicted_class'] else "  "
                print(f"{marker} {category}: {prob:.4f} ({prob * 100:.2f}%)")

            # 显示原始3分类概率（仅在命令行模式下）
            if args.show_3class and 'probabilities_3class_original' in result:
                print(f"\n原始3分类概率:")
                original_categories = ['油污', '划痕', '斑点']
                for category, prob in zip(original_categories, result['probabilities_3class_original']):
                    print(f"   {category}: {prob:.4f} ({prob * 100:.2f}%)")
                print("   (油污概率已合并至划痕和斑点)")

            # 可视化
            if args.visualize:
                save_vis_path = None
                if args.output:
                    save_vis_path = args.output.replace('.json', '_visualization.png')
                detector.visualize_prediction(result, save_vis_path, args.attention)

            # 保存结果
            if args.output:
                report = detector.generate_report([result], args.output)
                print(f"📁 结果已保存到: {args.output}")

        except Exception as e:
            print(f"❌ 预测失败: {e}")

    # 批量预测
    elif input_path.is_dir():
        print(f"📁 批量预测文件夹: {input_path}")

        # 收集图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []

        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))

        if not image_paths:
            print("⚠️ 文件夹中没有找到图像文件")
            return

        print(f"📊 找到 {len(image_paths)} 张图像")

        try:
            # 执行批量预测
            results = detector.predict_batch(image_paths, enhance_type=args.enhance)

            # 生成报告
            output_path = args.output or f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report = detector.generate_report(results, output_path)

            # 显示统计信息
            print("\n📈 批量预测统计:")
            print("=" * 50)
            print(f"总图像数: {report['total_images']}")
            print(f"平均置信度: {report['average_confidence']:.4f} ({report['average_confidence'] * 100:.2f}%)")
            print(f"平均推理时间: {report['average_inference_time']:.3f}秒")

            print(f"\n类别分布:")
            for category, count in report['category_distribution'].items():
                percentage = count / report['total_images'] * 100
                print(f"  {category}: {count}张 ({percentage:.1f}%)")

        except Exception as e:
            print(f"❌ 批量预测失败: {e}")


def main():
    """主函数"""
    parser = create_cli_parser()
    args = parser.parse_args()

    print("🚀 缺陷检测用户端应用程序（修改版 - 2分类：划痕/斑点）")
    print("=" * 60)

    # 设备设置
    device = None
    if args.device:
        device = torch.device(args.device)

    try:
        # 初始化检测器
        detector = DefectDetector(args.model, device=device)

        # GUI模式
        if args.gui:
            if not GUI_AVAILABLE:
                print("❌ GUI模块不可用，请安装tkinter")
                return

            print("🖥️ 启动GUI界面...")
            app = DefectDetectionGUI(detector)
            app.run()

        # 命令行模式
        else:
            run_cli_prediction(detector, args)

    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()


# ================================
# 便捷函数（修改版）
# ================================

def quick_predict(model_path, image_path, enhance_type=None):
    """快速预测函数（2分类版本）"""
    detector = DefectDetector(model_path)
    result = detector.predict_single(image_path, enhance_type=enhance_type)

    print(f"预测结果: {result['predicted_name']} (置信度: {result['confidence']:.2%})")

    # 显示原始3分类概率
    if 'probabilities_3class_original' in result:
        print("原始3分类概率:", end=" ")
        original_categories = ['油污', '划痕', '斑点']
        for category, prob in zip(original_categories, result['probabilities_3class_original']):
            print(f"{category}:{prob:.2%}", end=" ")
        print("\n(油污概率已合并)")

    return result


def create_demo_script(model_path, test_images_dir):
    """创建演示脚本（修改版）"""
    demo_code = f'''#!/usr/bin/env python3
"""
缺陷检测演示脚本（修改版 - 2分类：划痕/斑点）
"""

from defect_detection_app_modified import DefectDetector, quick_predict
import os

# 模型路径
MODEL_PATH = "{model_path}"

# 测试图像目录
TEST_IMAGES_DIR = "{test_images_dir}"

def demo_single_prediction():
    """演示单张图像预测"""
    print("🔍 单张图像预测演示（2分类版本）")
    print("-" * 40)

    # 获取测试图像
    test_images = []
    for file in os.listdir(TEST_IMAGES_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(TEST_IMAGES_DIR, file))

    if test_images:
        # 预测第一张图像
        result = quick_predict(MODEL_PATH, test_images[0])
        print(f"图像: {{os.path.basename(test_images[0])}}")
        print(f"预测: {{result['predicted_name']}}")
        print(f"置信度: {{result['confidence']:.2%}}")

def demo_batch_prediction():
    """演示批量预测"""
    print("\\n📁 批量预测演示（2分类版本）")
    print("-" * 40)

    detector = DefectDetector(MODEL_PATH)

    # 收集所有测试图像
    image_paths = []
    for file in os.listdir(TEST_IMAGES_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(TEST_IMAGES_DIR, file))

    if image_paths:
        results = detector.predict_batch(image_paths[:5])  # 只处理前5张
        report = detector.generate_report(results, "demo_results_2class.json")

        print(f"处理了 {{len(results)}} 张图像")
        print("类别分布（2分类）:")
        for category, count in report['category_distribution'].items():
            print(f"  {{category}}: {{count}}张")

if __name__ == "__main__":
    demo_single_prediction()
    demo_batch_prediction()
'''

    with open("demo_defect_detection_2class.py", "w", encoding="utf-8") as f:
        f.write(demo_code)

    print("📝 演示脚本已生成: demo_defect_detection_2class.py")


# ================================
# 模型转换工具
# ================================

def convert_3class_to_2class_model(model_3class_path, model_2class_path):
    """
    将3分类模型转换为2分类模型
    这是一个独立的转换工具函数
    """
    try:
        print("🔄 开始模型转换...")

        # 加载3分类模型
        model_3class = EnhancedResNetClassifier(num_classes=3, pretrained=False, use_attention=True)
        checkpoint = torch.load(model_3class_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model_3class.load_state_dict(checkpoint['model_state_dict'])
            original_accuracy = checkpoint.get('best_accuracy', 'N/A')
        else:
            model_3class.load_state_dict(checkpoint)
            original_accuracy = 'N/A'

        print(f"✅ 原始3分类模型加载完成，最佳准确率: {original_accuracy}")

        # 创建2分类模型
        model_2class = EnhancedResNetClassifier(num_classes=2, pretrained=False, use_attention=True)

        # 复制权重
        model_2class_dict = model_2class.state_dict()
        model_3class_dict = model_3class.state_dict()

        for key in model_2class_dict.keys():
            if 'classifier.5' not in key:  # 跳过最后的分类层
                model_2class_dict[key] = model_3class_dict[key]

        # 处理分类层权重
        if 'classifier.5.weight' in model_3class_dict:
            old_weight = model_3class_dict['classifier.5.weight']  # [3, 512]
            old_bias = model_3class_dict['classifier.5.bias']  # [3]

            # 创建新的2分类权重
            new_weight = torch.zeros(2, old_weight.size(1))
            new_bias = torch.zeros(2)

            # scratch (原索引1 -> 新索引0)，添加30%的oil权重
            new_weight[0] = old_weight[1] + 0.3 * old_weight[0]
            new_bias[0] = old_bias[1] + 0.3 * old_bias[0]

            # stain (原索引2 -> 新索引1)，添加70%的oil权重
            new_weight[1] = old_weight[2] + 0.7 * old_weight[0]
            new_bias[1] = old_bias[2] + 0.7 * old_bias[0]

            model_2class_dict['classifier.5.weight'] = new_weight
            model_2class_dict['classifier.5.bias'] = new_bias

        model_2class.load_state_dict(model_2class_dict)

        # 保存2分类模型
        save_checkpoint = {
            'model_state_dict': model_2class.state_dict(),
            'num_classes': 2,
            'categories': ['scratch', 'stain'],
            'category_names': ['划痕', '斑点'],
            'converted_from': model_3class_path,
            'original_3class_accuracy': original_accuracy,
            'conversion_time': datetime.now().isoformat()
        }

        torch.save(save_checkpoint, model_2class_path)
        print(f"✅ 2分类模型已保存至: {model_2class_path}")

        return model_2class_path

    except Exception as e:
        print(f"❌ 模型转换失败: {e}")
        raise


# ================================
# 使用示例
# ================================

if __name__ == "__main__":
    # 如果直接运行此文件，显示使用帮助
    if len(sys.argv) == 1:
        print("🚀 缺陷检测用户端应用程序（修改版 - 2分类：划痕/斑点）")
        print("=" * 60)
        print()
        print("使用方法:")
        print("1. GUI模式:")
        print("   python defect_detection_app_modified.py --model best_enhanced_model.pth --gui")
        print()
        print("2. 单张图像预测:")
        print("   python defect_detection_app_modified.py --model model.pth --input image.jpg")
        print()
        print("3. 批量预测:")
        print(
            "   python defect_detection_app_modified.py --model model.pth --input images_folder/ --output results.json")
        print()

        print()
        print("参数说明:")
        print("  --model, -m     : 模型文件路径 (必需)")
        print("  --input, -i     : 输入图像或文件夹路径")
        print("  --output, -o    : 输出结果文件路径")
        print("  --gui, -g       : 启动GUI界面")
        print("  --attention, -a : 生成注意力图")
        print("  --visualize, -v : 可视化预测结果")
        print("  --device, -d    : 指定设备 (cpu/cuda)")

        print()
        print("🔧 工具函数:")
        print("  - quick_predict(): 快速预测单张图像")
        print("  - convert_3class_to_2class_model(): 模型格式转换")
        print("  - create_demo_script(): 生成演示脚本")
        print()
        print("🎯 GUI界面特点:")
        print("  - 简洁的2分类结果显示（仅显示划痕和斑点概率）")

        print("  - 保留注意力图显示功能")
        print("  - 支持缺陷增强处理")
    else:
        main()