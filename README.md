**1.整体项目简介：本次课设项目包含三个部分：基础功能实现（basicGUI），图片滤镜实现（1.zip)，缺陷检测系统(defect_decetion)。**


---



**2.基础功能实现**

2.1环境部署：

方案一：使用requirements.txt

```
# 创建虚拟环境（推荐）
python -m venv image_processor_env

# 激活虚拟环境
# Windows:
image_processor_env\Scripts\activate
# macOS/Linux:
source image_processor_env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行程序
python image_processor_gui.py
```

方案二：逐步安装

```
# 图像处理库
pip install opencv-python
pip install numpy
pip install Pillow

# 验证安装
python -c "import cv2, numpy, PIL; print('安装成功')"

# 检查tkinter是否可用
python -c "import tkinter; print('tkinter可用')"

# 如果tkinter不可用：
# Ubuntu/Debian:
sudo apt-get install python3-tkinter

# CentOS/RHEL:
sudo yum install tkinter

# macOS (通常已包含):
# 重新安装Python或使用Homebrew

# Windows:
# 重新安装Python，确保勾选"Add Python to PATH"和"Install tkinter"
```

**3.图像滤镜**

```
python>=3.7.0
numpy>=1.21.6
opencv-python>=4.11.0
tkinter>=8.6
PIL>=5.2.0
```

**4.缺陷检测系统**

4.1.环境配置：

```
# 深度学习框架
torch>=1.12.0
torchvision>=0.13.0
torchaudio>=0.12.0

# 图像处理核心库
opencv-python>=4.5.0
Pillow>=8.3.0
numpy>=1.21.0

# 数据增强和预处理
albumentations>=1.3.0

# 机器学习和数据分析
scikit-learn>=1.0.0
pandas>=1.3.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0

# 进度条和工具
tqdm>=4.62.0

# GUI相关（Python标准库，通常已包含）
# tkinter

# 系统工具
pathlib>=1.0.1

#如GPU版本
# 替换上面的torch相关行为：
torch>=1.12.0+cu117
torchvision>=0.13.0+cu117
torchaudio>=0.12.0+cu117
--extra-index-url https://download.pytorch.org/whl/cu117
```

均为深度学习，模型训练的常用库。

4.2运行方式

    保证代码和模型在同一文件夹下。

4.2.1GUI模式

```
python defect_detection_app_modified.py --model best_enhanced_model.pth --gui
```

4.2.2单张图片预测

```
python defect_detection_app_modified.py --model model.pth --input image.jpg
```

4.2.3批量预测

```
python defect_detection_app_modified.py --model model.pth --input images_folder/ --output results.json

```
