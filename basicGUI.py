#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像处理工具箱GUI界面（大字体版本）
基于Tkinter开发的本地图形用户界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
from datetime import datetime


class ImageProcessorGUI:
    """图像处理GUI主类"""

    def __init__(self, root):
        self.root = root
        self.root.title("图像处理工具箱 v2.0")
        self.root.geometry("1400x900")  # 增大窗口尺寸以适应大字体
        self.root.configure(bg='#f0f0f0')

        # 设置窗口图标和样式
        self.setup_styles()

        # 初始化变量
        self.current_image_path = None
        self.original_image = None
        self.processed_image = None
        self.processing_thread = None

        # 创建GUI界面
        self.create_widgets()

        # 绑定事件
        self.bind_events()

    def setup_styles(self):
        """设置界面样式（大字体版本）"""
        style = ttk.Style()
        style.theme_use('clam')

        # 配置大字体样式
        style.configure('Title.TLabel', font=('Arial', 20, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 18, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 16), foreground='#7f8c8d')
        style.configure('Process.TButton', font=('Arial', 16, 'bold'), padding=(15, 8))

        # 配置其他控件的字体
        style.configure('TButton', font=('Arial', 12), padding=(10, 6))
        style.configure('TLabel', font=('Arial', 12))
        style.configure('TLabelframe.Label', font=('Arial', 13, 'bold'))
        style.configure('TRadiobutton', font=('Arial', 20))
        style.configure('TCheckbutton', font=('Arial', 11))
        style.configure('TSpinbox', font=('Arial', 11))
        style.configure('TCombobox', font=('Arial', 11))

        # 设置主题颜色
        style.configure('TNotebook', background='#ecf0f1')
        style.configure('TNotebook.Tab', padding=[25, 12], font=('Arial', 12, 'bold'))

        # 配置进度条样式
        style.configure('TProgressbar', thickness=25)

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="图像处理工具箱", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 25))

        # 左侧控制面板
        self.create_control_panel(main_frame)

        # 右侧图像显示区域
        self.create_image_display(main_frame)

        # 底部状态栏和日志
        self.create_status_panel(main_frame)

    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.Frame(parent, padding="15")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        control_frame.columnconfigure(0, weight=1)

        # 文件选择区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="15")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        file_frame.columnconfigure(1, weight=1)

        select_btn = ttk.Button(file_frame, text="选择图像", command=self.select_image)
        select_btn.grid(row=0, column=0, padx=(0, 15))

        self.file_path_var = tk.StringVar(value="未选择文件")
        file_label = ttk.Label(file_frame, textvariable=self.file_path_var, style='Info.TLabel')
        file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))

        # 处理方法选择
        method_frame = ttk.LabelFrame(control_frame, text="处理方法", padding="15")
        method_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        method_frame.columnconfigure(0, weight=1)

        # 创建选项卡
        self.notebook = ttk.Notebook(method_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 滤波器选项卡
        self.create_filter_tab()

        # 形态学操作选项卡
        self.create_morphology_tab()

        # 增强处理选项卡
        self.create_enhancement_tab()

        # 批量处理选项卡
        self.create_batch_tab()

        # 处理按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.process_button = ttk.Button(button_frame, text="开始处理",
                                         command=self.start_processing, style='Process.TButton')
        self.process_button.grid(row=0, column=0, padx=(0, 8), sticky=(tk.W, tk.E))

        self.save_button = ttk.Button(button_frame, text="保存结果",
                                      command=self.save_result, state='disabled')
        self.save_button.grid(row=0, column=1, padx=(8, 0), sticky=(tk.W, tk.E))

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                            mode='determinate')
        self.progress_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(15, 0))

    def create_filter_tab(self):
        """创建滤波器选项卡"""
        filter_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(filter_frame, text="滤波器")

        # 滤波器类型选择
        self.filter_type = tk.StringVar(value="lowpass")

        ttk.Radiobutton(filter_frame, text="理想低通滤波", variable=self.filter_type,
                        value="lowpass").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(filter_frame, text="理想高通滤波", variable=self.filter_type,
                        value="highpass").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(filter_frame, text="中值滤波", variable=self.filter_type,
                        value="median").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(filter_frame, text="带通滤波", variable=self.filter_type,
                        value="bandpass").grid(row=3, column=0, sticky=tk.W, pady=4)

        # 参数设置
        params_frame = ttk.LabelFrame(filter_frame, text="参数设置", padding="10")
        params_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        params_frame.columnconfigure(1, weight=1)

        # 截止频率
        ttk.Label(params_frame, text="截止频率:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.cutoff_var = tk.IntVar(value=20)
        cutoff_spin = ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.cutoff_var, width=12)
        cutoff_spin.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=4, padx=(15, 0))

        # 带通滤波参数
        ttk.Label(params_frame, text="最小值:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.min_value_var = tk.IntVar(value=100)
        min_spin = ttk.Spinbox(params_frame, from_=0, to=255, textvariable=self.min_value_var, width=12)
        min_spin.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=4, padx=(15, 0))

        ttk.Label(params_frame, text="最大值:").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.max_value_var = tk.IntVar(value=200)
        max_spin = ttk.Spinbox(params_frame, from_=0, to=255, textvariable=self.max_value_var, width=12)
        max_spin.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=4, padx=(15, 0))

    def create_morphology_tab(self):
        """创建形态学操作选项卡"""
        morph_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(morph_frame, text="形态学")

        # 形态学操作类型选择
        self.morph_type = tk.StringVar(value="opening")

        ttk.Radiobutton(morph_frame, text="开运算", variable=self.morph_type,
                        value="opening").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(morph_frame, text="闭运算", variable=self.morph_type,
                        value="closing").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(morph_frame, text="膨胀", variable=self.morph_type,
                        value="dilation").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(morph_frame, text="腐蚀", variable=self.morph_type,
                        value="erosion").grid(row=3, column=0, sticky=tk.W, pady=4)

        # 结构元素参数
        kernel_frame = ttk.LabelFrame(morph_frame, text="结构元素", padding="10")
        kernel_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        kernel_frame.columnconfigure(1, weight=1)
        kernel_frame.columnconfigure(3, weight=1)

        ttk.Label(kernel_frame, text="宽度:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.kernel_width_var = tk.IntVar(value=5)
        width_spin = ttk.Spinbox(kernel_frame, from_=3, to=21, textvariable=self.kernel_width_var, width=10)
        width_spin.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=4, padx=(8, 15))

        ttk.Label(kernel_frame, text="高度:").grid(row=0, column=2, sticky=tk.W, pady=4)
        self.kernel_height_var = tk.IntVar(value=5)
        height_spin = ttk.Spinbox(kernel_frame, from_=3, to=21, textvariable=self.kernel_height_var, width=10)
        height_spin.grid(row=0, column=3, sticky=(tk.W, tk.E), pady=4, padx=(8, 0))

    def create_enhancement_tab(self):
        """创建增强处理选项卡"""
        enhance_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(enhance_frame, text="增强处理")

        # 增强处理类型选择
        self.enhance_type = tk.StringVar(value="hist")

        ttk.Radiobutton(enhance_frame, text="直方图均衡化", variable=self.enhance_type,
                        value="hist").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(enhance_frame, text="边缘检测与增强", variable=self.enhance_type,
                        value="edge").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(enhance_frame, text="对数变换", variable=self.enhance_type,
                        value="log").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Radiobutton(enhance_frame, text="线条检测", variable=self.enhance_type,
                        value="line").grid(row=3, column=0, sticky=tk.W, pady=4)

        # 增强参数（可扩展）
        enhance_params_frame = ttk.LabelFrame(enhance_frame, text="增强参数", padding="10")
        enhance_params_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(15, 0))

        self.enhance_strength_var = tk.DoubleVar(value=1.0)
        ttk.Label(enhance_params_frame, text="增强强度:").grid(row=0, column=0, sticky=tk.W, pady=4)
        strength_scale = ttk.Scale(enhance_params_frame, from_=0.1, to=3.0,
                                   variable=self.enhance_strength_var, orient=tk.HORIZONTAL,
                                   length=200)
        strength_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=4, padx=(15, 0))
        enhance_params_frame.columnconfigure(1, weight=1)

    def create_batch_tab(self):
        """创建批量处理选项卡"""
        batch_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(batch_frame, text="批量处理")

        # 批量处理选项
        self.batch_all = tk.BooleanVar(value=False)
        ttk.Checkbutton(batch_frame, text="执行所有处理方法",
                        variable=self.batch_all).grid(row=0, column=0, sticky=tk.W, pady=8)

        # 选择特定方法
        methods_frame = ttk.LabelFrame(batch_frame, text="选择处理方法", padding="10")
        methods_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(15, 0))

        # 方法复选框
        self.method_vars = {}
        methods = [
            ("理想低通滤波", "lowpass"),
            ("理想高通滤波", "highpass"),
            ("中值滤波", "median"),
            ("直方图均衡化", "hist"),
            ("边缘检测", "edge"),
            ("对数变换", "log"),
            ("开运算", "opening"),
            ("闭运算", "closing")
        ]

        for i, (name, key) in enumerate(methods):
            var = tk.BooleanVar()
            self.method_vars[key] = var
            ttk.Checkbutton(methods_frame, text=name, variable=var).grid(
                row=i // 2, column=i % 2, sticky=tk.W, pady=4, padx=(0, 25))

        # 输出目录选择
        output_frame = ttk.LabelFrame(batch_frame, text="输出设置", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        output_frame.columnconfigure(1, weight=1)

        ttk.Button(output_frame, text="选择输出目录",
                   command=self.select_output_dir).grid(row=0, column=0, pady=4)

        self.output_dir_var = tk.StringVar(value="./results")
        ttk.Label(output_frame, textvariable=self.output_dir_var,
                  style='Info.TLabel').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(15, 0))

    def create_image_display(self, parent):
        """创建图像显示区域"""
        image_frame = ttk.Frame(parent, padding="15")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)

        # 图像显示标题
        display_title = ttk.Label(image_frame, text="图像预览", style='Subtitle.TLabel')
        display_title.grid(row=0, column=0, pady=(0, 15))

        # 图像显示区域（使用Canvas和Scrollbar）
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(canvas_frame, bg='white', relief='sunken', bd=2)
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 滚动条
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.image_canvas.configure(xscrollcommand=h_scrollbar.set)

        # 图像信息显示
        info_frame = ttk.Frame(image_frame)
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        info_frame.columnconfigure(1, weight=1)

        self.image_info_var = tk.StringVar(value="未加载图像")
        ttk.Label(info_frame, text="图像信息:", style='Info.TLabel').grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.image_info_var,
                  style='Info.TLabel').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(15, 0))

    def create_status_panel(self, parent):
        """创建状态栏和日志面板"""
        status_frame = ttk.Frame(parent, padding="15")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 0))
        status_frame.columnconfigure(0, weight=1)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Info.TLabel')
        status_label.grid(row=0, column=0, sticky=tk.W)

        # 日志区域
        log_frame = ttk.LabelFrame(status_frame, text="处理日志", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        log_frame.columnconfigure(0, weight=1)

        # 设置日志文本框的字体
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, width=80, font=('Arial', 11))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 清除日志按钮
        clear_btn = ttk.Button(log_frame, text="清除日志", command=self.clear_log)
        clear_btn.grid(row=1, column=0, pady=(8, 0))

    def bind_events(self):
        """绑定事件处理"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 绑定画布事件
        self.image_canvas.bind("<Button-1>", self.on_canvas_click)
        self.image_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.image_canvas.bind("<MouseWheel>", self.on_canvas_scroll)

    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)

    def update_status(self, status):
        """更新状态栏"""
        self.status_var.set(status)
        self.root.update_idletasks()

    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("PNG文件", "*.png"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self.load_and_display_image(file_path)
            self.log_message(f"已加载图像: {os.path.basename(file_path)}")

    def select_output_dir(self):
        """选择输出目录"""
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir_var.set(dir_path)
            self.log_message(f"输出目录已设置: {dir_path}")

    def load_and_display_image(self, image_path):
        """加载并显示图像"""
        try:
            # 使用OpenCV加载图像
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError("无法加载图像文件")

            # 显示图像
            self.display_image(self.original_image)

            # 更新图像信息
            height, width = self.original_image.shape[:2]
            channels = self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1
            file_size = os.path.getsize(image_path) / 1024  # KB

            info_text = f"尺寸: {width}x{height}, 通道: {channels}, 大小: {file_size:.1f}KB"
            self.image_info_var.set(info_text)

            # 启用处理按钮
            self.process_button.configure(state='normal')

        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
            self.log_message(f"加载图像失败: {str(e)}")

    def display_image(self, cv_image, title="图像预览"):
        """在Canvas上显示OpenCV图像"""
        try:
            # 转换颜色空间（OpenCV使用BGR，PIL使用RGB）
            if len(cv_image.shape) == 3:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_image

            # 转换为PIL图像
            pil_image = Image.fromarray(rgb_image)

            # 计算显示尺寸（保持比例）
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas还没有初始化尺寸，使用默认值
                canvas_width, canvas_height = 700, 500

            # 计算缩放比例
            img_width, img_height = pil_image.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # 不放大，只缩小

            if scale < 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 转换为Tkinter可显示的格式
            self.display_photo = ImageTk.PhotoImage(pil_image)

            # 清除画布并显示图像
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                self.image_canvas.winfo_width() // 2,
                self.image_canvas.winfo_height() // 2,
                anchor=tk.CENTER,
                image=self.display_photo
            )

            # 更新滚动区域
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

        except Exception as e:
            self.log_message(f"显示图像失败: {str(e)}")

    def start_processing(self):
        """开始图像处理"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择图像文件")
            return

        # 禁用处理按钮
        self.process_button.configure(state='disabled')
        self.update_status("处理中...")
        self.progress_var.set(0)

        # 在新线程中执行处理
        self.processing_thread = threading.Thread(target=self.process_image)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def process_image(self):
        """图像处理主函数（在后台线程中执行）"""
        try:
            # 获取当前选中的选项卡
            current_tab = self.notebook.select()
            tab_text = self.notebook.tab(current_tab, "text")

            if tab_text == "批量处理":
                self.process_batch()
            else:
                self.process_single()

        except Exception as e:
            self.root.after(0, lambda: self.on_processing_error(str(e)))
        finally:
            self.root.after(0, self.on_processing_complete)

    def process_single(self):
        """处理单个方法"""
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")

        processor = ImageProcessor(self.current_image_path)

        if tab_text == "滤波器":
            method = self.filter_type.get()
            if method == "lowpass":
                result = processor.ideal_lowpass_filter(self.cutoff_var.get())
                self.processed_image = result
            elif method == "highpass":
                result = processor.ideal_highpass_filter(self.cutoff_var.get())
                self.processed_image = result
            elif method == "median":
                result = processor.median_filter()
                self.processed_image = result
            elif method == "bandpass":
                result, _ = processor.band_pass_filter(
                    self.min_value_var.get(), self.max_value_var.get())
                self.processed_image = result

        elif tab_text == "形态学":
            method = self.morph_type.get()
            kernel_size = (self.kernel_width_var.get(), self.kernel_height_var.get())

            if method == "opening":
                result, _ = processor.morphological_opening(kernel_size)
                self.processed_image = result
            elif method == "closing":
                result, _ = processor.morphological_closing(kernel_size)
                self.processed_image = result
            elif method == "dilation":
                result, _ = processor.morphological_dilation(kernel_size)
                self.processed_image = result
            elif method == "erosion":
                result, _ = processor.morphological_erosion(kernel_size)
                self.processed_image = result

        elif tab_text == "增强处理":
            method = self.enhance_type.get()

            if method == "hist":
                result = processor.histogram_equalization()
                self.processed_image = result
            elif method == "edge":
                result = processor.edge_detection_and_enhancement()
                self.processed_image = result
            elif method == "log":
                result = processor.log_transform()
                self.processed_image = result
            elif method == "line":
                result, _ = processor.line_detection()
                self.processed_image = result

        # 更新进度
        self.root.after(0, lambda: self.progress_var.set(100))

        # 显示处理结果
        if self.processed_image is not None:
            self.root.after(0, lambda: self.display_image(self.processed_image, "处理结果"))
            self.root.after(0, lambda: self.log_message(f"处理完成: {tab_text} - {method}"))

    def process_batch(self):
        """批量处理"""
        processor = ImageProcessor(self.current_image_path)
        output_dir = self.output_dir_var.get()

        if self.batch_all.get():
            # 执行所有处理方法
            self.root.after(0, lambda: self.log_message("开始批量处理所有方法..."))
            results = processor.process_all(output_dir)
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.log_message(f"批量处理完成，结果保存至: {output_dir}"))
        else:
            # 执行选中的方法
            selected_methods = [key for key, var in self.method_vars.items() if var.get()]
            if not selected_methods:
                raise ValueError("请至少选择一个处理方法")

            total_methods = len(selected_methods)
            for i, method in enumerate(selected_methods):
                self.root.after(0, lambda m=method: self.log_message(f"正在处理: {m}"))

                # 执行对应的处理方法
                output_path = os.path.join(output_dir, f"{method}_result.jpg")

                if method == "lowpass":
                    processor.ideal_lowpass_filter(output_path=output_path)
                elif method == "highpass":
                    processor.ideal_highpass_filter(output_path=output_path)
                elif method == "median":
                    processor.median_filter(output_path=output_path)
                elif method == "hist":
                    processor.histogram_equalization(output_path=output_path)
                elif method == "edge":
                    processor.edge_detection_and_enhancement(output_path=output_path)
                elif method == "log":
                    processor.log_transform(output_path=output_path)
                elif method == "opening":
                    processor.morphological_opening(output_path=output_path)
                elif method == "closing":
                    processor.morphological_closing(output_path=output_path)

                # 更新进度
                progress = ((i + 1) / total_methods) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))

            self.root.after(0, lambda: self.log_message(f"批量处理完成，共处理 {total_methods} 个方法"))

    def on_processing_complete(self):
        """处理完成后的回调"""
        self.process_button.configure(state='normal')
        self.save_button.configure(state='normal')
        self.update_status("处理完成")
        self.log_message("图像处理完成")

    def on_processing_error(self, error_msg):
        """处理出错时的回调"""
        messagebox.showerror("处理错误", f"图像处理失败: {error_msg}")
        self.log_message(f"处理失败: {error_msg}")
        self.process_button.configure(state='normal')
        self.update_status("处理失败")
        self.progress_var.set(0)

    def save_result(self):
        """保存处理结果"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "没有可保存的处理结果")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存处理结果",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG文件", "*.jpg"),
                ("PNG文件", "*.png"),
                ("BMP文件", "*.bmp"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                self.log_message(f"结果已保存至: {os.path.basename(file_path)}")
                messagebox.showinfo("成功", "图像保存成功!")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                self.log_message(f"保存失败: {str(e)}")

    def on_canvas_click(self, event):
        """画布点击事件"""
        self.image_canvas.scan_mark(event.x, event.y)

    def on_canvas_drag(self, event):
        """画布拖拽事件"""
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def on_canvas_scroll(self, event):
        """画布滚轮事件"""
        self.image_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_closing(self):
        """窗口关闭事件"""
        if self.processing_thread and self.processing_thread.is_alive():
            if messagebox.askokcancel("退出", "正在处理图像，确定要退出吗？"):
                self.root.destroy()
        else:
            self.root.destroy()


# 导入之前创建的ImageProcessor类
class ImageProcessor:
    """图像处理类，包含各种图像处理方法"""

    def __init__(self, input_path):
        """初始化图像处理器

        Args:
            input_path (str): 输入图像路径
        """
        self.input_path = input_path
        self.image = None
        self.load_image()

    def load_image(self):
        """加载图像"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"图像文件未找到: {self.input_path}")
        self.image = cv2.imread(self.input_path)
        if self.image is None:
            raise ValueError(f"无法读取图像文件: {self.input_path}")

    def ideal_lowpass_filter(self, cutoff_freq=20, output_path='ideal_lowpass_result.jpg'):
        """理想低通滤波器"""
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(self.image)
        f_shift = np.fft.fftshift(f)
        rows, cols = self.image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)

        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if distance >= cutoff_freq:
                    mask[i, j] = 0

        f_shift_filtered = f_shift * mask

        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)

        ideal_lowpass = cv2.normalize(
            img_filtered, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        if output_path:
            cv2.imwrite(output_path, ideal_lowpass)
        return ideal_lowpass

    def ideal_highpass_filter(self, cutoff_freq=40, output_path='ideal_highpass_result.jpg'):
        """理想高通滤波器"""
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(self.image)
        f_shift = np.fft.fftshift(f)
        rows, cols = self.image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)

        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if distance <= cutoff_freq:
                    mask[i, j] = 0

        f_shift_filtered = f_shift * mask

        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)

        ideal_highpass = cv2.normalize(
            img_filtered, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        if output_path:
            cv2.imwrite(output_path, ideal_highpass)
        return ideal_highpass

    def line_detection(self, output_path='line_detection_result.jpg',
                       output_path_p='line_detection_p_result.jpg'):
        """线条变化检测（Hough变换）"""
        img = self.image.copy()
        img = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # 标准Hough变换
        lines = cv2.HoughLines(edges, 1, np.pi / 2, 118)
        result = img.copy()

        if lines is not None:
            for i_line in lines:
                for line in i_line:
                    rho = line[0]
                    theta = line[1]
                    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
                        pt1 = (int(rho / np.cos(theta)), 0)
                        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                        cv2.line(result, pt1, pt2, (0, 0, 255))
                    else:
                        pt1 = (0, int(rho / np.sin(theta)))
                        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                        cv2.line(result, pt1, pt2, (0, 0, 255), 1)

        if output_path:
            cv2.imwrite(output_path, result)
        return result, None

    def histogram_equalization(self, output_path='histogram_eq_result.jpg'):
        """直方图均衡化"""
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (256, 256))
        equ = cv2.equalizeHist(img_resized)

        if output_path:
            cv2.imwrite(output_path, equ)
        return equ

    def median_filter(self, output_path='median_filter_result.jpg'):
        """中值滤波"""
        from PIL import Image as PILImage

        image_pil = PILImage.open(self.input_path).convert('L')
        width, height = image_pil.width, image_pil.height

        output_width = width - 4
        output_height = height - 4

        input_array = np.array(image_pil)
        output_array = np.zeros((output_height, output_width), dtype=np.uint8)

        for i in range(2, height - 2):
            for j in range(2, width - 2):
                window = input_array[i - 2:i + 3, j - 2:j + 3].flatten()
                window.sort()
                output_array[i - 2, j - 2] = window[12]

        if output_path:
            result_image = PILImage.fromarray(output_array)
            result_image.save(output_path)
        return output_array

    def edge_detection_and_enhancement(self, output_path='edge_enhancement_result.jpg'):
        """边缘检测与图像增强"""
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype('float')

        gradient = np.zeros_like(img_gray)

        for x in range(img_gray.shape[0] - 1):
            for y in range(img_gray.shape[1] - 1):
                gx = abs(img_gray[x + 1, y] - img_gray[x, y])
                gy = abs(img_gray[x, y + 1] - img_gray[x, y])
                gradient[x, y] = gx + gy

        sharp = img_gray + gradient
        sharp = np.where(sharp > 255, 255, sharp)
        sharp = np.where(sharp < 0, 0, sharp)
        sharp = sharp.astype('uint8')

        if output_path:
            cv2.imwrite(output_path, sharp)
        return sharp

    def log_transform(self, output_path='log_transform_result.jpg'):
        """灰度图像对数变换"""
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        C = 255 / np.log(1 + 255)
        result = C * np.log(1 + img_gray)
        result = np.array(result, np.float64)

        if output_path:
            cv2.imwrite(output_path, result)
        return result

    def morphological_opening(self, kernel_size=(5, 5), output_path='opening_result.jpg'):
        """形态学开运算"""
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        opened_image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        pixel_sum = np.sum(opened_image)

        if output_path:
            cv2.imwrite(output_path, opened_image)
        return opened_image, pixel_sum

    def morphological_closing(self, kernel_size=(10, 10), output_path='closing_result.jpg'):
        """形态学闭运算"""
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        closed_image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        pixel_sum = np.sum(closed_image)

        if output_path:
            cv2.imwrite(output_path, closed_image)
        return closed_image, pixel_sum

    def morphological_dilation(self, kernel_size=(5, 5), output_path='dilation_result.jpg'):
        """形态学膨胀操作"""
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        dilation = cv2.dilate(self.image, kernel)
        pixel_sum = np.sum(dilation)

        if output_path:
            cv2.imwrite(output_path, dilation)
        return dilation, pixel_sum

    def morphological_erosion(self, kernel_size=(5, 5), output_path='erosion_result.jpg'):
        """形态学腐蚀操作"""
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        erosion = cv2.erode(self.image, kernel)
        pixel_sum = np.sum(erosion)

        if output_path:
            cv2.imwrite(output_path, erosion)
        return erosion, pixel_sum

    def band_pass_filter(self, min_value=200, max_value=256, output_path='bandpass_result.jpg'):
        """带通滤波器"""
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        output = np.zeros(image_gray.shape, np.uint8)

        for i in range(image_gray.shape[0]):
            for j in range(image_gray.shape[1]):
                if min_value < image_gray[i][j] < max_value:
                    output[i][j] = image_gray[i][j]
                else:
                    output[i][j] = 0

        pixel_sum = np.sum(output)

        if output_path:
            cv2.imwrite(output_path, output)
        return output, pixel_sum

    def process_all(self, output_dir='results'):
        """执行所有图像处理方法"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = {}

        results['ideal_lowpass'] = self.ideal_lowpass_filter(
            output_path=os.path.join(output_dir, 'ideal_lowpass.jpg'))

        results['ideal_highpass'] = self.ideal_highpass_filter(
            output_path=os.path.join(output_dir, 'ideal_highpass.jpg'))

        results['line_detection'] = self.line_detection(
            output_path=os.path.join(output_dir, 'line_detection.jpg'))

        results['histogram_eq'] = self.histogram_equalization(
            output_path=os.path.join(output_dir, 'histogram_eq.jpg'))

        results['median_filter'] = self.median_filter(
            output_path=os.path.join(output_dir, 'median_filter.jpg'))

        results['edge_enhancement'] = self.edge_detection_and_enhancement(
            output_path=os.path.join(output_dir, 'edge_enhancement.jpg'))

        results['log_transform'] = self.log_transform(
            output_path=os.path.join(output_dir, 'log_transform.jpg'))

        results['opening'] = self.morphological_opening(
            output_path=os.path.join(output_dir, 'opening.jpg'))

        results['closing'] = self.morphological_closing(
            output_path=os.path.join(output_dir, 'closing.jpg'))

        results['dilation'] = self.morphological_dilation(
            output_path=os.path.join(output_dir, 'dilation.jpg'))

        results['erosion'] = self.morphological_erosion(
            output_path=os.path.join(output_dir, 'erosion.jpg'))

        results['bandpass'] = self.band_pass_filter(
            output_path=os.path.join(output_dir, 'bandpass.jpg'))

        return results


def main():
    """主函数"""
    root = tk.Tk()
    app = ImageProcessorGUI(root)

    # 设置窗口居中
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    # 启动GUI
    root.mainloop()


if __name__ == "__main__":
    main()