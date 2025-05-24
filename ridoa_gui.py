import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget, QGroupBox, 
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QTextEdit, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QAction, QMenuBar, QStatusBar, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QFont, QIcon, QPixmap

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

try:
    import ridoa
except ImportError:
    print("警告: 未找到RIDOA模块，将使用模拟数据进行演示")
    
class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib画布类"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class LogWidget(QTextEdit):
    """日志窗口部件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        
    def add_log(self, text, color="black"):
        """添加日志条目"""
        # 根据日志类型设置颜色
        if "错误" in text:
            color = "red"
        elif "警告" in text:
            color = "orange"
        elif "完成" in text or "成功" in text:
            color = "green"
        elif "信源" in text and ":" in text:
            color = "blue"
            
        # 构建HTML格式的日志条目
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f'<span style="color:gray;">[{timestamp}]</span> <span style="color:{color};">{text}</span><br>'
        
        # 添加到日志文本
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(log_entry)
        self.setTextCursor(cursor)
        
        # 滚动到底部
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class SourceTable(QTableWidget):
    """信源参数表格"""
    sourceDataChanged = pyqtSignal()  # 重命名避免冲突
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["信源", "俯仰角 (°)", "方位角 (°)"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setRowCount(2)  # 默认两个信源
        
        # 初始默认值
        self.set_default_values()
        
        # 连接单元格变化信号
        self.cellChanged.connect(self.on_data_changed)
        
    def set_default_values(self):
        """设置默认值"""
        # 暂时阻断信号
        was_blocked = self.blockSignals(True)
        
        # 设置默认值
        default_data = [
            ["1", "30.0", "60.0"],
            ["2", "45.0", "90.0"]
        ]
        
        for row, row_data in enumerate(default_data):
            for col, text in enumerate(row_data):
                item = QTableWidgetItem(text)
                if col == 0:  # 信源编号列不可编辑
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(row, col, item)
        
        # 恢复先前的状态
        self.blockSignals(was_blocked)
        
    def get_source_data(self):
        """获取信源数据"""
        elevations = []
        azimuths = []
        
        for row in range(self.rowCount()):
            try:
                elev = float(self.item(row, 1).text())
                azim = float(self.item(row, 2).text())
                elevations.append(elev)
                azimuths.append(azim)
            except (ValueError, AttributeError):
                # 跳过无效数据
                pass
                
        return elevations, azimuths
        
    def set_source_count(self, count):
        """设置信源数量"""
        was_blocked = self.blockSignals(True)
        
        current_rows = self.rowCount()
        
        if count > current_rows:
            # 添加行
            for i in range(current_rows, count):
                self.insertRow(i)
                self.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.setItem(i, 1, QTableWidgetItem("30.0"))
                self.setItem(i, 2, QTableWidgetItem("120.0"))
        elif count < current_rows:
            # 删除行
            for i in range(current_rows-1, count-1, -1):
                self.removeRow(i)
        
        self.blockSignals(was_blocked)
                
    def on_data_changed(self):
        """数据变化响应"""
        self.sourceDataChanged.emit()  # 使用新的信号名称

class DOAEstimationThread(QThread):
    """DOA估计线程类"""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    
    def __init__(self, signal_data, config, mock=False, snr_db=20.0, source_info=None):
        """初始化DOA估计线程"""
        super().__init__()
        self.signal_data = signal_data
        self.config = config
        self.mock = mock
        self.snr_db = snr_db  # 使用传入的SNR值
        self.source_info = source_info  # 信源信息
        
    def run(self):
        try:
            self.log.emit("开始DOA估计...")
            self.progress.emit(10)
            # self.log.emit(f"生成信号: {len(self.elevations)}个信源, SNR={self.snr_db}dB")
            if not self.mock and hasattr(ridoa, 'DOASystem'):
                # 使用真实的RIDOA系统
                system = ridoa.DOASystem()
                self.log.emit(f"使用{self.config.nElements}个阵元进行估计")
                num_sources = self.config.maxSources
                self.log.emit(f"信源数量设置为: {num_sources}")
                
                self.progress.emit(30)
                result = system.estimate_doa(self.signal_data, num_sources)
                self.progress.emit(90)
            else:
                # 模拟RIDOA系统结果
                import time
                time.sleep(1)  # 模拟计算延迟
                self.progress.emit(30)
                
                # 创建模拟DOA结果
                self.log.emit("使用模拟数据进行DOA估计...")
                result = self.create_mock_result()
                
                self.progress.emit(70)
                time.sleep(0.5)  # 模拟更多延迟
            
            # 更新进度并发送结果
            self.progress.emit(100)
            self.finished.emit(result)
            
            # 记录完成日志
            self.log.emit(f"DOA估计完成! 检测到{len(result.estElevations)}个信源")
            for i, (elev, azim) in enumerate(zip(result.estElevations, result.estAzimuths)):
                self.log.emit(f"信源{i+1}: 俯仰角={elev:.1f}°, 方位角={azim:.1f}°")
                
        except Exception as e:
            self.log.emit(f"错误: {str(e)}")
            self.finished.emit(None)
            
    def create_mock_result(self):
        """创建模拟结果"""
        # 获取信源信息
        elevations, azimuths = self.get_source_info()
        
        # 创建一个模拟结果对象
        class MockResult:
            def __init__(self):
                # 估计结果 (略微添加随机偏差以模拟估计误差)
                self.estElevations = [e + np.random.uniform(-0.5, 0.5) for e in elevations]
                self.estAzimuths = [a + np.random.uniform(-0.5, 0.5) for a in azimuths]
                
                # 时间点
                self.timepoint = 1.0
                
                # 制造模拟的角度和时间点数据
                num_points = 100
                self.angles = []
                self.anglesTimepoints = []
                
                for e, a in zip(elevations, azimuths):
                    times = np.linspace(0, 1, num_points)
                    angles = e * np.cos(2*np.pi*times - np.deg2rad(a))
                    
                    self.angles.extend(angles)
                    self.anglesTimepoints.extend(times)
                
                # 模拟参数空间累加器
                elev_range = np.linspace(0, 90, 91)
                azim_range = np.linspace(0, 360, 361)
                
                self.elevations = elev_range
                self.azimuths = azim_range
                
                self.accumulator = np.zeros((len(elev_range), len(azim_range)))
                
                # 在估计位置附近创建高值区域
                for e, a in zip(self.estElevations, self.estAzimuths):
                    e_idx = int(e)
                    a_idx = int(a)
                    
                    if 0 <= e_idx < len(elev_range) and 0 <= a_idx < len(azim_range):
                        size = 5
                        for i in range(-size, size+1):
                            for j in range(-size, size+1):
                                if (0 <= e_idx+i < len(elev_range) and 
                                    0 <= a_idx+j < len(azim_range)):
                                    
                                    dist = np.sqrt(i**2 + j**2)
                                    val = np.exp(-0.5 * dist)
                                    
                                    self.accumulator[e_idx+i, a_idx+j] = val
        
        return MockResult()
    
    def get_source_info(self):
        """从信号数据或配置中提取信源信息"""
        if self.source_info is not None:
            return self.source_info
        else:
            # 返回默认信源信息
            return [30.0, 45.0], [60.0, 90.0]

class SignalGenerationThread(QThread):
    """信号生成线程类"""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    
    def __init__(self, elevations, azimuths, config, mock=False, snr_db=20.0):
        super().__init__()
        self.elevations = elevations
        self.azimuths = azimuths
        self.config = config
        self.mock = mock
        self.snr_db = snr_db  # 使用传入的SNR值，而不是从config获取
        
    def run(self):
        try:
            self.log.emit("开始生成模拟信号...")
            self.progress.emit(10)
            
            # 记录参数日志 - 使用本地SNR值
            self.log.emit(f"生成信号: {len(self.elevations)}个信源, SNR={self.snr_db}dB")
            for i, (elev, azim) in enumerate(zip(self.elevations, self.azimuths)):
                self.log.emit(f"信源{i+1}设置: 俯仰角={elev:.1f}°, 方位角={azim:.1f}°")
            
            # 保存信源信息到配置
            # self.config.source_info = (self.elevations, self.azimuths)
            
            if not self.mock and hasattr(ridoa, 'DOASystem'):
                # 使用真实的RIDOA系统
                system = ridoa.DOASystem()
                
                # 设置信号参数
                duration = 1.0  # 信号持续时间，秒
                snr_db = self.snr_db  # 使用本地SNR值
                
                self.progress.emit(30)
                # 生成信号 - 注意这里使用本地SNR值
                signal_data = system.generate_simulation_data(
                    self.elevations, self.azimuths, duration, snr_db
                )
                self.progress.emit(80)
                
            else:
                # 模拟信号生成
                import time
                time.sleep(0.5)  # 模拟计算延迟
                
                self.progress.emit(30)
                self.log.emit("使用模拟数据...")
                
                # 创建模拟信号数据
                duration = 1.0
                sampling_rate = self.config.samplingRate
                n_elements = self.config.nElements
                n_samples = int(duration * sampling_rate)
                
                # 模拟生成多通道数据
                signal_data = np.zeros((n_elements, n_samples), dtype=np.complex128)
                
                # 为每个通道生成信号
                time_axis = np.linspace(0, duration, n_samples)
                for ch in range(n_elements):
                    # 基础信号
                    signal = np.zeros(n_samples)
                    
                    # 为每个信源添加正弦信号，并添加通道间延迟
                    for i, (elev, azim) in enumerate(zip(self.elevations, self.azimuths)):
                        freq = 1000 * (i + 1)  # 不同信源使用不同频率
                        phase = 2 * np.pi * ch * np.sin(np.deg2rad(elev)) * 0.1  # 模拟通道延迟
                        signal += np.sin(2 * np.pi * freq * time_axis + phase)
                    
                    # 添加噪声
                    snr_linear = 10 ** (self.snr_db / 20)
                    signal_power = np.mean(signal ** 2)
                    noise_power = signal_power / snr_linear
                    noise = np.random.normal(0, np.sqrt(noise_power), n_samples)
                    
                    # 合并信号和噪声
                    signal_with_noise = signal + noise
                    
                    # 转为复数形式
                    hilbert_signal = signal_with_noise + 1j * np.imag(np.fft.hilbert(signal_with_noise))
                    signal_data[ch, :] = hilbert_signal
                
                self.progress.emit(80)
                time.sleep(0.5)  # 模拟延迟
            
            # 更新进度并发送结果
            self.progress.emit(100)
            self.finished.emit(signal_data)
            
            # 记录完成日志
            samples = signal_data.shape[1]
            self.log.emit(f"信号生成完成! 生成了{samples}个采样点")
                
        except Exception as e:
            self.log.emit(f"错误: {str(e)}")
            self.finished.emit(None)

class RIDOAApp(QMainWindow):
    """RIDOA主窗口类"""
    def __init__(self):
        super().__init__()
        
        # 设置窗口基本属性
        self.setWindowTitle("RIDOA - 旋转干涉仪多目标参数估计系统")
        self.setMinimumSize(1200, 800)
        
        # 初始化成员变量
        self.signal_data = None
        self.doa_result = None
        self.estimation_thread = None
        self.signal_generation_thread = None
        self.source_info = None
        
        # 检查RIDOA库是否可用
        self.mock_mode = not 'ridoa' in sys.modules
        
        # 初始化配置
        if not self.mock_mode:
            self.config = ridoa.DOAConfig.get_instance()
        else:
            # 模拟配置
            class MockConfig:
                def __init__(self):
                    self.nElements = 8
                    self.elementSpacing = 10.0
                    self.omega = 2 * np.pi
                    self.samplingRate = 40000
                    self.carrierFrequency = 3e9
                    self.estimateRate = 100
                    self.thetaPrecision = 0.1
                    self.precision = 1.0
                    self.accumulatorThreshold = 0.5
                    self.spectrumThreshold = 0.5
                    self.maxSources = 2
                    self.forwardSmoothingSize = 0
                    self.useGPU = True
                    self.gpuDeviceId = 0
                    
                def print(self):
                    print("模拟配置模式")
                    
            self.config = MockConfig()
        
        # 添加GUI特定属性 - 主要修改点
        self.snr_db_value = 20.0  # 使用独立变量存储SNR值
        
        # 创建UI
        self.create_ui()
        
        # 设置默认配置
        self.config_to_ui()
        
        # 添加初始日志
        status = "模拟模式" if self.mock_mode else "已连接库"
        self.log_widget.add_log(f"RIDOA系统已启动 ({status})")
        self.log_widget.add_log(f"当前使用{self.config.nElements}个阵元，{self.config.useGPU and 'GPU加速已启用' or 'CPU模式'}")
        
        # 更新状态栏
        self.status_bar.showMessage("就绪")
        
        # 启动系统资源监控
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self.update_resource_info)
        self.resource_timer.start(2000)  # 每2秒更新一次
        
    def create_ui(self):
        """创建用户界面"""
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左右分割器
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # 创建左侧配置面板
        left_panel = self.create_left_panel()
        
        # 创建右侧内容区
        right_panel = self.create_right_panel()
        
        # 添加到分割器
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setSizes([300, 900])  # 设置初始分割比例
        
        # 创建菜单栏
        self.create_menu()
        
        # 创建状态栏
        self.create_status_bar()
        
    def create_left_panel(self):
        """创建左侧配置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 系统配置分组
        system_group = QGroupBox("系统配置")
        system_layout = QFormLayout()
        
        # 添加各种配置控件
        # 1. 阵列参数
        system_layout.addRow(QLabel("<b>阵列参数</b>"))
        
        self.n_elements_spinbox = QSpinBox()
        self.n_elements_spinbox.setRange(3, 64)
        system_layout.addRow("阵元数量:", self.n_elements_spinbox)
        
        self.element_spacing_spinbox = QDoubleSpinBox()
        self.element_spacing_spinbox.setRange(0.5, 100)
        self.element_spacing_spinbox.setSingleStep(0.5)
        self.element_spacing_spinbox.setDecimals(1)
        system_layout.addRow("阵元间距 (λ/2):", self.element_spacing_spinbox)
        
        self.omega_spinbox = QDoubleSpinBox()
        self.omega_spinbox.setRange(0.1, 10)
        self.omega_spinbox.setSingleStep(0.1)
        self.omega_spinbox.setDecimals(2)
        system_layout.addRow("旋转角速度 (rad/s):", self.omega_spinbox)
        
        # 2. 信号参数
        system_layout.addRow(QLabel("<b>信号参数</b>"))
        
        self.sampling_rate_spinbox = QSpinBox()
        self.sampling_rate_spinbox.setRange(1000, 1000000)
        self.sampling_rate_spinbox.setSingleStep(1000)
        self.sampling_rate_spinbox.setSuffix(" Hz")
        system_layout.addRow("采样率:", self.sampling_rate_spinbox)
        
        self.snr_db_spinbox = QDoubleSpinBox()
        self.snr_db_spinbox.setRange(-20, 100)
        self.snr_db_spinbox.setSingleStep(1)
        self.snr_db_spinbox.setDecimals(1)
        self.snr_db_spinbox.setSuffix(" dB")
        system_layout.addRow("信噪比:", self.snr_db_spinbox)
        
        # 3. 算法参数
        system_layout.addRow(QLabel("<b>算法参数</b>"))
        
        self.max_sources_spinbox = QSpinBox()
        self.max_sources_spinbox.setRange(1, 10)
        self.max_sources_spinbox.valueChanged.connect(self.on_max_sources_changed)
        system_layout.addRow("最大信源数:", self.max_sources_spinbox)
        
        self.estimate_rate_spinbox = QSpinBox()
        self.estimate_rate_spinbox.setRange(10, 1000)
        self.estimate_rate_spinbox.setSingleStep(10)
        self.estimate_rate_spinbox.setSuffix(" Hz")
        system_layout.addRow("估计频率:", self.estimate_rate_spinbox)
        
        self.precision_spinbox = QDoubleSpinBox()
        self.precision_spinbox.setRange(0.01, 10)
        self.precision_spinbox.setSingleStep(0.1)
        self.precision_spinbox.setDecimals(2)
        self.precision_spinbox.setSuffix(" °")
        system_layout.addRow("角度精度:", self.precision_spinbox)
        
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.1, 0.99)
        self.threshold_spinbox.setSingleStep(0.05)
        self.threshold_spinbox.setDecimals(2)
        system_layout.addRow("检测阈值:", self.threshold_spinbox)
        
        self.use_gpu_checkbox = QCheckBox("使用GPU加速")
        system_layout.addRow("", self.use_gpu_checkbox)
        
        # 设置表单布局
        system_group.setLayout(system_layout)
        
        # 添加按钮
        buttons_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("应用配置")
        self.apply_button.clicked.connect(self.apply_config)
        
        self.reset_button = QPushButton("重置配置")
        self.reset_button.clicked.connect(self.config_to_ui)
        
        buttons_layout.addWidget(self.apply_button)
        buttons_layout.addWidget(self.reset_button)
        
        # 添加信源表格
        source_group = QGroupBox("信源设置")
        source_layout = QVBoxLayout()
        
        self.source_table = SourceTable()
        source_layout.addWidget(self.source_table)
        
        # 添加操作按钮
        source_buttons_layout = QHBoxLayout()
        
        self.add_source_button = QPushButton("添加信源")
        self.add_source_button.clicked.connect(self.add_source)
        
        self.remove_source_button = QPushButton("删除信源")
        self.remove_source_button.clicked.connect(self.remove_source)
        
        source_buttons_layout.addWidget(self.add_source_button)
        source_buttons_layout.addWidget(self.remove_source_button)
        
        source_layout.addLayout(source_buttons_layout)
        source_group.setLayout(source_layout)
        
        # 添加到面板布局
        layout.addWidget(system_group)
        layout.addLayout(buttons_layout)
        layout.addWidget(source_group)
        layout.addStretch(1)
        
        return panel
        
    def create_right_panel(self):
        """创建右侧内容区"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 创建操作按钮区
        control_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("生成模拟信号")
        self.generate_button.clicked.connect(self.generate_signal)
        
        self.load_button = QPushButton("加载信号文件")
        self.load_button.clicked.connect(self.load_signal)
        
        self.estimate_button = QPushButton("开始DOA估计")
        self.estimate_button.clicked.connect(self.start_estimation)
        self.estimate_button.setEnabled(False)  # 初始禁用
        
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)  # 初始禁用
        
        control_layout.addWidget(self.generate_button)
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.estimate_button)
        control_layout.addWidget(self.save_button)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # 创建可视化区
        self.viz_tabs = QTabWidget()
        
        # 添加各个可视化标签页
        # 1. 信号时域图
        self.signal_canvas = MplCanvas(width=5, height=4, dpi=100)
        self.viz_tabs.addTab(self.wrap_canvas(self.signal_canvas), "信号时域图")
        
        # 2. MUSIC谱图
        self.spectrum_canvas = MplCanvas(width=5, height=4, dpi=100)
        self.viz_tabs.addTab(self.wrap_canvas(self.spectrum_canvas), "MUSIC谱")
        
        # 3. 角度模糊曲线图
        self.ambiguity_canvas = MplCanvas(width=5, height=4, dpi=100)
        self.viz_tabs.addTab(self.wrap_canvas(self.ambiguity_canvas), "角度模糊曲线")
        
        # 4. 参数空间累加器图
        self.accumulator_canvas = MplCanvas(width=5, height=4, dpi=100)
        self.viz_tabs.addTab(self.wrap_canvas(self.accumulator_canvas), "参数空间累加器")
        
        # 5. 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["信源", "俯仰角 (°)", "方位角 (°)"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.viz_tabs.addTab(self.result_table, "估计结果")
        
        # 添加日志区域
        self.log_widget = LogWidget()
        
        # 可视化区域和日志区域的分割器
        viz_log_splitter = QSplitter(Qt.Vertical)
        viz_log_splitter.addWidget(self.viz_tabs)
        viz_log_splitter.addWidget(self.log_widget)
        viz_log_splitter.setSizes([600, 200])  # 设置初始分割比例
        
        # 添加到右侧面板布局
        layout.addLayout(control_layout)
        layout.addWidget(viz_log_splitter)
        
        return panel
        
    def wrap_canvas(self, canvas):
        """包装Matplotlib画布，添加工具栏和布局"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加导航工具栏
        toolbar = NavigationToolbar2QT(canvas, widget)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        return widget
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        new_action = QAction("新建", self)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("打开...", self)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        theme_menu = view_menu.addMenu("主题")
        light_action = QAction("亮色", self)
        dark_action = QAction("暗色", self)
        theme_menu.addAction(light_action)
        theme_menu.addAction(dark_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        
        self.system_info_label = QLabel()
        self.status_bar.addPermanentWidget(self.system_info_label)
        
    def config_to_ui(self):
        """将配置更新到UI控件"""
        self.n_elements_spinbox.setValue(self.config.nElements)
        self.element_spacing_spinbox.setValue(self.config.elementSpacing)
        self.omega_spinbox.setValue(self.config.omega)
        self.sampling_rate_spinbox.setValue(self.config.samplingRate)
        self.max_sources_spinbox.setValue(self.config.maxSources)
        self.estimate_rate_spinbox.setValue(self.config.estimateRate)
        self.precision_spinbox.setValue(self.config.precision)
        self.threshold_spinbox.setValue(self.config.accumulatorThreshold)
        self.use_gpu_checkbox.setChecked(self.config.useGPU)
        
        # 使用本地SNR值 - 主要修改点
        self.snr_db_spinbox.setValue(self.snr_db_value)
        
        # 更新信源表格
        self.source_table.set_source_count(self.config.maxSources)
        
    def ui_to_config(self):
        """将UI控件值更新到配置"""
        self.config.nElements = self.n_elements_spinbox.value()
        self.config.elementSpacing = self.element_spacing_spinbox.value()
        self.config.omega = self.omega_spinbox.value()
        self.config.samplingRate = self.sampling_rate_spinbox.value()
        self.config.maxSources = self.max_sources_spinbox.value()
        self.config.estimateRate = self.estimate_rate_spinbox.value()
        self.config.precision = self.precision_spinbox.value()
        self.config.accumulatorThreshold = self.threshold_spinbox.value()
        self.config.useGPU = self.use_gpu_checkbox.isChecked()
        
        # 更新本地SNR值 - 主要修改点
        self.snr_db_value = self.snr_db_spinbox.value()
        
    def apply_config(self):
        """应用配置"""
        self.ui_to_config()
        self.log_widget.add_log("配置已更新")
        self.log_widget.add_log(f"阵元数量: {self.config.nElements}, "
                             f"{'启用GPU加速' if self.config.useGPU else '使用CPU计算'}")
    
    def on_max_sources_changed(self, value):
        """最大信源数变化时更新信源表格"""
        self.source_table.set_source_count(value)
        
    def add_source(self):
        """添加信源"""
        current_count = self.source_table.rowCount()
        self.source_table.set_source_count(current_count + 1)
        self.max_sources_spinbox.setValue(current_count + 1)
        
    def remove_source(self):
        """删除信源"""
        current_count = self.source_table.rowCount()
        if current_count > 1:
            self.source_table.set_source_count(current_count - 1)
            self.max_sources_spinbox.setValue(current_count - 1)
        
    def generate_signal(self):
        """生成模拟信号"""
        # 获取信源参数
        elevations, azimuths = self.source_table.get_source_data()
        
        if not elevations or len(elevations) == 0:
            QMessageBox.warning(self, "警告", "请设置至少一个有效信源")
            return
            
        self.source_info = (elevations, azimuths)
        # 更新配置
        self.ui_to_config()
        
        # 禁用操作按钮
        self.generate_button.setEnabled(False)
        self.load_button.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动信号生成线程 - 使用本地SNR值
        self.signal_generation_thread = SignalGenerationThread(
            elevations, azimuths, self.config, self.mock_mode, self.snr_db_value
        )
        self.signal_generation_thread.finished.connect(self.on_signal_generated)
        self.signal_generation_thread.progress.connect(self.progress_bar.setValue)
        self.signal_generation_thread.log.connect(self.log_widget.add_log)
        self.signal_generation_thread.start()
        
    def on_signal_generated(self, signal_data):
        """信号生成完成回调"""
        # 恢复按钮状态
        self.generate_button.setEnabled(True)
        self.load_button.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        if signal_data is None:
            self.log_widget.add_log("信号生成失败", "red")
            return
            
        # 保存信号数据
        self.signal_data = signal_data
        
        # 启用估计按钮
        self.estimate_button.setEnabled(True)
        
        # 绘制信号时域图
        self.plot_signal(signal_data)
        
        # 更新状态栏
        self.status_bar.showMessage("信号数据已生成")
        
    def plot_signal(self, signal_data):
        """绘制信号时域图"""
        # 清除当前图像
        self.signal_canvas.axes.clear()
        
        # 选择前5个通道绘制
        num_channels = min(5, signal_data.shape[0])
        num_samples = signal_data.shape[1]
        time_axis = np.arange(num_samples) / self.config.samplingRate
        
        for i in range(num_channels):
            # 提取实部
            signal = np.real(signal_data[i, :])
            self.signal_canvas.axes.plot(time_axis, signal, label=f"通道 {i+1}")
            
        self.signal_canvas.axes.set_xlabel("时间 (秒)")
        self.signal_canvas.axes.set_ylabel("幅度")
        self.signal_canvas.axes.set_title("阵列信号时域波形")
        self.signal_canvas.axes.legend()
        self.signal_canvas.axes.grid(True)
        
        # 刷新画布
        self.signal_canvas.draw()
        
    def load_signal(self):
        """加载信号文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择信号数据文件", "", "NumPy文件 (*.npy);;所有文件 (*)"
        )
        
        if not file_path:
            return
            
        try:
            # 加载NumPy数组
            signal_data = np.load(file_path)
            
            # 检查数据格式
            if len(signal_data.shape) != 2:
                raise ValueError("信号数据必须是二维数组")
                
            # 转换为复数矩阵
            if np.iscomplexobj(signal_data):
                self.signal_data = signal_data
            else:
                self.log_widget.add_log("警告: 加载的数据不是复数类型，将转换为复数")
                self.signal_data = signal_data.astype(np.complex128)
                
            # 记录日志
            self.log_widget.add_log(f"成功加载信号数据，形状: {self.signal_data.shape}")
            
            # 启用估计按钮
            self.estimate_button.setEnabled(True)
            
            # 绘制信号
            self.plot_signal(self.signal_data)
            
        except Exception as e:
            self.log_widget.add_log(f"加载信号数据失败: {str(e)}", "red")
        
    def start_estimation(self):
        """开始DOA估计"""
        if self.signal_data is None:
            QMessageBox.warning(self, "警告", "请先生成或加载信号数据")
            return
            
        # 更新配置
        self.ui_to_config()
        
        # 禁用操作按钮
        self.generate_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.estimate_button.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动DOA估计线程 - 使用本地SNR值
        self.estimation_thread = DOAEstimationThread(
            self.signal_data, self.config, self.mock_mode, self.snr_db_value,
            source_info=self.source_info
        )
        self.estimation_thread.finished.connect(self.on_estimation_finished)
        self.estimation_thread.progress.connect(self.progress_bar.setValue)
        self.estimation_thread.log.connect(self.log_widget.add_log)
        self.estimation_thread.start()
        
    def on_estimation_finished(self, result):
        """DOA估计完成回调"""
        # 恢复按钮状态
        self.generate_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.estimate_button.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        if result is None:
            self.log_widget.add_log("DOA估计失败", "red")
            return
            
        # 保存估计结果
        self.doa_result = result
        
        # 启用保存按钮
        self.save_button.setEnabled(True)
        
        # 绘制结果图
        self.plot_spectrum(result)
        self.plot_ambiguity_curves(result)
        self.plot_accumulator(result)
        self.update_result_table(result)
        
        # 更新状态栏
        self.status_bar.showMessage("DOA估计完成")
        
    def plot_spectrum(self, result):
        """绘制MUSIC谱"""
        # 清除当前图像
        self.spectrum_canvas.axes.clear()
        
        # 还原未来MUSIC实现中的代码：从结果中提取频谱数据或使用示例数据
        # 现在仅使用示例数据进行显示
        angles = np.linspace(-90, 90, 361)
        spectrum = np.zeros_like(angles)
        
        # 在估计的方向处创建峰值
        for elev in result.estElevations:
            idx = np.argmin(np.abs(angles - elev))
            spectrum[idx-5:idx+6] = np.exp(-0.5 * np.linspace(-3, 3, 11)**2)
        
        # 归一化
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
            
        # 绘制
        self.spectrum_canvas.axes.plot(angles, spectrum)
        self.spectrum_canvas.axes.set_xlabel("角度 (°)")
        self.spectrum_canvas.axes.set_ylabel("归一化功率")
        self.spectrum_canvas.axes.set_title("MUSIC 角度谱")
        self.spectrum_canvas.axes.grid(True)
        
        # 添加垂直线标记估计角度
        for i, elev in enumerate(result.estElevations):
            self.spectrum_canvas.axes.axvline(
                x=elev, color='r', linestyle='--', 
                label=f"信源 {i+1}: {elev:.1f}°"
            )
            
        self.spectrum_canvas.axes.legend()
        
        # 刷新画布
        self.spectrum_canvas.draw()
        
    def plot_ambiguity_curves(self, result):
        """绘制角度模糊曲线"""
        # 清除当前图像
        self.ambiguity_canvas.axes.clear()
        
        # 检查是否有角度和时间点数据
        if hasattr(result, 'angles') and hasattr(result, 'anglesTimepoints') and \
           len(result.angles) > 0 and len(result.anglesTimepoints) > 0:
                
            # 绘制散点图
            self.ambiguity_canvas.axes.scatter(
                result.anglesTimepoints, result.angles, 
                s=10, alpha=0.7, marker='.', c='blue'
            )
            
            # 添加标题和标签
            self.ambiguity_canvas.axes.set_xlabel("时间 (秒)")
            self.ambiguity_canvas.axes.set_ylabel("角度 (°)")
            self.ambiguity_canvas.axes.set_title("角度模糊曲线分布")
            self.ambiguity_canvas.axes.grid(True)
            
        else:
            # 如果没有数据，显示提示信息
            self.ambiguity_canvas.axes.text(
                0.5, 0.5, "没有角度模糊曲线数据", 
                horizontalalignment='center', verticalalignment='center',
                transform=self.ambiguity_canvas.axes.transAxes
            )
            
        # 刷新画布
        self.ambiguity_canvas.draw()
        
    def plot_accumulator(self, result):
        """绘制参数空间累加器"""
        # 清除当前图像
        self.accumulator_colorbar.remove() if hasattr(self, 'accumulator_colorbar') else None
        self.accumulator_canvas.axes.clear()
        
        # 检查是否有累加器数据
        if hasattr(result, 'accumulator') and result.accumulator.size > 0:
            # 获取累加器数据
            acc = result.accumulator
            
            # 创建坐标网格
            if hasattr(result, 'elevations') and hasattr(result, 'azimuths') and \
               len(result.elevations) > 0 and len(result.azimuths) > 0:
                elevations = result.elevations
                azimuths = result.azimuths
            else:
                elevations = np.linspace(0, 90, acc.shape[0])
                azimuths = np.linspace(0, 360, acc.shape[1])
                
            # 绘制热力图
            im = self.accumulator_canvas.axes.imshow(
                acc, aspect='auto', origin='lower',
                extent=[min(azimuths), max(azimuths), min(elevations), max(elevations)],
                cmap='jet'
            )
            
            # 添加颜色条到子图中，便于后续清除
            self.accumulator_colorbar = self.accumulator_canvas.fig.colorbar(
                im, ax=self.accumulator_canvas.axes
            )
            self.accumulator_colorbar.set_label("累加值")
            
            # 标记估计位置
            for i, (elev, azim) in enumerate(zip(result.estElevations, result.estAzimuths)):
                self.accumulator_canvas.axes.plot(
                    azim, elev, 'ro', markersize=8,
                    label=f"信源 {i+1}: ({elev:.1f}°, {azim:.1f}°)"
                )
                
            # 添加标题和标签
            self.accumulator_canvas.axes.set_xlabel("方位角 (°)")
            self.accumulator_canvas.axes.set_ylabel("俯仰角 (°)")
            self.accumulator_canvas.axes.set_title("参数空间累加器")
            self.accumulator_canvas.axes.legend()
            
        else:
            # 如果没有数据，显示提示信息
            self.accumulator_canvas.axes.text(
                0.5, 0.5, "没有累加器数据", 
                horizontalalignment='center', verticalalignment='center',
                transform=self.accumulator_canvas.axes.transAxes
            )
            
        # 刷新画布
        self.accumulator_canvas.draw()
        
    def update_result_table(self, result):
        """更新结果表格"""
        # 清除当前表格
        self.result_table.setRowCount(0)
        
        # 添加估计结果
        for i, (elev, azim) in enumerate(zip(result.estElevations, result.estAzimuths)):
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)
            
            self.result_table.setItem(row_position, 0, QTableWidgetItem(str(i+1)))
            self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{elev:.2f}"))
            self.result_table.setItem(row_position, 2, QTableWidgetItem(f"{azim:.2f}"))
        
    def save_results(self):
        """保存结果"""
        if self.doa_result is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return
            
        # 打开保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "NumPy文件 (*.npz);;文本文件 (*.txt);;所有文件 (*)"
        )
        
        if not file_path:
            return
            
        try:
            # 根据文件扩展名选择保存方式
            if file_path.endswith('.npz'):
                # 保存为NumPy压缩文件
                np.savez(
                    file_path,
                    elevations=self.doa_result.estElevations,
                    azimuths=self.doa_result.estAzimuths,
                    accumulator=self.doa_result.accumulator if hasattr(self.doa_result, 'accumulator') else None,
                    angles=self.doa_result.angles if hasattr(self.doa_result, 'angles') else None,
                    timepoints=self.doa_result.anglesTimepoints if hasattr(self.doa_result, 'anglesTimepoints') else None
                )
            else:
                # 保存为文本文件
                with open(file_path, 'w') as f:
                    f.write("RIDOA 估计结果\n")
                    f.write("=================\n\n")
                    
                    f.write("估计信源:\n")
                    for i, (elev, azim) in enumerate(zip(self.doa_result.estElevations, self.doa_result.estAzimuths)):
                        f.write(f"信源 {i+1}: 俯仰角 = {elev:.2f}°, 方位角 = {azim:.2f}°\n")
                        
            self.log_widget.add_log(f"结果已保存到: {file_path}")
            
        except Exception as e:
            self.log_widget.add_log(f"保存结果失败: {str(e)}", "red")
        
    def new_project(self):
        """新建项目"""
        # 询问用户是否保存当前项目
        if self.signal_data is not None or self.doa_result is not None:
            reply = QMessageBox.question(
                self, "新建项目", "是否保存当前项目?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_project()
            elif reply == QMessageBox.Cancel:
                return
                
        # 重置数据
        self.signal_data = None
        self.doa_result = None
        
        # 重置UI
        self.config_to_ui()
        
        # 清除图表
        self.signal_canvas.axes.clear()
        self.signal_canvas.draw()
        
        self.spectrum_canvas.axes.clear()
        self.spectrum_canvas.draw()
        
        self.ambiguity_canvas.axes.clear()
        self.ambiguity_canvas.draw()
        
        self.accumulator_canvas.axes.clear()
        self.accumulator_canvas.draw()
        
        # 清除结果表格
        self.result_table.setRowCount(0)
        
        # 禁用按钮
        self.estimate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # 更新日志
        self.log_widget.add_log("已创建新项目")
        
        # 更新状态栏
        self.status_bar.showMessage("新项目")
        
    def open_project(self):
        """打开项目"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "RIDOA项目文件 (*.ridoa);;所有文件 (*)"
        )
        
        if not file_path:
            return
            
        self.log_widget.add_log("项目加载功能尚未实现")
        
    def save_project(self):
        """保存项目"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "RIDOA项目文件 (*.ridoa);;所有文件 (*)"
        )
        
        if not file_path:
            return
            
        self.log_widget.add_log("项目保存功能尚未实现")
        
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, "关于RIDOA",
            "RIDOA - 基于旋转干涉仪的多目标参数估计系统\n\n"
            "版本: 0.1.0\n"
            "许可: MIT\n\n"
            "© 2025 RIDOA开发团队"
        )
        
    def update_resource_info(self):
        """更新资源信息"""
        try:
            import psutil
            
            # 获取CPU和内存使用情况
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # 更新状态栏
            self.system_info_label.setText(
                f"CPU: {cpu_usage:.1f}% | 内存: {memory_usage:.1f}% | "
                f"{'GPU: 启用' if self.config.useGPU else 'CPU模式'}"
            )
            
        except ImportError:
            # psutil不可用，显示简化信息
            self.system_info_label.setText(
                f"{'GPU: 启用' if self.config.useGPU else 'CPU模式'}"
            )

def main():
    """主程序入口"""
    try:
        app = QApplication(sys.argv)
        
        # 设置应用样式
        app.setStyle('Fusion')
        
        # 创建并显示主窗口
        window = RIDOAApp()
        window.show()
        
        # 执行应用
        sys.exit(app.exec_())
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()