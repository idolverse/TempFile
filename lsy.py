import cv2
import mediapipe as mp
import time
import serial
import threading
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class HandDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self, frame, handNo=0, draw=False):
        lmList = []

        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    lmList.append([id, cx, cy])

                    if draw and id == 0:
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
        return lmList

def serial_monitor(ser, status_signal):
    """独立线程监听Arduino串口输出"""
    while True:
        try:
            if ser.in_waiting > 0:
                arduino_data = ser.readline().decode('utf-8').strip()
                if arduino_data:
                    status_signal.emit(f"[Arduino]: {arduino_data}")
        except Exception as e:
            status_signal.emit(f"串口连接异常: {str(e)}")
            break
        time.sleep(0.01)

def draw_text_with_chinese(frame, text, position, font_size=16, color=(255, 255, 0)):
    """使用PIL绘制中文文本（适配小屏幕字体）"""
    try:
        # 将OpenCV的BGR格式转为PIL的RGB格式
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 设置中文字体路径
        font_path = None
        try:
            # Windows系统默认中文字体
            font_path = "C:/Windows/Fonts/simhei.ttf"
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        except:
            try:
                # Linux系统默认中文字体
                font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            except:
                try:
                    # macOS系统默认中文字体
                    font_path = "/System/Library/Fonts/PingFang.ttc"
                    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                except:
                    # 如果都找不到，使用默认字体
                    font = ImageFont.load_default()
                    print("警告: 未找到中文字体，使用默认字体")
        
        # 绘制文本
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))  # PIL使用RGB顺序
        
        # 将PIL图像转回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文本绘制错误: {e}")
        # 出错时返回原始帧
        return frame

class VideoThread(QThread):
    """视频处理线程"""
    update_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(str)
    
    def __init__(self, detector, ser, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.ser = ser
        self.running = False
        self.hand = [["手腕", False], ["食指", False], ["中指", False], 
                    ["无名指", False], ["拇指", False], ["小指", False]]
        self.frame_count = 0
        self.PROCESSING_INTERVAL = 1
        self.WINDOW_SIZE = 5
        self.finger_history = {
            0: deque(maxlen=self.WINDOW_SIZE),  # 手腕
            1: deque(maxlen=self.WINDOW_SIZE),  # 食指
            2: deque(maxlen=self.WINDOW_SIZE),  # 中指
            3: deque(maxlen=self.WINDOW_SIZE),  # 无名指
            4: deque(maxlen=self.WINDOW_SIZE),  # 拇指
            5: deque(maxlen=self.WINDOW_SIZE)   # 小指
        }
        for i in range(self.WINDOW_SIZE):
            for finger in self.finger_history:
                self.finger_history[finger].append(False)
        
        # 视频优化参数
        self.resize_frame = True  # 是否调整帧尺寸
        self.target_width = 640   # 目标宽度（小屏幕优化）
        self.target_height = 480  # 目标高度
        self.skip_frames = 1      # 跳帧处理，每N帧处理1帧
        self.current_skip = 0
        
    def run(self):
        try:
            self.running = True
            prevTime = 0
            
            # 打开摄像头（添加错误处理）
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.update_status.emit("摄像头打开失败")
                return
                
            # 设置摄像头分辨率（小屏幕优化）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.update_status.emit(f"系统就绪，正在检测手势...")

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.update_status.emit("读取帧失败")
                    break
                    
                self.frame_count += 1
                self.current_skip += 1
                
                # 跳帧处理，减少计算量
                if self.current_skip <= self.skip_frames:
                    continue
                self.current_skip = 0
                
                # 调整帧尺寸（如果原始尺寸过大）
                if self.resize_frame and (frame.shape[1] > self.target_width or frame.shape[0] > self.target_height):
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                
                # 始终检测手部并绘制关键点
                frame = self.detector.findHands(frame)
                lmList = self.detector.findPosition(frame)
                
                # 每帧都检测手指状态，但只在必要时更新平均值
                current_state = [False] * 6  # 初始化当前帧的手指状态
                
                if len(lmList) > 0: 
                    j = 1
                    
                    for i in range(1, 6):
                        if i == 1:  # 拇指检测
                            if lmList[4][1] > lmList[3][1]:
                                current_state[4] = True  # 拇指弯曲
                        else:  # 其他四指检测
                            finger_tip = i * 4
                            finger_pip = i * 4 - 2
                            
                            if finger_tip < len(lmList) and finger_pip < len(lmList):
                                if lmList[finger_tip][2] > lmList[finger_pip][2]:
                                    current_state[j] = True  # 手指弯曲
                            
                            if j == 3:
                                j += 2
                            else:
                                j += 1
                
                # 更新滑动窗口数据
                for i in range(6):
                    self.finger_history[i].append(current_state[i])
                
                # 每5帧计算一次平均值并决定最终状态
                if self.frame_count % self.WINDOW_SIZE == 0:
                    change = False
                    threshold = self.WINDOW_SIZE // 2  # 超过半数帧为True则认为弯曲
                    
                    for i in range(6):
                        # 计算平均值
                        count_true = sum(self.finger_history[i])
                        new_state = count_true > threshold
                        
                        # 如果状态变化，记录变化
                        if new_state != self.hand[i][1]:
                            self.hand[i][1] = new_state
                            change = True
                            self.update_status.emit(f"[Python] Frame {self.frame_count}: {self.hand[i][0]}: {'弯曲' if new_state else '伸直'}")
                    
                    # 如果状态变化，发送新命令
                    if change and self.ser and self.ser.is_open:
                        msg = ""
                        for i in range(6):
                            if self.hand[i][1]:
                                msg += "1"
                            else:
                                msg += "0"

                        msg += '\n'
                        self.update_status.emit(f"[Python] Sending: {msg.strip()}")
                        
                        try:
                            self.ser.write(msg.encode("ascii"))
                            self.ser.flush()
                        except serial.SerialException as e:
                            self.update_status.emit(f"[Python] Serial write error: {e}")
                
                # 计算并显示实际FPS（字体大小调整为18）
                currentTime = time.time()
                if prevTime != 0:
                    fps = 1 / (currentTime - prevTime)
                    frame = draw_text_with_chinese(frame, f"实际FPS: {int(fps)}", (10, 50), 18, (255, 0, 255))
                prevTime = currentTime
                
                # 显示处理参数（字体大小调整为18）
                frame = draw_text_with_chinese(frame, f"滑动窗口: {self.WINDOW_SIZE}帧", (10, 80), 18, (255, 255, 0))
                frame = draw_text_with_chinese(frame, f"帧计数: {self.frame_count}", (10, 110), 18, (255, 255, 0))

                # 添加状态显示（字体大小调整为16，间距缩小）
                y_offset = 140
                for i, (name, state) in enumerate(self.hand):
                    color = (0, 255, 0) if state else (0, 0, 255)
                    frame = draw_text_with_chinese(
                        frame, 
                        f"{name}: {'弯曲' if state else '伸直'}", 
                        (10, y_offset + i * 30),  # 行间距缩小
                        16,  # 字体大小减小
                        color
                    )

                # 转换BGR到RGB用于Qt显示
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_frame.emit(frame)

            cap.release()
            self.update_status.emit("视频线程已停止")
        except Exception as e:
            self.update_status.emit(f"视频线程异常: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def stop(self):
        self.running = False
        self.wait()  # 等待线程安全退出

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和初始大小
        self.setWindowTitle("手势控制系统")
        
        # 初始化UI
        self.init_ui()
        
        # 串口相关
        self.ser = None
        self.serial_thread = None
        
        # 启动时全屏显示
        self.showFullScreen()
        
    def init_ui(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 水平分割视频区和控制区(保持7:3比例更适合小屏幕)
        main_layout = QHBoxLayout(central_widget)
        
        # 视频显示区
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.StyledPanel)
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label)
        main_layout.addWidget(video_frame, 7)  # 视频区占70%，更适合小屏幕
        
        # 控制区
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        
        # 标题
        title_label = QLabel("手势控制操作")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; margin: 15px 0;")
        control_layout.addWidget(title_label)
        
        # 按钮区
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)  # 按钮间距
        
        # 开始按钮 - 绿色
        self.start_btn = QPushButton("开始程序")
        self.start_btn.setMinimumSize(180, 60)
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                border: 2px solid #45a049;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 2px solid #3e8e41;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.start_btn.clicked.connect(self.start_program)
        button_layout.addWidget(self.start_btn)
        
        # 结束按钮 - 红色
        self.stop_btn = QPushButton("结束程序")
        self.stop_btn.setMinimumSize(180, 60)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                background-color: #f44336;
                color: white;
                border-radius: 10px;
                border: 2px solid #d32f2f;
            }
            QPushButton:hover {
                background-color: #d32f2f;
                border: 2px solid #c62828;
            }
            QPushButton:pressed {
                background-color: #c62828;
            }
        """)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        # 退出按钮 - 灰色
        self.exit_btn = QPushButton("退出程序")
        self.exit_btn.setMinimumSize(180, 50)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 11pt;
                background-color: #607D8B;
                color: white;
                border-radius: 10px;
                border: 2px solid #546E7A;
            }
            QPushButton:hover {
                background-color: #546E7A;
                border: 2px solid #455A64;
            }
            QPushButton:pressed {
                background-color: #455A64;
            }
        """)
        self.exit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.exit_btn)
        
        control_layout.addLayout(button_layout)
        
        # 状态显示区
        status_label = QLabel("状态显示")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("font-size: 13pt; font-weight: bold; margin: 20px 0 10px;")
        control_layout.addWidget(status_label)
        
        self.status_text = QLabel("系统未启动")
        self.status_text.setWordWrap(True)
        self.status_text.setStyleSheet("""
            background-color: #f0f0f0; 
            padding: 10px; 
            border-radius: 6px; 
            font-size: 11pt;
            min-height: 80px;
        """)
        control_layout.addWidget(self.status_text)
        
        # 操作提示
        help_label = QLabel("操作提示")
        help_label.setAlignment(Qt.AlignCenter)
        help_label.setStyleSheet("font-size: 12pt; font-weight: bold; margin: 15px 0 8px;")
        control_layout.addWidget(help_label)
        
        help_text = """
        <p style="font-size:10pt; margin: 5px 0;">• 点击"开始程序"按钮启动手势识别</p>
        <p style="font-size:10pt; margin: 5px 0;">• 系统会检测手指弯曲状态并发送信号</p>
        <p style="font-size:10pt; margin: 5px 0;">• 点击"结束程序"按钮停止手势识别</p>
        <p style="font-size:10pt; margin: 5px 0;">• 点击"退出程序"按钮关闭应用</p>
        <p style="font-size:10pt; margin: 5px 0;">• 视频窗口显示当前手势识别状态</p>
        <p style="font-size:10pt; margin: 5px 0;">• 状态区域显示系统运行信息</p>
        """
        self.help_text = QLabel(help_text)
        self.help_text.setWordWrap(True)
        self.help_text.setStyleSheet("background-color: #f8f9fa; padding: 8px; border-radius: 5px;")
        control_layout.addWidget(self.help_text)
        
        # 版权信息
        copyright_label = QLabel("© 2025 手势控制系统")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("font-size: 9pt; color: #666; margin-top: 20px;")
        control_layout.addWidget(copyright_label)
        
        main_layout.addWidget(control_frame, 3)  # 控制区占30%
    
    def start_program(self):
        """开始程序按钮处理函数"""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_text.setText("系统正在启动...")
        
        try:
            # 打开串口（添加错误处理）
            self.ser = serial.Serial(
                port="COM3",  # 根据实际情况调整
                baudrate=9600,
                timeout=0.1,
                write_timeout=1
            )
            self.status_text.setText(f"串口 {self.ser.port} 打开成功")
            
            # 启动串口监听线程
            self.serial_thread = threading.Thread(
                target=serial_monitor, 
                args=(self.ser, self.update_status),
                daemon=True
            )
            self.serial_thread.start()
            
            # 启动视频处理线程
            self.detector = HandDetector(maxHands=1, detectionCon=0.7)
            self.video_thread = VideoThread(self.detector, self.ser)
            self.video_thread.update_frame.connect(self.update_video_frame)
            self.video_thread.update_status.connect(self.update_status)
            self.video_thread.start()
            
        except serial.SerialException as e:
            self.status_text.setText(f"串口打开失败: {e}")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        except Exception as e:
            self.status_text.setText(f"启动程序失败: {e}")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def stop_program(self):
        """结束程序按钮处理函数"""
        self.status_text.setText("系统正在关闭...")
        
        # 停止视频线程
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
        
        # 关闭串口
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.status_text.setText("串口已关闭")
        
        # 恢复按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("等待视频流...")
        self.status_text.setText("系统已停止")
    
    def update_video_frame(self, frame):
        """更新视频帧显示，确保铺满视频区域"""
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # 让视频帧自适应视频标签大小，保持比例并平滑缩放
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def update_status(self, message):
        """更新状态文本"""
        self.status_text.setText(message)
    
    def resizeEvent(self, event):
        """窗口大小变化时，更新视频显示"""
        if hasattr(self, 'video_label') and self.video_label.pixmap():
            self.video_label.setPixmap(self.video_label.pixmap().scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.stop_program()
        event.accept()

if __name__ == "__main__":
    # 添加全局异常处理
    def exception_hook(exctype, value, traceback):
        print(f"全局异常捕获: {exctype}, {value}")
        print(traceback.format_exc())
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)
    
    sys._excepthook = sys.excepthook
    sys.excepthook = exception_hook
    
    app = QApplication(sys.argv)
    # 设置全局字体，确保中文显示正常
    font = app.font()
    font.setFamily("SimHei")  # Windows/Linux默认中文字体
    app.setFont(font)
    
    window = MainWindow()
    sys.exit(app.exec_())