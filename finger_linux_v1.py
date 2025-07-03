import cv2
import mediapipe as mp
import time
import serial
import threading
from collections import deque
import os

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
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), -1)
        return lmList

def serial_monitor(ser):
    """独立线程监听Arduino串口输出"""
    while True:
        try:
            if ser.in_waiting > 0:
                arduino_data = ser.readline().decode('utf-8').strip()
                if arduino_data:
                    print(f"[Arduino]: {arduino_data}")
        except:
            break
        time.sleep(0.01)

def main():
    prevTime = 0
    hand = [["Wrist", False], ["IndexFinger", False], ["Middle", False], 
            ["Ring", False], ["Thumb", False], ["Pinky", False]]
    
    # 帧计数器和处理间隔
    frame_count = 0
    PROCESSING_INTERVAL = 1  # 每帧都检测，但每5帧取平均值
    
    # 创建滑动窗口存储最近5帧的手指状态
    WINDOW_SIZE = 5
    finger_history = {
        0: deque(maxlen=WINDOW_SIZE),  # 手腕
        1: deque(maxlen=WINDOW_SIZE),  # 食指
        2: deque(maxlen=WINDOW_SIZE),  # 中指
        3: deque(maxlen=WINDOW_SIZE),  # 无名指
        4: deque(maxlen=WINDOW_SIZE),  # 拇指
        5: deque(maxlen=WINDOW_SIZE)   # 小指
    }
    
    # 初始化窗口数据
    for i in range(WINDOW_SIZE):
        for finger in finger_history:
            finger_history[finger].append(False)

    try:
        # 串口配置（Linux下通常为/dev/ttyUSBx或/dev/ttyACM0）
        # 可通过`ls /dev/ttyUSB*`或`ls /dev/ttyACM*`查找Arduino串口
        ser = serial.Serial(
            port="/dev/ttyUSB0",  # 根据实际情况调整
            baudrate=9600,
            timeout=0.1,
            write_timeout=1
        )
        print(f"Serial port {ser.port} opened successfully")
        time.sleep(2)
        
        # 启动串口监听线程
        serial_thread = threading.Thread(target=serial_monitor, args=(ser,), daemon=True)
        serial_thread.start()
        print("Serial monitor started")
        
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        print("请检查串口权限或使用`ls /dev/ttyUSB*`查找正确端口")
        return

    try:
        # Linux下使用V4L2摄像头驱动
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            # 尝试默认驱动（fallback）
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to open camera")
                return
                
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        detector = HandDetector(maxHands=1, detectionCon=0.7)
        
        print(f"System ready. Averaging over {WINDOW_SIZE} frames.")
        print("Press 'q' to quit.")
        print("Arduino output will be displayed with [Arduino] prefix")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            frame_count += 1
            
            # 始终检测手部并绘制关键点
            frame = detector.findHands(frame)
            lmList = detector.findPosition(frame)
            
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
                finger_history[i].append(current_state[i])
            
            # 每5帧计算一次平均值并决定最终状态
            if frame_count % WINDOW_SIZE == 0:
                change = False
                threshold = WINDOW_SIZE // 2  # 超过半数帧为True则认为弯曲
                
                for i in range(6):
                    # 计算平均值
                    count_true = sum(finger_history[i])
                    new_state = count_true > threshold
                    
                    # 如果状态变化，记录变化
                    if new_state != hand[i][1]:
                        hand[i][1] = new_state
                        change = True
                        print(f"[Python] Frame {frame_count}: {hand[i][0]}: {'BENT' if new_state else 'STRAIGHT'}")
                
                # 如果状态变化，发送新命令
                if change:
                    msg = ""
                    for i in range(6):
                        if hand[i][1]:
                            msg += "1"
                        else:
                            msg += "0"

                    msg += '\n'
                    print(f"[Python] Sending: {msg.strip()}")
                    
                    try:
                        ser.write(msg.encode("ascii"))
                        ser.flush()
                    except serial.SerialException as e:
                        print(f"[Python] Serial write error: {e}")
            
            # 计算并显示实际FPS
            currentTime = time.time()
            if prevTime != 0:
                fps = 1 / (currentTime - prevTime)
                cv2.putText(frame, f"Actual FPS: {int(fps)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            prevTime = currentTime
            
            # 显示处理参数
            cv2.putText(frame, f"Averaging over {WINDOW_SIZE} frames", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 添加状态显示
            y_offset = 160
            for i, (name, state) in enumerate(hand):
                color = (0, 255, 0) if state else (0, 0, 255)
                cv2.putText(frame, f"{name}: {'BENT' if state else 'STRAIGHT'}", 
                           (10, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)

            cv2.imshow("Hand Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")
        cv2.destroyAllWindows()
        print("Program terminated")

if __name__ == "__main__":
    main()