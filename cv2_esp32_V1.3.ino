#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SERVOMIN 102
#define SERVOMAX 480 
#define SERVO_FREQ 50 
#define MAX_ITERATIONS 150  // 统一迭代次数
#define STEP_SIZE 5         // 步长
#define GESTURE_LENGTH 6    // 手势数据长度

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); 

bool state0[GESTURE_LENGTH] = {false, false, false, false, false, false};
bool state1[GESTURE_LENGTH] = {false, false, false, false, false, false};
bool change = false;
char sData;
String state;

// 手指对应的舵机通道
int wrist = 0;        // 手腕
int indexFinger = 1;  // 食指
int middle = 6;       // 中指
int ring = 3;         // 无名指
int thumb = 4;        // 拇指
int pinky = 5;        // 小指

int fingerPins[] = {wrist, indexFinger, middle, ring, thumb, pinky};

// 角度转换函数
int degToPwm(int degree) {
  return map(degree, 0, 180, SERVOMIN, SERVOMAX);
}

// 舵机初始化函数 - 所有舵机回到0度（伸直位置）
void initializeServos() {
  Serial.println("Initializing servos to starting positions...");
  
  int zeroDegPwm = degToPwm(0);   // 0度对应的PWM值
  int wristStartPwm = degToPwm(45); // 手腕初始角度
  
  // 设置初始位置
  pwm.setPWM(wrist, 0, wristStartPwm);
  pwm.setPWM(indexFinger, 0, zeroDegPwm);
  pwm.setPWM(middle, 0, zeroDegPwm);
  pwm.setPWM(ring, 0, zeroDegPwm);
  pwm.setPWM(thumb, 0, zeroDegPwm);
  pwm.setPWM(pinky, 0, zeroDegPwm);
  
  delay(1000); // 等待舵机到达初始位置
  Serial.println("Servos initialized");
}

// 验证接收到的数据是否有效
bool validateGestureData(String data) {
  if (data.length() != GESTURE_LENGTH) {
    Serial.println("Error: Invalid data length");
    return false;
  }
  
  for (int i = 0; i < GESTURE_LENGTH; i++) {
    char c = data.charAt(i);
    if (c != '0' && c != '1') {
      Serial.println("Error: Invalid character in gesture data");
      return false;
    }
  }
  return true;
}

TaskHandle_t receiveData;

// 数据接收任务
void receiveDataCode(void * parameter) {
  for(;;) {
    while(Serial.available()) {
      sData = Serial.read();
      delay(2);
      
      if(sData == '\n') {
        // 验证数据有效性
        if(validateGestureData(state)) {
          // 解析接收到的6位字符串
          for(int i = 0; i < GESTURE_LENGTH; i++) {
            state0[i] = (state.charAt(i) == '1');
          }
          change = true;
          
          // 调试输出
          Serial.print("Received valid gesture: ");
          for(int i = 0; i < GESTURE_LENGTH; i++) {
            Serial.print(state0[i] ? "1" : "0");
          }
          Serial.println();
        }
        state = ""; // 清空字符串
        break;
      } else if (sData >= '0' && sData <= '1') {
        // 只接受0和1字符
        state += sData;
      } else {
        // 忽略无效字符
        Serial.println("Warning: Invalid character ignored");
      }
      
      // 防止字符串过长
      if (state.length() > GESTURE_LENGTH) {
        state = "";
        Serial.println("Warning: Data too long, buffer cleared");
      }
    }
    delay(2);
  }
}

// 改进的手指移动函数
void moveFinger(int fingerId, bool flex, int iteration) {
  if (flex) {
    // 弯曲逻辑

      float fPwm = SERVOMIN + (SERVOMAX - SERVOMIN) * float(iteration) / float(150);
      int iPwm = round(fPwm);
      pwm.setPWM(fingerId, 0,iPwm);
  } 
  else {
    // 伸直逻辑：从弯曲位置(510)渐进到伸直位置(102)
    float fPwm = SERVOMAX - (SERVOMAX - SERVOMIN) * float(iteration) / float(150);
    int iPwm = round(fPwm);
    pwm.setPWM(fingerId, 0, iPwm);
  }
}

// 检查是否有手指状态改变
bool hasStateChanged() {
  for (int i = 0; i < GESTURE_LENGTH; i++) {
    if (state0[i] != state1[i]) {
      return true;
    }
  }
  return false;
}

void setup() {
  Serial.begin(9600);
  Serial.println("ESP32-S3 Robotic Hand Control Starting...");
  
  // 初始化 I2C (ESP32-S3 默认: SDA=GPIO8, SCL=GPIO9)
  Wire.begin();
  
  // 初始化舵机驱动器
  pwm.begin();
  pwm.setOscillatorFrequency(25000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(10);
  
  Serial.println("PWM Driver initialized");

  // 初始化舵机到起始位置
  initializeServos();
  
  // 创建数据接收任务
  xTaskCreatePinnedToCore(
    receiveDataCode,
    "receiveData",      // 修正任务名称
    10000,
    NULL,
    1,                  // 提高任务优先级
    &receiveData,       // 修正变量名
    0);
    
  Serial.println("Data receive task created");
  Serial.println("Ready to receive gesture commands (format: 6-digit binary string)");
  Serial.println("Example: 101010 (wrist=1, index=0, middle=1, ring=0, thumb=1, pinky=0)");
  delay(500);
}

void loop() {
  if(change && hasStateChanged()) {
    Serial.println("Processing gesture change...");
    
    // 渐进式移动舵机
    for(int i = STEP_SIZE; i <= MAX_ITERATIONS; i += STEP_SIZE) {
      // 控制每个手指
      for(int j = 0; j < GESTURE_LENGTH; j++) {
        if(state0[j] != state1[j]) {
          moveFinger(fingerPins[j], state0[j], i);
          delay(1); // 减少延迟，提高响应速度
        } 
      }
      delay(15); // 控制动作速度，可根据需要调整
    }
    
    // 更新状态
    for(int i = 0; i < GESTURE_LENGTH; i++) {
      state1[i] = state0[i];
    }

    change = false;
    Serial.println("Gesture processing complete");
    
    // 输出当前状态
    Serial.print("Current finger states: ");
    for(int i = 0; i < GESTURE_LENGTH; i++) {
      Serial.print(state1[i] ? "1" : "0");
    }
    Serial.println();
  }
  
  delay(50); // 减少主循环延迟
}

// 可选：添加紧急停止功能
void emergencyStop() {
  Serial.println("Emergency stop activated!");
  // 停止所有舵机
  for(int i = 0; i < GESTURE_LENGTH; i++) {
    pwm.setPWM(fingerPins[i], 0, 0);
  }
}
