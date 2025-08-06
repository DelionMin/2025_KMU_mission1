import time
import sys
import os
import signal
import numpy as np
import cv2
from math import inf,sin,atan,cos
from ultralytics import YOLO

# 이 코드(class)는 step함수에서 다음과 같이 동작한다.
# 1. 이미지(카메라 데이터)와 class 내의 state를 기반으로 차선의 위치를 알아낸다.
#   1.1. 이미지를 항공뷰로 변환한다.
# 2. 차선의 위치를 기반으로 position을 구한다.
# 3. p제어를 하여 앞바퀴의 각도를 반환한다. 속도는 임의로 설정한다.

#<YOLO 모델 학습 후 추가>
model = YOLO("/home/xytron/xycar_ws/src/xycar_drive/xycar_drive/detector.pt")

class ChangeDrive:
  INIT = 0
  line_change = 1
  detect = 2
  return_1 = 3
  return_2 = 4

  
  def __init__(self) -> None:
    # 차선 인식 시 사용할 기울기 THRESHOLDS
    # TO BE TUNED
    self.THRESHOLD_SLOPE = 0.6
    self.height = 0
    self.width = 0

    # 가장 최근에 인식된 best_line (buffer)
    self.best_line_left_prev = None
    self.best_line_right_prev = None
    
    self.state = self.INIT
    self.state2 = 1

    # 현재 차선
    self.lane_constant = 0

    self.y1=0
    self.y2=0
    self.x1=0
    self.x2=0

    self.camere_diff = 0.2 # 카메라와 중심 거리

    # 클래스 ID 매핑 (detector.pt 모델에 따라 조정 필요)
    self.class_names = {
      'signal_red': 0,
      'signal_yellow': 1,
      'signal_green': 2,
      'rubbercone': 3,
      'car' : 4
    }

    self.yolo_list = []
    
    #라이다
    # 전방 180° 인덱스(505포인트 기준)
    # 전방 30도 272→233 (40개)
    self.MIN_VALID_DISTANCE = 0.01  # 최소 유효 거리 [m]
    self.confidence_threshold = 0.6
    self._front_indices = np.arange(283, 221, -1)  # 258‥247 (12)
    
    #가까운 차량 추종
    self.prev_error = 0
    self.min_Kp = 1
    self.min_Kd = 3 # 미분
    self.min_dt = 0.1
    
    self.error = 0


    #카메라 노출값 조정
    os.system('v4l2-ctl -d /dev/videoCAM -c auto_exposure=1')
  
    
    
  def _get_front_ranges(self):
    """전방 배열(front_ranges, 길이 40)을 반환."""
    front_range = self._ranges[self._front_indices]
    return front_range  # 길이 40


  
  def get_value(self, image, ranges, ultrasonic):
    """
    센서 값 받아오기 위한 메소드
    """
    self._image = image
    self._ranges = np.array(ranges) if not isinstance(ranges, np.ndarray) else ranges


  def lane_color_detection(self,image, roi_vertices): # default 상태일 때 앙쪽 차선 색깔 판별
    """
    image: BGR 이미지 (OpenCV 읽은 원본)
    roi_vertices: 차선 후보가 있을 것으로 예상되는 다각형 좌표 (np.array 형태)
    
    returns:
    - mask_white, mask_yellow: 차선 후보 영역 내 흰색, 노란색 마스크
    - color_detected: 'white', 'yellow', 'unknown' 중 판별 결과
    """

    # 1. ROI 마스크 생성 (차선 후보 영역)
    mask = np.zeros_like(image[:,:,0])  # 단일 채널 mask
    cv2.fillPoly(mask, [roi_vertices], 255)
    roi_image = cv2.bitwise_and(image, image, mask=mask)

    # 2. BGR → HSV 변환
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    # 3. 흰색 범위 (Hue 무관, 낮은 채도, 높은 명도)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 4. 노란색 범위 (Hue 약 20~30, 채도 높음, 명도 중간 이상)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5. 흰색과 노란색 픽셀 개수 카운트
    white_count = cv2.countNonZero(mask_white)
    yellow_count = cv2.countNonZero(mask_yellow)

    # 6. 색깔 판별
    if white_count > yellow_count and white_count > 200:  # 임계치 500 픽셀은 상황에 맞게 조정
      color_detected = 'white'
    elif yellow_count > white_count and yellow_count > 200:
      color_detected = 'yellow'
    else:
      color_detected = 'unknown'

    return color_detected
  
  def detect_current_lane(self,image,width,height):
    
    roi_left = np.array([[   # 왼쪽 사다리꼴 절반
        (width * 0.1, height),
        (width * 0.3, height * 0.6),
        (width * 0.5, height * 0.6),
        (width * 0.5, height)
    ]], dtype=np.int32)
    

    # 오른쪽
    roi_right = np.array([[   # 오른쪽 사다리꼴 절반
        (width * 0.5, height),
        (width * 0.5, height * 0.6),
        (width * 0.7, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)
    
    left_color = self.lane_color_detection(image, roi_left)
    right_color = self.lane_color_detection(image, roi_right)

    if (left_color == 'white') and (right_color == 'yellow'): # 1차선
      return 1
    
    elif (left_color == 'yellow') and (right_color == 'white'): # 2차선
      return 2
    
    elif (left_color == 'white') and (self.right_color == 'white'): # error
      return 0

    elif (self.left_color == 'yellow') and (self.right_color == 'yellow'): # error
      return 0


  def get_best_line(self, lines):
    '''
    입력된 선들 중에서 대푯값 추출
    gets: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    returns: [[x1, y1, x2, y2]]


    *** 검증 필요 ***
    '''
    best_line = None
    m_left, b_left = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    size = len(lines)
    if size != 0:
      for line in lines:
        x1, y1, x2, y2 = line[0]
        x_sum += x1 + x2
        y_sum += y1 + y2
        if(x2 != x1):
          m_sum += float(y2-y1)/float(x2-x1)
        else:
          m_sum += 0
      
      x_avg = x_sum / (size*2)
      y_avg = y_sum / (size*2)
      m_left = m_sum / size
      b_left = y_avg- m_left * x_avg
    
    if m_left != 0.0:
      y1 = 0
      y2 = self.height_frame    
      '''
      y1, y2 실제 주행 환경에서는 값 맞춰서 바꾸기
      '''

      x1 = int((y1 - b_left) / m_left)
      x2 = int((y2 - b_left) / m_left)

      best_line = [[x1, y1, x2, y2]]

    return best_line

  def find_yellow(self,image_yellow): # 노란색 검출 후 직선의 기울기, 절편 반환
      # 노란 이미지 생성
    hsv = cv2.cvtColor(image_yellow, cv2.COLOR_BGR2HSV)
    
    # 노란색 범위 정의 (Hue: 20~30, Saturation: 100~255, Value: 100~255)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # 노란색 마스크 생성
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 원본 이미지에 마스크 적용하여 노란색만 추출
    yellow_only = cv2.bitwise_and(image_yellow, image_yellow, mask=yellow_mask)
    
    # 노랑 이미지에 GaussianBlur(=정규 분포 양상으로 픽셀 값 깎기, 흐릿하게 만들어서 영상처리에 자주 쓴다)
    # 노이즈 줄이고 선은 더 선명하게 보이도록
    yellow_blur = cv2.GaussianBlur((yellow_only), (5, 5), 0)
    
    # blur 이미지에 Canny 변환해서 Edge detection
    yellow_edges = cv2.Canny(yellow_blur, 50, 150)

    # roi_edges에서 직선 인식
    yellow_lines = cv2.HoughLinesP(yellow_edges, rho=1, theta=np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)

    if yellow_lines is None:
      return 0,0,0
    
    # 왼쪽, 오른쪽 차선 넣어둘 리스트
    final_yellow_lines = []

    for line in yellow_lines:
      x1, y1, x2, y2 = line[0]


      if x1 == x2 and y1 == y2:
        continue

      
      # ***< 영상처리 시 Y값의 경우 아래로 갈수록 커진다는 점 유의 >***
    
      final_yellow_lines.append(line)

    if not final_yellow_lines:
      return 0,0,0

    best_yellow_line = self.get_best_line(final_yellow_lines)    

    x1, y1, x2, y2 = best_yellow_line[0]

    self.x1=x1
    self.x2=x2
    self.y1=y1
    self.y2=y2


    # ax + by + c = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c
    
  def find_white(self,image_white): # 하얀색 검출 후 직선의 기울기, 절편 반환
      # 하얀 이미지 생성
    hsv = cv2.cvtColor(image_white, cv2.COLOR_BGR2HSV)
    

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # 하얀색 마스크 생성
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 원본 이미지에 마스크 적용하여 하얀색만 추출
    white_only = cv2.bitwise_and(image_white, image_white, mask=white_mask)
    
    # 하얀 이미지에 GaussianBlur(=정규 분포 양상으로 픽셀 값 깎기, 흐릿하게 만들어서 영상처리에 자주 쓴다)
    # 노이즈 줄이고 선은 더 선명하게 보이도록
    white_blur = cv2.GaussianBlur((white_only), (5, 5), 0)
    
    # blur 이미지에 Canny 변환해서 Edge detection
    white_edges = cv2.Canny(white_blur, 50, 150)

    # roi_edges에서 직선 인식
    white_lines = cv2.HoughLinesP(white_edges, rho=1, theta=np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)

    if white_lines is None:
      return 0,0,0

    # 왼쪽, 오른쪽 차선 넣어둘 리스트
    final_white_lines = []

    for line in white_lines:
      x1, y1, x2, y2 = line[0]


      if x1 == x2 and y1 == y2:
        continue

      
      # ***< 영상처리 시 Y값의 경우 아래로 갈수록 커진다는 점 유의 >***
    
      final_white_lines.append(line)

    if not final_white_lines:
      return 0,0,0

    best_white_line = self.get_best_line(final_white_lines)    

    x1, y1, x2, y2 = best_white_line[0]


    # ax + by + c = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c

  def _detect_car(self):
    """
    YOLO 모델로 상대차량 감지,
    바운딩 박스 중심점, 넓이, 높이 리스트만 반환
    """
    if self._image is None:
        return []

    detected_boxes = []

    try:
      results = model(self._image, verbose=False)
      for result in results:
        boxes = result.boxes
        if boxes is not None:
          for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            if (confidence > self.confidence_threshold and 
              class_id == self.class_names['car']):
              xyxy = box.xyxy[0].cpu().numpy()
              x1, y1, x2, y2 = xyxy
              center = [(x1 + x2) / 2, (y1 + y2) / 2]
              width = x2 - x1
              height = y2 - y1
              detected_boxes.append({
                'center': center,
                'width': width,
                'height': height
              })
      return detected_boxes

    except Exception as e:
      print(f"Error in _detect_car: {e}")
      return []
      
    '''
    반환 예시
    [
    {
        'center': [190.65, 270.45],
        'width': 140.5,
        'height': 139.5
    },
    {
        'center': [460.0, 225.0],
        'width': 120.0,
        'height': 150.0
    }
  ]
    '''
    
  def _result_car(self):
    """
    YOLO 모델로 상대차량 감지,
    바운딩 박스 중심점, 넓이, 높이 리스트만 반환
    """
    if self._image is None:
        return []

    try:
      results = model(self._image, verbose=False)


      return results

    except Exception as e:
      print(f"Error in _detect_car: {e}")
      return []


  def find_closeObstacle_lane(self, image):
    '''    {
        'center': [190.65, 270.45],
        'width': 140.5,
        'height': 139.5
    }

  yolo = [100, 100, 100, 100] #알아서 yolo로 받아오기

    '''
    yolo = self._detect_car()
    yolo_y = yolo['center'][1]     
    yolo_x = yolo['center'][0]
    yolo_width = yolo['width']
    yolo_height = yolo['height']

    a,b,c = self.find_yellow(image)
  
    if b == 0:
      yellow_x = -(c/a)

    elif a == 0:
      return 0

    elif (a/b) != 0:
      k = yolo_y + yolo_height//2
      yellow_x = (-c-b*k)/a


      if (yellow_x > yolo_x): # 1차선
        return 1
      
      elif(yellow_x < yolo_x): #2차선
        return 2
      
      elif(yellow_x == yolo_x):
        return 0

  def _get_valid_distance(self, ranges_subset):
    """
    유효한 거리 값들에서 최소값 계산
    """
    valid_ranges = ranges_subset[ranges_subset > self.MIN_VALID_DISTANCE]
    return valid_ranges.min() if valid_ranges.size > 0 else np.inf

  def change_lane(self):


#-------------------------------------- 속도 pid 제어
    image = self._image
    
    self.yolo_list = self._detect_car()
    results = self._result_car()

    #속도 = 가까운 차 추종
    center = self._ranges[self._front_indices]
    dist_min = self._get_valid_distance(center)
    
    self.error = dist_min -50 # 50 목표 거리
    
    derivative = (self.error-self.prev_error)
    speed = self.min_Kp * self.error + self.min_Kd * derivative
        # 이전 오차 저장
    self.prev_error = self.error

  
#------------------------------------------------- state masine 시작

#------------------------------------------------- init state

    if self.state == self.INIT:
      
      #중앙선 따라가는 pid
      Kp_center = 1
      Kd_center = 3 # 미분
      Ka_center = 3 # 각도비례
      prev_error_center = 0
      derivative_center = 0
      error1_center = 0
      error2_center = 0


      #조향 pid
      Kp_angle = 1
      Kd_angle = 3 # 미분
      Ka_angle = 3 # 각도비례
      prev_error_angle = 0
      derivative_angle = 0
      error1_angle = 0
      error2_angle = 0

      self.height, self.width = image.shape[:2]
      angle = 0
      if (self.height>0) and (self.width>0):
        self.state = self.line_change

# -------------------------------------------------- line_change state

    elif self.state == self.line_change:
      a,b,c = self.find_yellow(image)
  
      # PD 제어 구현 ->  선 위치 맞추기, 선 각도 , 라이다나 옆차선 차량 유무 확인
      
      # 위치 맞추기 pid


      if not ((self.width/2-10)<(-(c+b*self.height)/a)<(self.width/2+10)):
        target1_center = 320
        current1_center = (-(c+b*self.height)/a)
        target2_center = 0

        if b==0:
          current2_center=0
        else:
          current2_center = -(a/b)

        error1_center = target1_center - current1_center
        error2_center = target2_center - current2_center
        derivative_center = (error1_center - prev_error_center) 

        angle = Kp_center * error1_center + Kd_center * derivative_center - Ka_center * error2_center
        prev_error_center = error1_center


      a,b,c = self.find_yellow(image)

      if a == 0:
        pass

      elif (b == 0) and ((self.width/2-10)<(-c/a)<(self.width/2+10)):
        self.state = self.detect
      
      elif (b == 0):
        pass
      
      elif ((self.width/2-10)<(-(c+b*self.height)/a)<(self.width/2+10)) and ((self.width/2-10)<(-c/a)<(self.width/2+10)):
        self.state = self.detect

# -------------------------------------------------- detect state

    elif self.state == self.detect:
      angle = 0



      if len(self.yolo_list) == 2:
        self.state = self.line_change
          
    
      elif len(self.yolo_list) == 1:
        g = self.find_closeObstacle_lane(image)
        
        if g==2:
          self.state = self.return_1
        elif g==1:
          self.state = self.return_2
        else:
          pass
      else:
        pass
      
# -------------------------------------------------- return_1 state

    elif self.state == self.return_1:

      y_a, y_b, y_c = self.find_yellow(image)

      height, width = image.shape[:2]

      # 오른쪽 절반 영역을 검은색(0)으로 설정
      if len(image.shape) == 3:
        # 컬러 이미지
        image[:, width // 2:] = (0, 0, 0)

      else:
        # 흑백 이미지
        image[:, width // 2:] = 0

      w_a, w_b, w_c = self.find_white(image)
      

      #pid 시작

      target1_angle = 320
      current1_angle = ( (-(w_c + w_b * self.height) / w_a) + (-(y_c + y_b * self.height) / y_a) ) / 2

      target2_angle = 0

      if w_b==0:
        m_w=0
      else:
        m_w = -(w_a/w_b)

      if y_b==0:
        m_y=0
      else:
        m_y = -(y_a/y_b)

      current2_angle = m_w + m_y

      error1_angle = target1_angle - current1_angle
      error2_angle = target2_angle - current2_angle
      derivative_angle = (error1_angle - prev_error_angle) 

      angle = Kp_angle * error1_angle + Kd_angle * derivative_angle - Ka_angle * error2_angle
      prev_error_angle = error1_angle

# -------------------------------------------------- return_2 state

    elif self.state == self.return_2:

      y_a, y_b, y_c = self.find_yellow(image)
      height, width = image.shape[:2]

      # 왼쪽 절반 영역을 검은색(0)으로 설정
      if len(image.shape) == 3:
        # 컬러 이미지
        image[:, :width // 2] = (0, 0, 0)
      else:
        # 흑백 이미지
        image[:, :width // 2] = 0


      w_a, w_b, w_c = self.find_white(image)
      
      
      #pid 시작

      target1_angle = 320
      current1_angle = ( (-(w_c + w_b * self.height) / w_a) + (-(y_c + y_b * self.height) / y_a) ) / 2

      target2_angle = 0

      if w_b==0:
        m_w=0
      else:
        m_w = -(w_a/w_b)

      if y_b==0:
        m_y=0
      else:
        m_y = -(y_a/y_b)

      current2_angle = m_w + m_y

      error1_angle = target1_angle - current1_angle
      error2_angle = target2_angle - current2_angle
      derivative_angle = (error1_angle - prev_error_angle) 

      angle = Kp_angle * error1_angle + Kd_angle * derivative_angle - Ka_angle * error2_angle
      prev_error_angle = error1_angle
    
    # debug
    
    yolo_result = results[0]  # 첫 번째 결과 (단일 이미지 처리 기준)
    # 시각화를 위한 BGR 이미지 복사
    image_with_boxes = image.copy()

    # 결과 박스와 라벨 시각화
    for box in yolo_result.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])  # 박스 좌표
      conf = float(box.conf[0])              # 신뢰도
      cls_id = int(box.cls[0])               # 클래스 ID
      label = model.names[cls_id]            # 클래스 이름

      # 박스 그리기
      cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

      # 텍스트 라벨
      text = f"{label} {conf:.2f}"
      cv2.putText(image_with_boxes, text, (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      

    # 노란 차선 표시
    x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
    cv2.line(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 이미지 표시 (필요시)
    cv2.imshow("YOLO Result", image_with_boxes)
    # cv2.waitKey(0)


    return angle,speed
  
  def test(self):
    image = self._image
    
    _, _, _ =self.find_yellow(image)
    
    angle = 0

    self.yolo_list = self._detect_car()
    results = self._result_car()

    #속도 = 가까운 차 추종
    center = self._ranges[self._front_indices]
    dist_min = self._get_valid_distance(center)
    
    error = dist_min -1 # 1 목표 거리
    
    derivative = (error-self.prev_error)
    
    speed = self.min_Kp * error + self.min_Kd * derivative
    if speed <0:
      speed = 0
      
        # 이전 오차 저장
    self.prev_error = error


    # debug

    yolo_result = results[0]  # 첫 번째 결과 (단일 이미지 처리 기준)
    # 시각화를 위한 BGR 이미지 복사


    # 결과 박스와 라벨 시각화
    for box in yolo_result.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])  # 박스 좌표
      conf = float(box.conf[0])              # 신뢰도
      cls_id = int(box.cls[0])               # 클래스 ID
      label = model.names[cls_id]            # 클래스 이름

      # 박스 그리기
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

      # 텍스트 라벨
      text = f"{label} {conf:.2f}"
      cv2.putText(image, text, (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      
    cv2.line(image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)

    cv2.imshow("YOLO Result", image)


    return angle,speed
  
  def test2(self):

    image = self._image

    speed = 5

          #중앙선 따라가는 pid
    Kp_center = 1
    Kd_center = 3 # 미분
    Ka_center = 3 # 각도비례

    a,b,c = self.find_yellow(image)

    # PD 제어 구현 ->  선 위치 맞추기, 선 각도 , 라이다나 옆차선 차량 유무 확인
    
    # 위치 맞추기 pid


    if not ((self.width/2-10)<(-(c+b*self.height)/a)<(self.width/2+10)):
      target1_center = 320
      current1_center = (-(c+b*self.height)/a)
      target2_center = 0

      if b==0:
        current2_center=0
      else:
        current2_center = -(a/b)

      error1_center = target1_center - current1_center
      error2_center = target2_center - current2_center
      derivative_center = (error1_center - prev_error_center) 

      angle = Kp_center * error1_center + Kd_center * derivative_center - Ka_center * error2_center
      prev_error_center = error1_center
    

    return angle, speed

  def step(self):
    
    angle, speed = self.test()
    #angle, speed = self.change_lane()
    return angle, speed
