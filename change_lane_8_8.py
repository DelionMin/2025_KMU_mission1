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

class ChangeDrive:     # change lane 클래스
  INIT = 0             # 초기 설정 state
  yello_follow = 1      # 중앙선 추종 state
  detect = 2           # 분기 계산 state
  return_1 = 3         # 1차선 복귀 state
  return_2 = 4         # 2차선 복귀 state
  sidedrive_1 = 5      # 1차선 주행 중인데 2차선에 방해차량이 있는 case
  sidedrive_2 = 6      # 2차선 주행 중인데 1차선에 방해차량이 있는 case
  
  def __init__(self) -> None:
    # 차선 인식 시 사용할 기울기 THRESHOLDS
    # TO BE TUNED
    
    self.height = 0                 # 화면 높이
    self.width = 0                  # 화면 너비

    
    self.state = self.INIT           # init state 로 시작

    self.lane_constant = 0           # 현재 차선

    # 노란 선 좌표 버퍼
    self.y1=0
    self.y2=0
    self.x1=0
    self.x2=0


    self.camere_diff = 0.2           # 카메라와 중심 거리 (일단 사용x)
    self.THRESHOLD_SLOPE = 0.6       # 기울기 threshold (일단 사용x)


    # 클래스 ID 매핑 (detector.pt 모델에 따라 조정 필요)
    self.class_names = {
      'signal_red': 0,
      'signal_yellow': 1,
      'signal_green': 2,
      'rubbercone': 3,
      'car' : 4
    }

    # yolo 객체 배열
    self.yolo_list = []


    # 라이다
    # 전방 180° 인덱스(505포인트 기준)
    # 전방 30도 272→233 (40개)
    self.MIN_VALID_DISTANCE = 0.01                 # 최소 유효 거리 [m]
    self.confidence_threshold = 0.6                # yolo threshold
    self._front_indices = np.arange(283, 221, -1)  # 258‥247 (12)
    



    #가까운 차량 추종 pid 계수

    self.min_Kp = 0.25
    self.min_Kd = 0.75

    # scaling factor
    self.scaling_factor = 120

    self.error = 0
    self.prev_error = 0

    #카메라 노출값 조정
    os.system('v4l2-ctl -d /dev/videoCAM -c auto_exposure=1')
    
    
  def _get_front_ranges(self):
    """전방 배열(front_ranges, 길이 40)을 반환."""
    front_range = self._ranges[self._front_indices]
    return front_range  # 길이 40
  
  def get_value(self, image, ranges, ultrasonic): # detecter 에서 센서 값을 받아옴
    """
    센서 값 받아오기 위한 메소드
    """
    self._image = image
    self._ranges = np.array(ranges) if not isinstance(ranges, np.ndarray) else ranges

  def lane_color_detection(self,image, roi_vertices): # 차선 색깔 판별
    # 입력 : image, 특정 구역을 가리키는 roi
    # 출력 : white, yellow, unknown 중 해당하는 문자열

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

    # 3. 흰색 범위 (Hue 무관, 낮은 채도, 높은 명도) / 상수값 조정 필요
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 4. 노란색 범위 (Hue 약 20~30, 채도 높음, 명도 중간 이상) / 상수값 조정 필요
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5. 흰색과 노란색 픽셀 개수 카운트
    white_count = cv2.countNonZero(mask_white)
    yellow_count = cv2.countNonZero(mask_yellow)

    # 6. 색깔 판별
    if white_count > yellow_count and white_count > 200:      # 임계치 200 픽셀은 상황에 맞게 조정
      color_detected = 'white'                                # 흰차선

    elif yellow_count > white_count and yellow_count > 200:
      color_detected = 'yellow'                               # 노란차선

    else:
      color_detected = 'unknown'                              # 재판별 요망

    return color_detected
  
  def detect_current_lane(self,image,width,height):   # 차량의 현재 차선 반환

    # 양 쪽 차선의 색깔을 lane_color_detection 함수를 통해 구한 뒤 차선 정보 반환
    # 입력 : image, 카메라 너비, 카메로 높이
    # 출력 : 현재 차선에 해당하는 숫자


    # 왼쪽 roi
    roi_left = np.array([[   # 왼쪽 사다리꼴 절반
        (width * 0.1, height),
        (width * 0.3, height * 0.6),
        (width * 0.5, height * 0.6),
        (width * 0.5, height)
    ]], dtype=np.int32)
    

    # 오른쪽 roi
    roi_right = np.array([[   # 오른쪽 사다리꼴 절반
        (width * 0.5, height),
        (width * 0.5, height * 0.6),
        (width * 0.7, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)
    
    left_color = self.lane_color_detection(image, roi_left)    # 왼쪽 차선 색
    right_color = self.lane_color_detection(image, roi_right)  # 오른쪽 차선 색

    if (left_color == 'white') and (right_color == 'yellow'): # 1차선
      return 1
    
    elif (left_color == 'yellow') and (right_color == 'white'): # 2차선
      return 2
    
    elif (left_color == 'white') and (self.right_color == 'white'): # error
      return 0

    elif (self.left_color == 'yellow') and (self.right_color == 'yellow'): # error
      return 0


  def get_best_line(self, lines): # lane drive에서 가져옴
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
      y2 = self.height 
      '''
      y1, y2 실제 주행 환경에서는 값 맞춰서 바꾸기
      '''

      x1 = int((y1 - b_left) / m_left)
      x2 = int((y2 - b_left) / m_left)

      best_line = [[x1, y1, x2, y2]]

    return best_line

  def find_yellow(self,image_yellow): # 노란색 직선 검출 후 직선 상수 반환, 일단 동작 확인함

    # 입력 : image
    # 출력 : 추출한 노란색 차선의 a, b, c 값 (ax+by+c=0)
    
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

    if yellow_lines is None:   # 차선을 찾지 못하면 0,0,0 반환
      return 0,0,0
    
    # 왼쪽, 오른쪽 차선 넣어둘 리스트
    final_yellow_lines = []

    for line in yellow_lines:
      x1, y1, x2, y2 = line[0]     # 직선 좌표 추출


      if x1 == x2 and y1 == y2:    # 두 점이 같은 점을 가리키면 패스
        continue

      
      # ***< 영상처리 시 Y값의 경우 아래로 갈수록 커진다는 점 유의 >***
    
      final_yellow_lines.append(line)  # final 리스트에 추가

    if not final_yellow_lines:     # final 리스트가 비었다면 0,0,0 반환
      return 0,0,0

    best_yellow_line = self.get_best_line(final_yellow_lines)  # 가장 바람직한 차선 추출

    x1, y1, x2, y2 = best_yellow_line[0]                       # 추출한 차선의 좌표 지정

    # 좌표들을 버퍼에 저장

    self.x1=x1
    self.x2=x2
    self.y1=y1
    self.y2=y2

    # 좌표로부터 a,b,c 값 추출
    # ax + by + c = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return a, b, c
    
  def find_white(self,image_white): # 흰색 직선 검출 후 직선 상수 반환, 일단 동작 확인함
    # 입력 : image
    # 출력 : 추출한 흰색 차선의 a, b, c 값 (ax+by+c=0)
    
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

    if white_lines is None:   # 차선을 찾지 못하면 0,0,0 반환
      return 0,0,0

    # 왼쪽, 오른쪽 차선 넣어둘 리스트
    final_white_lines = []

    for line in white_lines:
      x1, y1, x2, y2 = line[0]    # 직선 좌표 추출


      if x1 == x2 and y1 == y2:   # 두 점이 같은 점을 가리키면 패스
        continue

      
      # ***< 영상처리 시 Y값의 경우 아래로 갈수록 커진다는 점 유의 >***
    
      final_white_lines.append(line)  # final 리스트에 추가

    if not final_white_lines:     # final 리스트가 비었다면 0,0,0 반환
      return 0,0,0

    best_white_line = self.get_best_line(final_white_lines)    # 가장 바람직한 차선 추출

    x1, y1, x2, y2 = best_white_line[0]                        # 추출한 차선의 좌표 지정


    # 좌표로부터 a,b,c 값 추출
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

  def find_closeObstacle_lane(self, image): # 장애물 차량의 차선 정보를 판별 / 검증 필요
    # yolo의 좌표와 장애물의 좌표를 비교해 차선정보 추출
    # 입력 : image
    # 출력 : 장애물이 위치한 차선 숫자

    '''    {
        'center': [190.65, 270.45],
        'width': 140.5,
        'height': 139.5
    }

  yolo = [100, 100, 100, 100] #알아서 yolo로 받아오기

    '''
    yolo = self._detect_car()       # yolo 객체 받아오기
    yolo_y = yolo['center'][1]      # yolo로 인식한 차량의 y 좌표 추출
    yolo_x = yolo['center'][0]      # yolo로 인식한 차량의 x 좌표 추출
    yolo_width = yolo['width']      # yolo로 인식한 차량의 너비 추출
    yolo_height = yolo['height']    # yolo로 인식한 차량의 높이 추출

    a,b,c = self.find_yellow(image) # 노란 직선 계산
    
    if a==0 and b==0 and c==0:      # 직선을 찾지 못하면 0 반환
      return 0
  
    if b == 0:                      # b==0 이면 ax+c=0 꼴의 직선이므로 x=-(c/a)
      yellow_x = -(c/a)

    elif a == 0:                    # a==0 이면 by+c=0 꼴의 직선이므로 오류, 재판단
      return 0

    elif (a/b) != 0:                # 기울기(-a/b)가 0이 아닌 일반적인 직선일때
      car_mid_y = yolo_y + yolo_height//2   # 차량의 밑바닥 중앙 y좌표 계산
      yellow_x = (-c-b*car_mid_y)/a         # 위 좌표와 같은 높이의 노란차선 x좌표 계산


      if (yellow_x > yolo_x): # 1차선
        return 1
      
      elif(yellow_x < yolo_x): #2차선
        return 2
      
      elif(yellow_x == yolo_x): #재판별 요망
        return 0

  def _get_valid_distance(self, ranges_subset):
    """
    유효한 거리 값들에서 최소값 계산
    """
    valid_ranges = ranges_subset[ranges_subset > self.MIN_VALID_DISTANCE]
    return valid_ranges.min() if valid_ranges.size > 0 else np.inf

  def change_lane(self):

    image = self._image
    self.yolo_list = self._detect_car()
    results = self._result_car()

#-------------------------------------- 속도 pid 제어 / 해당 코드는 어떤 state던지 계속 돌아감
    
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
# 초기 설정 state

    if self.state == self.INIT:
      
      #중앙선 따라가는 pid 계수들 초기화
      Kp_center = 1
      Kd_center = 3 

      Kpa_center = 3 
      Kda_center = 3 

      prev_error1_center = 0
      prev_error2_center = 0

      derivative1_center = 0
      derivative2_center = 0

      error1_center = 0
      error2_center = 0


      # 조향(return_1, return_2) pid 계수들 초기화
      # 둘다 대칭을 이루는 같은 조향을 할 것으로 예상되고 한번에 한 state만 실행될 예정이므로 변수통일함

      Kp_angle = 1
      Kd_angle = 3 # 미분
      Ka_angle = 3 # 각도비례
      prev_error_angle = 0
      derivative_angle = 0
      error1_angle = 0
      error2_angle = 0

      # 카메라 정보 받아오기
      self.height, self.width = image.shape[:2]
      
      # 각도 변수 지정
      angle = 0


      # (self.height>0) and (self.width>0) 이란 것은 카메라를 정상적으로 로드할 때를 뜻함
      # 문제시 수정 가능

      # 현재 차선 정보 받아오기
      current_line = self.detect_current_lane(image, self.width, self.height)
      


      # 1차선에서 오른쪽 차량 감지 시 sidedrive_1 state로
      if current_line == 1:
        if (self.height > 0) and (self.width > 0) and (초음파로 오른쪽 차 감지):
          self.state = self.sidedrive_1

        elif (self.height > 0) and (self.width > 0):
          self.state = self.line_change

      # 2차선에서 왼쪽 차량 감지 시 sidedrive_2 state로
      elif current_line == 2:
        if (self.height > 0) and (self.width > 0) and (초음파로 왼쪽 차 감지):
          self.state = self.sidedrive_2

        elif (self.height>0) and (self.width>0):
          self.state = self.line_change

      elif current_line == 0: # error
        pass
          



# -------------------------------------------------- yello_follow state
# 중앙선 추종 state

    elif self.state == self.yello_follow:

      # 노란차선 구하기
      a,b,c = self.find_yellow(image)  
  
      # PD 제어 구현 ->  선 위치 맞추기, 선 각도 , 라이다나 옆차선 차량 유무 확인
    

      # 위치 맞추기 pid
      if a == 0:  # 노답 , x축에 평행한 직선임 / 다시 find yellow 부터
        angle = 0
        pass

      elif a != 0:  # 일단 가로선은 아니다
        
        # 먼저 탈출시킬 상황들이면(노란선을 정확히 추종시) detect 분기로 보낼거
        # self.threshold_mid는 노란선이 중앙에 들어갔다고 판단할 threshold임
          # - 예를들어 10이라면 중앙좌표(self.width/2)에서 +-10까지 허용하는 거

        # 위 프레임과 노란 직선의 교점
        x_top = -(c/a)
        
        # 아래 프레임과 노란 직선의 교점
        x_bottom = -(c+b*self.height)/a
        
        # 카메라 중앙 좌표
        camera_center = self.width/2

        # 노란선이 들어올 범위를 정할 threshold
        threshold_mid = 10

        # 삐뚤어졌지만 수직선에 가까워 위와 아래 교점이 범위 내에 들어간 직선
        if ((camera_center-threshold_mid)<x_bottom<(camera_center+threshold_mid)):
          if((camera_center-threshold_mid)<x_top<(camera_center+threshold_mid)):
          
            angle = 0
            self.state = self.detect

        # 수직선(b == 0, ax+c=0)이며 중앙범위 내에 들어감
        if (b == 0) and ((camera_center-threshold_mid)<x_top<(camera_center+self.threshold_mid)): 
          
          angle = 0
          self.state = self.detect  


        # 노란선의 밑변 교점을 320, 노란선의 두 교점의 x좌표 차이를 0으로 하게 pid

        current1_center = (-(c+b*self.height)/a) # 밑변에서의 노란 직선의 교점
        target1_center = 320
        error1_center = target1_center - current1_center
        derivative1_center = (error1_center - prev_error1_center) 



        if (b == 0): # 범위에 들지 않는 수직선
          current2_center = 0 # 직선의 윗프레임 교점과 아랫프레임 교점의 좌표 차이

        elif (b != 0):

        # -(a/b) 는 직선의 기울기

          current2_center = - self.height / (a/b) # 직선의 윗프레임 교점과 아랫프레임 교점의 좌표 차이


        target2_center = 0 # 윗변 교점과 아랫변 교점의 x 좌표 차이
        
        
        error2_center = target2_center - current2_center

        derivative2_center = (error1_center - prev_error2_center) 

        angle = Kp_center * error1_center + Kd_center * derivative1_center + Kpa_center * error2_center + Kda_center * derivative2_center
        
        
        prev_error1_center = error1_center
        prev_error2_center = error1_center




# -------------------------------------------------- detect state
# 분기 계산 state

    elif self.state == self.detect:
      angle = 0


      # 차량 두개 감지한 상황
      # 일단 지금 알고리즘은 두개 중 하나가 완전히 감지되지 않을 때까지 계속 주행하기로함
      # 따라서 line_change로 state를 돌리며 차량이 사라질 때까지 대기
      if len(self.yolo_list) == 2:
        self.state = self.line_change
          

      # 차량이 한 개 감지된 상황
      elif len(self.yolo_list) == 1:

        # 장애물 차량의 차선 계산
        obstacle_lane = self.find_closeObstacle_lane(image)
        

        if obstacle_lane == 2:        # 장애물이 2차선에 있다면 1차선으로 복귀
          self.state = self.return_1

        elif obstacle_lane == 1:      # 장애물이 1차선에 있다면 2차선으로 복귀
          self.state = self.return_2

        else:    # 예외처리
          pass
      else:      # 예외처리
        pass
      
# -------------------------------------------------- return_1 state
# 1차선 복귀 state / 검증필요

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
      # 왼쪽 흰 차선을 찾는 알고리즘 과정임, 더 나은 방법이 있다면 수정



      #pid 초안

      # 카메라 중앙 좌표와 두 직선의 교점의 평균 좌표가 같아지게 각도 pid
      target1_angle = 320
      current1_angle = ( (-(w_c + w_b * self.height) / w_a) + (-(y_c + y_b * self.height) / y_a) ) / 2

      # 양 차선 각도 계산
      if w_b==0:
        m_w=0
      else:
        m_w = -(w_a/w_b)

      if y_b==0:
        m_y=0
      else:
        m_y = -(y_a/y_b)

      # 흰차선의 각도(m_w) 와 노란차선의 각도(m_y) 의 합을 0으로 하게 각도 pid
      target2_angle = 0
      current2_angle = m_w + m_y

      error1_angle = target1_angle - current1_angle
      error2_angle = target2_angle - current2_angle
      derivative_angle = (error1_angle - prev_error_angle) 

      angle = Kp_angle * error1_angle + Kd_angle * derivative_angle - Ka_angle * error2_angle
      prev_error_angle = error1_angle

# -------------------------------------------------- return_2 state
# 2차선 복귀 state / 검증필요

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
      # 오른쪽 흰 차선을 찾는 알고리즘 과정임, 더 나은 방법이 있다면 수정




      #pid 초안

      # 카메라 중앙 좌표와 두 직선의 교점의 평균 좌표가 같아지게 각도 pid
      target1_angle = 320
      current1_angle = ( (-(w_c + w_b * self.height) / w_a) + (-(y_c + y_b * self.height) / y_a) ) / 2

      # 양 차선 각도 계산
      if w_b==0:
        m_w=0
      else:
        m_w = -(w_a/w_b)

      if y_b==0:
        m_y=0
      else:
        m_y = -(y_a/y_b)

      # 흰차선의 각도(m_w) 와 노란차선의 각도(m_y) 의 합을 0으로 하게 각도 pid
      target2_angle = 0
      current2_angle = m_w + m_y

      error1_angle = target1_angle - current1_angle
      error2_angle = target2_angle - current2_angle
      derivative_angle = (error1_angle - prev_error_angle) 

      angle = Kp_angle * error1_angle + Kd_angle * derivative_angle - Ka_angle * error2_angle
      prev_error_angle = error1_angle
    
# -------------------------------------------------- sidedrive_1 state

    elif self.state == self.sidedrive_1:
# 차선따라 각도 조향 코드
      if (초음파로 오른쪽차 미 감지):
        self.state = self.yello_follow


# -------------------------------------------------- sidedrive_2 state

    elif self.state == self.sidedrive_2:
# 차선따라 각도 조향 코드
      if (초음파로 왼쪽차 미 감지):
        self.state = self.yello_follow



# -----------------------------------------------------------------

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
  
  def Do_Test(self): # 앞차 속도 pid로 잘 추종하는지 test
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
    
    speed = int((self.min_Kp * error + self.min_Kd * derivative) * self.scaling_factor)
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

  def Do_Test2(self):  # 중앙 노란선 잘 추종하는지 test2
    
    image = self._image
    speed = 5

    if self.state == self.INIT:

      #중앙선 따라가는 pid
      Kp_center = 1
      Kd_center = 3 

      Kpa_center = 3 # 각도비례
      Kda_center = 3 # 각도비례

      prev_error1_center = 0
      prev_error2_center = 0

      derivative1_center = 0
      derivative2_center = 0

      error1_center = 0
      error2_center = 0


      self.height, self.width = image.shape[:2]
      angle = 0

      if (self.height>0) and (self.width>0):
        self.state = self.yello_follow



    elif self.state == self.yello_follow:

      
      a,b,c = self.find_yellow(image)


      # PD 제어 구현 ->  선 위치 맞추기, 선 각도 , 라이다나 옆차선 차량 유무 확인
      


      # 위치 맞추기 pid
      if a == 0:  # 노답 , x축에 평행한 직선임 / 다시 find yellow 부터
        angle = 0
        pass

      elif a != 0:

        if ((self.width/2-10)<(-(c+b*self.height)/a)<(self.width/2+10)) and ((self.width/2-10)<(-c/a)<(self.width/2+10)):    # 삐뚤어졌지만 범위 내에 들어간 직선
          angle = 0


        if (b == 0) and ((self.width/2-10)<(-c/a)<(self.width/2+10)): # 수직선이며 중앙범위 내에 들어감
          angle = 0



        current1_center = (-(c+b*self.height)/a) # 밑변에서의 노란 직선의 교점
        target1_center = 320
        error1_center = target1_center - current1_center
    
        derivative1_center = (error1_center - prev_error1_center) 



        if (b == 0): # 범위에 들지 않는 수직선
          current2_center = 0 # 직선의 윗프레임 교점과 아랫프레임 교점의 좌표 차이

        elif (b != 0):

        # -(a/b) 는 직선의 기울기

          current2_center = - self.height / (a/b)


        target2_center = 0 # 윗변 교점과 아랫변 교점의 x 좌표 차이
        
        
        error2_center = target2_center - current2_center

        derivative2_center = (error1_center - prev_error2_center) 

        angle = Kp_center * error1_center + Kd_center * derivative1_center + Kpa_center * error2_center + Kda_center * derivative2_center
        
        
        prev_error1_center = error1_center
        prev_error2_center = error1_center
      
      # 중앙선 출력
      cv2.line(image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)

      cv2.imshow("Yellow line", image)
      
      # 직선 a,b,c 값 출력
      print(a,b,c)

    
    
    return angle, speed

  def step(self):
    
    angle, speed = self.Do_Test2()
    #angle, speed = self.change_lane()




    # yolo_list = self._detect_car()
    # extra_car = len(yolo_list)
  
    # print(f"speed : {speed}")
    # print(f"angle : {angle}")
    # print(f"extra car : {extra_car}")
    
    return angle, speed
