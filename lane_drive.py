import time
import sys
import os
import signal
import numpy as np
import cv2
from math import inf

# 이 코드(class)는 step함수에서 다음과 같이 동작한다.
# 1. 이미지(카메라 데이터)와 class 내의 state를 기반으로 차선의 위치를 알아낸다.
#   1.1. 이미지를 항공뷰로 변환한다.
# 2. 차선의 위치를 기반으로 position을 구한다.
# 3. p제어를 하여 앞바퀴의 각도를 반환한다. 속도는 임의로 설정한다.

class LaneDrive:
  from enum import Enum
  
  # 차선 변경 주행 state
  class State_change(Enum):
    INIT = 0
    DRIVE_STATE_START = 1
    DRIVE_STATE_END = 2

  def __init__(self) -> None:
    
    # 카메라 exposure 조절
    os.system('v4l2-ctl -d /dev/videoCAM -c auto_exposure=1')
    os.system(f'v4l2-ctl -d /dev/videoCAM -c exposure_time_absolute={25}')


    self.height_frame = 480

    # 인식된 차선과의 교점(cursor)을 찾기 위한 프레임 상 가로선(baseline)의 y축 위치
    # 추후에 baseline_y_rate 로 바꾼 다음에 
    # int(height * baseline_y_rate) 로 변경해도 되고
    # TO BE TUNED
    self.baseline_y_rate = 0.75
    
    # 차선 일반적인 폭
    # TO BE TUNED
    # 1920 => 640 이니까 일단 실제 대회 차량에서는 80 정도..?
    self.lane_width = 240

    # 차선 인식 시 사용할 기울기 THRESHOLDS
    # TO BE TUNED
    self.THRESHOLD_SLOPE = 0.6
    self.THRESHOLD_SLOPE_YELLOW = 0.05

    # 가장 최근에 인식된 best_line (buffer)
    self.best_line_left_prev = None
    self.best_line_right_prev = None

    # 차선 색 상수
    self.COLOR_GREEN = (0, 255, 0)
    self.COLOR_RED = (0, 0, 255)


    # 인식된 차선과의 교점
    self.cursor_left = None
    self.cursor_right = None

    # PD 제어 목적 이전 angle_factor 저장용 변수
    self.angle_factor_prev = None
    self.angle_factor = None
    
    # PD 제어용 각 term에 대한 weight
    # TO BE TUNED
    self.K_P = 1
    self.K_D = 0

    # 픽셀 -> 조향각 scaling factor
    # TO BE TUNED
    self.p_con_p = 0.4
        
  def get_value(self, image, ranges):
    """
    센서 값 받아오기 위한 메소드
    """
    self._image = image
    self._ranges = ranges

  def region_of_interest(self, img, vertices):
    '''
    vertices 범위 밖 값들은 다 0으로 만들어 mask를 씌우는 함수
    '''
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
    # channel이 여러 개인 경우 (e.g. BGR은 channel 3개)
      channel_count = img.shape[2]
      ignore_mask_color = (255,) * channel_count
    else:
      ignore_mask_color = 255

    # mask에 vertices 모양대로 ignore_mask_color 값 넣음
    cv2.fillPoly(mask, [vertices], ignore_mask_color)

    # img, mask에 bitwise_and 연산 해서 겹치지 않는 부분은 0으로
    masked_image = cv2.bitwise_and(img, mask)
    
    # 어케 나오는지 궁금하면 이거 기
    # cv2.imshow("masked_image", masked_image)

    return masked_image

  def filter_lines(self, lines_unfiltered, slope_threshold):
    """
    <gets>
    lines_unfiltered: HoughlinesP로 감지된 선분들 
    slope: 차선 가능성 있는 선분들 걸러내기 위한 기울기 threshold
    <returns>
    left_lines: 왼쪽 차선 정보를 담은 선분들의 집합
    right_lines: 오른쪽 차선 정보를 담은 선분들의 집합
    """

    # 왼쪽, 오른쪽 차선 넣어둘 리스트
    left_lines = []
    right_lines = []

    if lines_unfiltered is not None:
      for line in lines_unfiltered:
        x1, y1, x2, y2 = line[0]

        # 수직선은 제외
        if x2 - x1 == 0:
          continue
        slope = (y2 - y1) / (x2 - x1)

        # 기울기가 작은 직선들은 제외
        if abs(slope) < slope_threshold:
          continue

        # 기울기 부호로 좌, 우측 판정
        # ***< 영상처리 시 Y값의 경우 아래로 갈수록 커진다는 점 유의 >***
        if slope < 0:
          left_lines.append(line)
        else:
          right_lines.append(line)

    return left_lines, right_lines




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
    
  def compute_intersection(self, line1, line2):
    """
    두 [[x1, y1, x2, y2]] 자료형으로 나타내어진 직선들의 교점을 선형대수학적 연산을 통해 계산한다.
    gets: [[x1, y1, x2, y2]] 자료형으로 나타내어진 직선 2개
    returns: int(x), int(y) (or None => 두 직선이 평행할 때)
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Line 1 coefficients
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    # Line 2 coefficients
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Determinant
    det = A1 * B2 - A2 * B1

    if det == 0:
      return None  # Lines are parallel or coincident

    # Cramer's Rule
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return int(x), int(y)



  def follow_lane(self):
    
    # angle 초기화
    angle = 0

    image = self._image
    # === 2. Convert to grayscale and apply Canny ===
    # 흑백 이미지 생성
    frame = image.copy()

    # =========================== < Generalize된 hough transform 전처리 > ===========================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 흑백 이미지에 GaussianBlur(=정규 분포 양상으로 픽셀 값 깎기, 흐릿하게 만들어서 영상처리에 자주 쓴다)
    # 노이즈 줄이고 선은 더 선명하게 보이도록
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # blur 이미지에 Canny 변환해서 Edge detection
    edges = cv2.Canny(blur, 50, 150)
    # =================================== < ROI focused 전처리 > ===================================

    # 이미지 hsv 형태로 변환
    hsv_yellow = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 노란색 범위 상, 하한선
    lower_yellow = np.array([20, 60, 60])
    upper_yellow = np.array([40, 255, 255])

    mask_yellow = cv2.inRange(hsv_yellow, lower_yellow, upper_yellow)
    yellow_parts = cv2.bitwise_and(image, image, mask=mask_yellow)

    gray_yellow = cv2.cvtColor(yellow_parts, cv2.COLOR_BGR2GRAY)
    blur_yellow = cv2.GaussianBlur(gray_yellow, (5, 5), 0)

    edges_yellow = cv2.Canny(blur_yellow, 50, 150)

    # <인수인계 및 DEBUGGING>
    # 감도 확인 후 lower_yellow, upper_yellow 수정
    # 2025.07.23 (1) -> 자율주행 스튜디오 유리문 앞 노이즈 심함, 그거 잡자
    cv2.imshow("gray_yellow", gray_yellow) 

    # ============================================================================================= 

    # 프레임 높이, 너비 저장
    # FYI) frame.shape => [(Height), (Width), (channel -> e.g. BGR)]
    height, width = image.shape[:2]

    baseline_y_pixel = int(height * self.baseline_y_rate)

    # 화면 상에 가로로 그은 직선 (차선과 교점 찾기 위한 프레임 기준 가로선)
    base_line = [[0, baseline_y_pixel, width, baseline_y_pixel]]

    # RoI(Region of Interest)의 좌표를 나타내는 np.array
    roi_vertices = np.array([[
        (width * 0, height),
        (width * 0, height * 0.6),
        (width * 1, height * 0.6),
        (width * 1, height)
    ]], dtype=np.int32)

    # 소실점 근처 노란색 점선 인식 위한 집중 ROI
    roi_vertices_focused = np.array([[
        (width * 0.25, height * 1.0),
        (width * 0.25, height * 0.6),
        (width * 0.75, height * 0.6),
        (width * 0.75, height * 1.0)
    ]], dtype=np.int32)

    # 사다리꼴을 image에 그리는 함수
    cv2.polylines(frame, roi_vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.polylines(frame, roi_vertices_focused, isClosed=True, color=(0, 255, 255), thickness=2)

    # Edge detection을 거친 이미지에 roi 마스킹
    roi_edges = self.region_of_interest(edges, roi_vertices)
    roi_yellow_edges = self.region_of_interest(edges_yellow, roi_vertices_focused)

    # roi_edges에서 직선 인식
    lines = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=10)

    # 인식된 선분 정제
    left_lines, right_lines = self.filter_lines(lines, self.THRESHOLD_SLOPE)

    # 노란 선분 인식
    lines_yellow = cv2.HoughLinesP(roi_yellow_edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=10)




    # DEBUGGING <HOUGH TRANSFORM YELLOW MONITOR>

    frame_hough = self._image.copy()

    if lines_yellow is not None:
      for line in lines_yellow:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame_hough, (x1, y1), (x2, y2), (255, 0, 255), 2)

    cv2.imshow("yellow hough", frame_hough)
    # DEBUGGING <HOUGH TRANSFORM YELLOW MONITOR>

    
    # DEBUGGING <HOUGH TRANSFORM WHITE MONITOR>

    lines_white = left_lines + right_lines

    if lines_white is not None:
      for line in lines_white:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame_hough, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imshow("white hough", frame_hough)
    # DEBUGGING <HOUGH TRANSFORM WHITE MONITOR>

    
    # 2025.07.23 (2) -> S자 코스 현재 차로 위치 파악해서 반대쪽 흰 차선 무시
    # 위 모니터 코드로 S자 코스에서 반대쪽 흰 차선 보는지 확인(거의 확신하긴 해)하고 해결
    # -> 가운데 중앙선 기울기로 현재 차로 파악하고 흰 차선 기울기 보고 거르면 될 듯, 소실점 앵글 생각해서
    # 현재 좌측 차선 -> 중앙선 기울기 양수 -> 좌측 레인 위치는 중앙선 X좌표 - self.lane_width
    # 현재 우측 차선 -> 중앙선 기울기 음수 -> 우측 레인 위치는 중앙선 X좌표 + self.lane_width

    # 2025.07.23 (3) -> 체커 극복
    # 후보 1. 허프 직선 길이 최소 조건 잡아서
    # 후보 1-1. 체커에 주황색 가로 baseline 점 갯수 센 다음에 checker detected 잡으면 체커 찍히는 앵글에서는
    # 조향각이 별로 안 클 거니까 그 때는 차선 인식하기 쉬울거, 그래서 허프 변환 길이 최소값 확 키워버리자
 


    # 노란 선분 정제
    left_lines_yellow, right_lines_yellow = self.filter_lines(lines_yellow, self.THRESHOLD_SLOPE_YELLOW)

    # 노란 선분 반영
    if left_lines is not None and left_lines_yellow is not None:
      left_lines = left_lines + left_lines_yellow
    if right_lines is not None and right_lines_yellow is not None:
      right_lines = right_lines + right_lines_yellow
    

    best_line_left = self.get_best_line(left_lines)
    best_line_right = self.get_best_line(right_lines)

    # best_line_left, best_line_right가 인식 X -> None일 때
    # _prev를 넣어서 intersection 연산
    # _prev로 line plot 한 경우에는 BGR (0, 0, 255) 빨간 선으로 표시해서 구분

    # 왼쪽 차선 인식 안 된 경우
    if best_line_left is None:
      best_line_left = self.best_line_left_prev
      lane_color_left = self.COLOR_RED
    # 왼쪽 차선 인식 된 경우
    else:
      self.best_line_left_prev = best_line_left
      lane_color_left = self.COLOR_GREEN
    
    # 오른쪽 차선 인식 안 된 경우
    if best_line_right is None:
      best_line_right = self.best_line_right_prev
      lane_color_right = self.COLOR_RED
    # 오른쪽 차선 인식 된 경우
    else:
      self.best_line_right_prev = best_line_right
      lane_color_right = self.COLOR_GREEN


    if best_line_left is not None:
      # 왼쪽 차선 표시
      x1, y1, x2, y2 = best_line_left[0]
      cv2.line(frame, (x1, y1), (x2, y2), lane_color_left, 2)

      # 왼쪽 차선과 baseline 교점 좌표 연산
      self.cursor_left = self.compute_intersection(best_line_left, base_line)


    if best_line_right is not None:
      # 오른쪽 차선 표시
      x1, y1, x2, y2 = best_line_right[0]
      cv2.line(frame, (x1, y1), (x2, y2), lane_color_right, 2)

      # 오른쪽 차선과 baseline 교점 좌표 연산
      self.cursor_right = self.compute_intersection(best_line_right, base_line)


    # ====================== 커서 초기 조건 안전성 설정 시작 ======================

    # 커서 인식 안되었을 시
    if self.cursor_left is None and self.cursor_right is None:
      pass

    elif self.cursor_left is None:
      self.cursor_left = np.array(self.cursor_right).copy()
      self.cursor_left[0] = self.cursor_right[0] + self.lane_width

    elif self.cursor_right is None:
      self.cursor_right = np.array(self.cursor_left).copy()
      self.cursor_right[0] = self.cursor_right[0] + self.lane_width

    # ====================== 커서 초기 조건 안전성 설정 끝 ======================

    if self.cursor_left is not None and self.cursor_right is not None:
    # 좌, 우측 커서가 모두 다 유효한 값을 가질 때
      
      # 커서 잇는 직선 그리고, 각 커서에 동그라미 마크 표시
      cv2.line(frame, self.cursor_left, self.cursor_right, (0, 0, 255), 2)
      cv2.circle(frame, self.cursor_left, 5, (255, 255, 0), 2)
      cv2.circle(frame, self.cursor_right, 5, (255, 255, 0), 2)

      # 각 커서 x 좌표 표시
      text_left = f"{self.cursor_left[0]}"
      text_right = f"{self.cursor_right[0]}"
      cv2.putText(frame, text_left, self.cursor_left, cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 255), 2, cv2.LINE_AA)
      cv2.putText(frame, text_right, self.cursor_right, cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 255), 2, cv2.LINE_AA)

      # 두 커서 좌표 중점
      # x값(angle_factor)을 조향에 활용 (실제 활용 시에는 프레임 width / 2 빼서)
      angle_factor_point = (np.array(self.cursor_right) + np.array(self.cursor_left)) // 2
      angle_factor = angle_factor_point[0]

      # 추후에 PD 제어 후 angle_factor 위치 표시할 변수
      angle_factor_PD_point = np.array(angle_factor_point).copy()

      # PD 제어 구현
      if self.angle_factor_prev is not None:
        # DEBUGGING
        print("PD WORKING")
        # DEBUGGING
        angle_factor_PD = int(self.K_P * angle_factor + self.K_D * (angle_factor - self.angle_factor_prev))

        # 버퍼 업데이트
        self.angle_factor_prev = angle_factor_PD

        # PD 제어 적용 후 Circle 표시할 위치
        angle_factor_PD_point[0] = angle_factor_PD

        # PD 제어 적용 후 텍스트 표시할 위치
        angle_factor_PD_point_text = angle_factor_PD_point.copy()
        angle_factor_PD_point_text[1] = angle_factor_PD_point_text[1] - 30
        
        # cv2.imshow에 텍스트 띄우기
        text_angle_factor_PD = f"angle_factor_PD: {angle_factor_PD}"
        cv2.putText(frame, text_angle_factor_PD, angle_factor_PD_point_text, cv2.FONT_HERSHEY_SIMPLEX,
          0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, angle_factor_PD_point, 5, (255, 255, 0), 2)
        
        # 조향각 연산
        angle = int((angle_factor_PD - (width // 2)) * self.p_con_p)

        # angle 출력 위치 정의
        angle_point = angle_factor_PD_point.copy()
        angle_point[1] = angle_point[1] + 15

        # 조향각 출력
        text_angle = f"angle: {angle}"
        cv2.putText(frame, text_angle, angle_point, cv2.FONT_HERSHEY_SIMPLEX,
          0.5, (0, 255, 255), 2, cv2.LINE_AA)
      
      else:
        # DEBUGGING
        print("PD_initiated")
        # DEBUGGING
        self.angle_factor_prev = angle_factor
        
    cv2.imshow("Lane Detection", frame)

    speed = 10

    return angle, speed

  def change_lane(self):
    pass

  def step(self):
    """
    메인 루프 내에서 Drive 클래스 인스턴스 활용시 실질적으로 작동시킬 메소드
    """

    angle, speed = self.follow_lane()


    return angle, speed
