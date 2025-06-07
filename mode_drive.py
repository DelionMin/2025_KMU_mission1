import time
import sys
import os
import signal
import numpy as np
import cv2
from math import inf
from ultralytics import YOLO
from collections import deque

# 이 코드(class)는 step함수에서 다음과 같이 동작한다.
# 1. 이미지(카메라 데이터)와 class 내의 state를 기반으로 차선의 위치를 알아낸다.
#   1.1. 이미지를 항공뷰로 변환한다.
# 2. 차선의 위치를 기반으로 position을 구한다.
# 3. p제어를 하여 앞바퀴의 각도를 반환한다. 속도는 임의로 설정한다.


class Drive:
    from enum import Enum
    
    # 차선 추종 주행 state 
    class State(Enum):
        DRIVE_STATE_NONE = 0
        DRIVE_STATE_LEFT = 1
        DRIVE_STATE_RIGHT = 2
    
    # 차선 변경 주행 state
    class State_change(Enum):
        INIT = 0
        DRIVE_STATE_START = 1
        DRIVE_STATE_END = 2

    # YOLO 클래스
    class Class_YOLO(Enum):
        CLASS_SIGNAL_GREEN = 0
        CLASS_RUBBERCONE = 1
        CLASS_CAR_YELLOW = 2
        CLASS_CAR_BLACK = 3
        CLASS_SIGNAL_RED = 4
        CLASS_SIGNAL_ORANGE = 5

    def __init__(self) -> None:
        
        # 카메라, 라이다 데이터 초기화
        self._image = None
        self._ranges = None

        # YOLO 관련 모델 정의 및 추론 결과 초기화
        self.yolo_model = YOLO(os.path.join(os.path.dirname(__file__), "detector.pt"))
        self.result_yolo = None


        # ================== [ 영상 처리 관련 변수 시작 ] ==================


        # 카메라 이미지의 크기를 픽셀 단위로 나타내는 상수
        self._camera_width = 640  
        self._camera_height = 480 

        # 연산량을 줄이기 위해 이미지 크기를 줄였을 때의 차원을 나타내는 상수
        # 원본 이미지 크기의 높이와 너비에 비해 각각 8배 작다.
        # 640X480 => 80X60
        self._camera_width_s = self._camera_width // 8 
        self._camera_height_s = self._camera_height // 8  

        # perspective matrix 생성
        self._persptrans_matrix = self.__get_persptrans_matrix()

        # 흰색, 노란색 픽셀을 검출할 때의 민감도 설정
        self._white_sensitivity = 30
        self._yellow_sensitivity = 5

        ### Base line 관련 상수 ###
        # base line ratio는 기준선을 화면의 어느 지점으로 정할지를 결정하는 비율이다.
        # 0.8이면, 이미지의 높이가 1일 때 (프레임 좌측 하단 꼭짓점 기준) 0.2의 높이를 가지는 선이 기준선이 되는 것이다.
        self._base_line_ratio = 0.8
        self._base_line_position = int(self._camera_height_s * self._base_line_ratio)
        self._base_line_data_length = self._camera_width_s

        # ================== [ 영상 처리 관련 변수 끝 ] ==================


         # ================== [ 차선 추종 관련 변수 시작 ] ==================
        ### 차선 내 위치 관련 상수 ###
        self._pos_l = 0
        self._pos_r = self._base_line_data_length - 1
        self._pos = 0

        ### 차선 판단 관련 상수 ###
        self._center_range = 10
        self._state = self.State.DRIVE_STATE_NONE
        self._lane_width = 36
        self._lane_width_min = 12
        self._score_bound_min = -24

        ### (차선 위 차량 위치 => 조향각) 전환 비례 상수 ###
        self._p_con_p = 2

        # ================== [ 차선 추종 관련 변수 끝 ] ==================
        

        # ================== [ PD 제어 관련 변수 시작 ] ==================
        # PD 제어 목적 각도 값 버퍼
        self._angle_prev = 0

        # PD 제어 Gain 정의
        self._K_p = 0.1
        self._K_d = 0.9
        # ==================  [ PD 제어 관련 변수 끝 ]  ==================


        
        # ================== [ 차선 변경 관련 변수 시작 ] ==================

        # 차선 변경 분기, 변경 방향 플래그
        self._flag_change_lane = False
        self._change_to_the_left = True
        
        # 차선 변경 절차 내 State 초기화
        self._state_change = self.State_change.INIT

        # 차선 변경 알고리즘 내 YOLO ROI
        self._ROI_width_YOLO = 60

        # 차선 변경 속도, 조향각 기본값
        self._speed_change = 45
        self._angle_change_init = 3
        self._angle_change_offset = 5
       
        # car_detection 관련 변수
        self._car_detection_initialized = False
        self._car_detection_size_threshold = 1000

        # 추월 안전거리
        self.BYPASS_SAFETY_DISTANCE = 17.5
        
        # 추월 후 차선 안정화 관련 변수
        self._lane_stabilized = True
        self._stabilizer = self._lane_width
        self._grip_left_prev = 0
        self._grip_right_prev = self._camera_width_s
        self._target_lane_detected = False
        self._flag_speed_straight = False
        
        # 차선 정렬 여부 판단 관련 변수
        self._bird_eye_view = None
        self._alignment_ok = False

        # ================== [ 차선 변경 관련 변수 끝 ] ==================




        # ================== [ 곡률 계산 관련 변수 시작 ] ==================
        self._pos_history = deque(maxlen=20)
        self._curv_threshold_low = 0.01
        self._curv_threshold_high = 0.05
        self._speed_straight = 45.0
        self._speed_curve = 37.5
        self._speed_sharp_curve = 30.0
        # ================== [ 곡률 계산 관련 변수 끝 ] ==================

        # 차선 인식 Logging 토글
        self._enable_logging = True

    def get_value(self, image, ranges):
        """
        센서 값 받아오기 위한 메소드
        """
        self._image = image
        self._ranges = ranges

    def __get_persptrans_matrix(self):
        """
        cv2.warpPerspective() 함수에 사용할 Perspective Transform Matrix를 반환하는 함수.\n
        이를 통해 도로의 항공뷰를 만들 수 있다.
        ```
             * <-소실점!
           _____           ________
         /       \        |        |
        /         \   ->  |        |
        -----------       ----------
        ```
        Perspective Transform Matrix는 위와 같이 마름모 모양의 영역을 직사각형으로 펼치는 데에 쓰이는 Matrix이다.
        이때 좌표는 왼쪽에서 오른쪽으로 가면 x값 증가, 위에서 아래로 내려가면 y값이 증가한다.
        이때 마름모의 왼쪽 위의 좌표부터 순서대로 아래와 같다.
        (f_{소실점-왼쪽 아래 좌표 직선}(_road_margin_y),_road_margin_y), (f_{소실점-오른쪽 아래 좌표 직선}(_road_margin_y),_road_margin_y)
        (0 - _road_margin_x, _camera_hight), (_camera_width + _road_margin_x, _camera_hight)

        Returns:
            Any: Perspective Transform Matrix
        """

        # 시점 변경 관련 상수
        road_vanish_y = 240
        road_margin_x = 400
        road_margin_y = 300  

        # 마름모 왼쪽 변 기울기 (dy/dx)
        dx = self._camera_width / 2 + road_margin_x
        dy = self._camera_height - road_vanish_y

        # xr: 마름모 왼쪽 변 직선과 y =_road_margin_y 인 직선의 교점의 x 좌표를 구한다.
        # xl: 마름모는 이미지의 중심 세로선에 대칭이므로 오른쪽 변 직선과 y =_road_margin_y 의 교점의 x좌표를 구한다.
        xr = (dx / dy) * (road_margin_y - road_vanish_y) + (self._camera_width / 2)
        xl = self._camera_width - xr

        # 
        src = np.array(
            [
                (xl, road_margin_y),
                (xr, road_margin_y),
                (-road_margin_x, self._camera_height),
                (self._camera_width + road_margin_x, self._camera_height),
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [
                (0, 0),
                (self._camera_width, 0),
                (0, self._camera_height),
                (self._camera_width, self._camera_height),
            ],
            dtype=np.float32,
        )

        return cv2.getPerspectiveTransform(src, dst)

    def __detect_lane(self, image):
        """
        이 함수는 base line(기준선)에 감지된 lane으로 추정되는 흰색 픽셀들과 노란색 픽셀들들의 위치를 반환하는 함수이다.\n
        예를 들어 base line의 데이터 길이가 20이라면, 아래와 같은 출력이 나오는 것이다.\n
        -> 01000,00010,00000,10000
        """

        # 항공뷰 구하기 (perspective transform)
        img = cv2.warpPerspective(
            image,
            self._persptrans_matrix,
            (self._camera_width, self._camera_height),
        )

        # 이미지 크기 줄이기 - 연산량 감소 목적 (resize)
        img_small = cv2.resize(
            img,
            dsize=(self._camera_width_s, self._camera_height_s),
            interpolation=cv2.INTER_NEAREST,
        )

        # cvtColor : 색공간 변환(BGR -> HSV)
        img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        # inRange: 차선 색만 검출. 이때 차선 색은 흰색이라 가정한다.
        # 따라서 흰색 픽셀만 검출한다.
        lower_white = np.array([0, 0, 255 - self._white_sensitivity])
        upper_white = np.array([255, self._white_sensitivity, 255])
        img_white = cv2.inRange(img_hsv, lower_white, upper_white)

        # 위 연산을 노란색 픽셀들에 대해서도 수행한다.
        lower_yellow = np.array([29 - self._yellow_sensitivity, 0, 200])
        upper_yellow = np.array([29 + self._yellow_sensitivity, 255, 255])
        img_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        # 항공뷰에서 추출한 흰색, 노란색 이미지의 한 row 만을 추출하여 관찰한다.
        base_line_white = img_white[self._base_line_position, :]
        base_line_yellow = img_yellow[self._base_line_position, :]

        # 차선 추종 관찰 모니터 함수
        self._logging(img_original=img_white + img_yellow)

        # 항공뷰 저장 (추후 차선 정렬 확인용)
        self._bird_eye_view = img_white + img_yellow

        return base_line_white, base_line_yellow

    def _logging(self, img_original):
        """
        차선 추종 관찰 모니터 함수
        """
        # logging 토글 구현
        if not self._enable_logging:
            return

        # lane 정보를 기준선과 함께 화면에 띄운다.
        img_lane = img_original.copy()  #
        img_lane = cv2.cvtColor(img_lane, cv2.COLOR_GRAY2BGR)

        img_lane = cv2.line(
        img_lane,
        (0, self._base_line_position),
        (self._camera_width_s - 1, self._base_line_position),
        (0, 255, 0),
        1,
        )

        img_lane = cv2.line(
            img_lane,
            (self._pos_l, self._base_line_position),
            (self._pos_l, self._base_line_position),
            (0, 0, 255),
            1,
        )
        img_lane = cv2.line(
            img_lane,
            (self._pos_r, self._base_line_position),
            (self._pos_r, self._base_line_position),
            (255, 0, 0),
            1,
        )
        img_lane = cv2.resize(
            img_lane,
            dsize=(self._camera_width, self._camera_height),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Lane Image", img_lane)
        cv2.waitKey(1)

    def __choose_lane(self, positions):
        """
        lane으로 추정될 수 있는 데이터들을 가져와, 실제 lane은 무엇인지 판단하고
        차선의 중앙에서 얼마나 떨어져있는지(position)를 바탕으로 조향각(angle)을 반환하는 함수이다.

        이 함수는 y=-abs(curr_x - prev_x) 휴리스틱 함수를 이용하여 검출된 차선 중
        이전에 결정된 차선에 가장 가까운 차선의 위치를 고른다.

        초기에는 먼 거리에 존재하는 차선도 인식시키기 위해(차량이 보고 있는 차선은 매번 다르므로)
        하한선 개념을 도입하여 차선이 검출될 때마다 하한선을 조금식 올리며 점차 그 간극을 줄여나간다.
        """
        # score는 가능한 가장 낮은 점수의 값으로 설정
        score_l, pos_l = -inf, None
        score_r, pos_r = -inf, None

        # stabilizer 탈출 분기
        flag_angle_stabilized = False

        # stabilized 판단 시 양 쪽 차선 모두 관측 되는지 여부 flag
        _flag_both_lanes_detected = False

        # positions 리스트(또는 np.array)를 순회하여 가장 점수가 큰 위치 결정
        for pos, valid in enumerate(positions):
            if not valid:
                continue

            # 이전 차선의 위치와 차선 후보의 거리 차를 이용하여 점수를 낸다.
            score_l_cand = -abs(pos - self._pos_l)
            score_r_cand = -abs(pos - self._pos_r)

            # 점수가 가장 큰 차선 후보의 인덱스 번호를 저장한다.
            if score_l_cand > score_l:
                score_l = score_l_cand
                pos_l = pos
            if score_r_cand > score_r:
                score_r = score_r_cand
                pos_r = pos

        # score가 유효한 값인지 확인: 너무 작은 score는 무시한다
        detect_l = score_l > self._score_bound_min
        detect_r = score_r > self._score_bound_min

        # 차선 결정
        if detect_l:
            # 경우 1: 왼쪽과 오른쪽 차선 둘 다 검출됐을 때
            if detect_r:
                _flag_both_lanes_detected = True
                # 두 차선이 너무 가깝거나, 역전됐을 때
                if (pos_r - pos_l) < self._lane_width_min:
                    if score_l > score_r:
                        pos_r = pos_l + self._lane_width_min
                    else:
                        pos_l = pos_r - self._lane_width_min
            # 경우 2: 왼쪽 차선만 검출됐을 때 (보통 우회전인 경우)
            else:
                # 현재 상태가 왼쪽일 때 - 유효하지 않은 값
                if self._state == self.State.DRIVE_STATE_LEFT:
                    pos_l = self._pos_l
                pos_r = pos_l + self._lane_width
        else:
            # 경우 3: 오른쪽 차선만 검출됐을 때 (보통 좌회전인 경우)
            if detect_r:
                # 현재 상태가 오른쪽일 때 - 유효하지 않은 값
                if self._state == self.State.DRIVE_STATE_RIGHT:
                    pos_r = self._pos_r
                pos_l = pos_r - self._lane_width
            # 경우 4: 두 차선 모두 다 검출되지 않았을 때
            else:
                # 이전 차선 정보를 사용한다
                pos_l = self._pos_l
                pos_r = self._pos_r

        # 차선 정보 업데이트
        self._pos_l = pos_l
        self._pos_r = pos_r
        # 차량 위치 업데이트
        self._pos = pos_l + pos_r - self._base_line_data_length
        
        
        # State 업데이트
        if self._pos < -self._center_range:
            self._state = self.State.DRIVE_STATE_LEFT
        elif self._pos > self._center_range:
            self._state = self.State.DRIVE_STATE_RIGHT
        else:
            self._state = self.State.DRIVE_STATE_NONE

        if self._state == self.State.DRIVE_STATE_NONE:
            self._score_bound_min = -8
        else:
            self._score_bound_min = -24


        # 차선 변경 시 차량 위치 안정화
        if not self._lane_stabilized:
            angle = 0
            pos_lane_array = np.where(positions == 255)

            if pos_lane_array[0].size > 0:
                
                # 좌측 차선으로 차선 변경 시
                if self._change_to_the_left:
                    
                    grip_left_array = pos_lane_array[0]

                    if grip_left_array.size > 0:
                        grip_left = min(grip_left_array)

                        # 관측되던 차선이 더 이상 관측되지 않는 경우
                        if abs(grip_left - self._grip_left_prev) > (self._camera_width_s // 4):
                            angle = 5 

                        self._grip_left_prev = grip_left
                        self._pos = grip_left + grip_left + self._stabilizer - self._base_line_data_length

                        angle = self._p_con_p * (self._pos)                  
                
                # 좌측 차선으로 차선 변경 시
                else:
                    grip_right_array = pos_lane_array[0]

                    if grip_right_array.size > 0:
                        grip_right = max(grip_right_array)

                        # 관측되던 차선이 더 이상 관측되지 않는 경우
                        if abs(grip_right - self._grip_right_prev) > (self._camera_width_s // 4):
                            angle = 5 

                        self._grip_right_prev = grip_right
                        self._pos = grip_right + grip_right - self._stabilizer - self._base_line_data_length

                        angle = self._p_con_p * (self._pos) 
            
            # 차선 순간 정렬 확인 분기
            if not self._alignment_ok and not self._lane_stabilized and self.__lanes_aligned() :
                self._alignment_ok = True
            
            # 차선 순간 정렬 확인 시 조향각 수렴
            if self._alignment_ok:
                if angle > 2:
                    angle = 2
                else:
                    angle / 2

                if angle < 0.1:
                    self._lane_stabilized = True
                    self._flag_speed_straight = True

        # 조향각 (비례 상수 * 차량위치, 차선 정중앙 간 오차)        
        else:
            angle = self._p_con_p * self._pos

        # 곡률 계산용 deque append
        self._pos_history.append(self._pos)

        # 차선 정렬 확보된 경우
        if self._lane_stabilized:

            # PD 제어 구현
            angle_d_term = angle - self._angle_prev 

            angle = self._K_p * angle \
                    + self._K_d * angle_d_term

        # 안전 거리 알고리즘
        ranges = self._ranges
        left_ranges = ranges[0:89]
        min_left_ranges = min(left_ranges)
        
        right_ranges = ranges[270:355]
        min_right_ranges = min(right_ranges)

        # 안전 거리 비상 조향
        if min_left_ranges < 1 or min_right_ranges < 1:
            print("CAUTION! ABOUT TO CRASH")

            if min_left_ranges < 1:
                angle = 20
                print("TO THE RIGHT")
            else:
                angle = -20   
                print("TO THE LEFT")       
        
        # PD 제어용 버퍼
        self._angle_prev = angle

        return angle       


    def __change_lane(self, lane_white, lane_yellow):
        """
        이 함수는 차선 정보 lane_white, lane_yellow를 입력 받아 State 기반으로 angle, speed를 반환하여 실질적으로
        차선 변경을 구현하는 함수이다.

        1. INIT state
        차량이 차선 내에 정상적으로 위치할 때 한 쪽은 노란색 차선, 나머지 한 쪽은 흰색 차선을 탐지한다는 점을 바탕으로
        차선 변경 방향을 결정한다.

        2. DRIVE_STATE_START state
        실질적인 차선 변경 주행을 시작한다. 라이다 측정 거리를 기반으로 조향각의 크기를 연산해 기존 차선 위 앞 차량과
        거리가 가까울 수록 더 큰 조향각으로 주행한다. 

        3. DRIVE_STATE_END state
        차선 변경 주행을 마무리 한 뒤 State machine을 초기화하고 차선 정렬 flag를 False로 바꿔 차선 추종 주행에서
        차선을 안정적으로 찾아갈 수 있도록 한다.
        """

        # 라이다 데이터 불러오기
        ranges = self._ranges

        # INIT state
        if (self._state_change == self.State_change.INIT):

            pos_white_array = np.where(lane_white == 255)

            if pos_white_array[0].size > 0:
                lane_white_mean = np.array(pos_white_array).mean()
                pos_current = self._base_line_data_length / 2         

                # 흰색 차선이 차량 왼쪽에 위치할 경우 => 우측 차선으로 차선 변경 판단
                if (lane_white_mean < pos_current):
                    self._change_to_the_left = False
                # 흰색 차선이 차량 오른쪽에 위치할 경우 => 좌측 차선으로 차선 변경 판단
                else:
                    self._change_to_the_left = True

                # State 천이
                self._state_change = self.State_change.DRIVE_STATE_START

            # speed, angle 초기화
            speed = self._speed_change
            angle = 0

        # DRIVE_STATE_START state
        elif(self._state_change == self.State_change.DRIVE_STATE_START):
            
            # speed, angle 초기화
            speed = self._speed_change
            angle = self._angle_change_init

            # 좌측 차선으로 변경
            if self._change_to_the_left:
                ranges_right = ranges[270:359]
                min_ranges_right = min(ranges_right)
                if min_ranges_right > 20:
                    min_ranges_right = 20
                angle = self._angle_change_offset + (20 - min_ranges_right) / 2
            
            # 우측 차선으로 변경
            else:
                ranges_left = ranges[0:90]
                min_ranges_left = min(ranges_left)
                if min_ranges_left > 20:
                    min_ranges_left = 20
                angle = self._angle_change_offset + (20 - min_ranges_left) / 2

            # ================= [ 차선 변경 대상 차선 탐지 알고리즘 시작 ] =================
            pos_white_array = np.where(lane_white == 255)

            if pos_white_array[0].size > 0:
                
                # 좌측 차선으로 변경
                if self._change_to_the_left:

                    pos_white_left = min(pos_white_array[0])

                    if(pos_white_left < (self._camera_width_s // 2)):
                        self._target_lane_detected = True
                    
                    if (pos_white_left > 0 and self._target_lane_detected):
                        # State 천이
                        self._state_change = self.State_change.DRIVE_STATE_END
                        self._target_lane_detected = False

                # 우측 차선으로 변경
                else:
                    pos_white_right = max(pos_white_array[0])

                    if(pos_white_right > (self._camera_width_s // 2)):
                        self._target_lane_detected = True

                    if (pos_white_right < self._camera_width_s and self._target_lane_detected):
                        # State 천이
                        self._state_change = self.State_change.DRIVE_STATE_END
                        self._target_lane_detected = False
            # ================= [ 차선 변경 대상 차선 탐지 알고리즘 끝 ] =================

        # DRIVE_STATE_END state
        elif(self._state_change == self.State_change.DRIVE_STATE_END):
            
            # State machine 초기화
            self._flag_change_lane = False
            self._lane_stabilized = False
            self._state_change = self.State_change.INIT
            self._alignment_ok = False

            # angle, speed 초기화
            angle = 0
            speed = self._speed_change
            
        # 차선 변경 방향 맞춰주기
        if (self._change_to_the_left):
            angle = angle * (-1)
        
        # PD 제어용 버퍼
        self._angle_prev = angle

        return angle, speed
    

    def _YOLO_step(self):
        """
        이 함수는 YOLO 모델에 이미지를 입력하여 객체 인식 추론을 수행하는 함수이다.
        """
        image = self._image
        self.result_yolo = self.yolo_model(image, verbose=False)
        

    def car_detected(self):
        """
        이 함수는 EGO 차량 전방에 추월의 대상이 되는 차량이 (1. 카메라 ROI 내에 있는지), (2. 거리가 충분히 가까운지)
        인식한 뒤 추월 주행 시작 시점을 결정하는 함수이다.
        """
        # YOLO 결과, 라이다 데이터 불러오기
        result_yolo = self.result_yolo
        ranges = self._ranges

        LIDAR_SAFETY_DISTANCE = self.BYPASS_SAFETY_DISTANCE
        if (not self._car_detection_initialized):
            LIDAR_SAFETY_DISTANCE = 10

        _flag_car_detected = False

        ranges_ROI = np.concatenate([ranges[0:5], ranges[355:360]])

        min_ROI = np.min(ranges_ROI)

        if result_yolo:
            for box in result_yolo[0].boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                center_x = (x1 + x2) / 2
                car_size = abs(x1 - x2) * abs(y1 - y2)
                
                # 노란 차, 검은 차 인식 여부 확인
                if ((cls_id == self.Class_YOLO.CLASS_CAR_BLACK.value) or (cls_id == self.Class_YOLO.CLASS_CAR_YELLOW.value)):
                    
                    # 1. 인식된 차량 객체 위치가 ROI 내에 있을 시
                    # 2. 인식된 차량의 크기가 적당히 클 시
                    # => 위 두 조건 모두 충족 시 차량 인식 판정
                    if ((self._camera_width // 2 - self._ROI_width_YOLO) < center_x < (self._camera_width // 2 + self._ROI_width_YOLO) \
                    and (car_size > self._car_detection_size_threshold)):
                        _flag_car_detected = True

        # 1. 전방 차량 거리가 충분히 가까울 시
        # 2. 전방 차량이 인식되었을 시
        # => 위 두 조건 모두 충족 시 True 반환
        if (min_ROI < LIDAR_SAFETY_DISTANCE) and (_flag_car_detected):
            if (not self._car_detection_initialized):
                self._car_detection_initialized = True
            return True

        else:
            return False

    def __calculate_curv(self):
        """
        이 함수는 pos 큐를 기반으로 도로의 곡률을 계산하여 반환하는 함수이다.
        """
        # 큐 크기가 5개 미만일 시 곡률 0으로 판정
        if len(self._pos_history) < 5:
            return 0.0
        
        # 최근 position 값들을 numpy 배열로 변환
        positions = np.array(list(self._pos_history))
        time_points = np.arange(len(positions))
        
        # 2차 다항식 피팅을 통한 곡률 계산
        # 2차 다항식 계수 구하기: y = ax^2 + bx + c
        try:
            coeffs = np.polyfit(time_points, positions, 2)
            a = coeffs[0]  
            curv = abs(a) * 2
            
            # 계산된 곡률 값 (0에 가까울 수록 직선)
            return curv
        except:
            return 0.0

    def __adjust_speed_by_curv(self):
        """
        이 함수는 앞서 연산한 곡률을 바탕으로 차량의 속력을 결정하여 반환하는 함수이다.
        """

        # 곡률 계산
        curv = self.__calculate_curv()
        
        # 직선 구간
        if curv < self._curv_threshold_low:
            return self._speed_straight
        # 일반 커브 구간
        elif curv < self._curv_threshold_high:
            return self._speed_curve
         # 급커브 구간
        else:
            return self._speed_sharp_curve

    def __lanes_aligned(self):
        """
        이 함수는 차선 항공뷰 이미지를 바탕으로 Hough 변환을 통해 직선을 탐지함으로써 차선 정렬 여부를 반환하는 함수이다.
        """

        img = self._bird_eye_view
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        edges = binary.copy() 

        # Hough 변환 수행
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10,
                                minLineLength=10, maxLineGap=5)

        img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Hough 변환으로 인식한 차선 기울기 값을 바탕으로 정렬 여부 판정
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
                cv2.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 1)

            vertical_lines = [a for a in angles if abs(a) > 80]
            ratio = len(vertical_lines) / len(angles)
            
            # 정렬 판정
            if ratio > 0.8:
                return True
           # 정렬 실패 판정
            else:
                return False
        # 차선 인식 실패 판정
        else:
            return False

    def step(self):
        """
        메인 루프 내에서 Drive 클래스 인스턴스 활용시 실질적으로 작동시킬 메소드
        """
        
        image = self._image
        self._YOLO_step()

        lane_white, lane_yellow = self.__detect_lane(image)

        # 추월 분기 판정
        if self.car_detected() and self._lane_stabilized and not (self._flag_change_lane):
            self._flag_change_lane = True
            self._flag_speed_straight = False

        # 추월 분기 활성화 시
        if self._flag_change_lane:
            angle, speed = self.__change_lane(lane_white, lane_yellow)
        
        # 추월 분기 비활성화 시 (=일반 차선 추종 주행 시)
        else:
            lane_group = lane_white + lane_yellow
            
            # 차선 추종 조향각 연산
            angle = self.__choose_lane(lane_group)

            # 차선 변경 후 정렬 미완료 시
            if not self._lane_stabilized:
                speed = self._speed_change
            # 차선 변경 후 정렬 완료 시
            else:
                # 직선 주행시
                if self._flag_speed_straight:
                    speed = self._speed_straight

                # 곡선 주행시
                else:
                    speed = self.__adjust_speed_by_curv()
        
        print("======================")
        print("FINAL angle: ", angle)
        print("FINAL speed: ", speed)

        return angle, speed
