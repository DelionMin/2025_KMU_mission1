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

    class State(Enum):
        DRIVE_STATE_NONE = 0
        DRIVE_STATE_LEFT = 1
        DRIVE_STATE_RIGHT = 2
    
    class State_change(Enum):
        INIT = 0
        DRIVE_STATE_START = 1
        DRIVE_STATE_STRAIGHT = 2
        DRIVE_STATE_END = 3

    class Angle_change(Enum):
        INIT = 0
        DRIVE_STATE_START = 3
        DRIVE_STATE_STRAIGHT = 0
        DRIVE_STATE_END = -10
        # START, END의 부호는 반대

    class Class_YOLO(Enum):
        CLASS_SIGNAL_GREEN = 0
        CLASS_RUBBERCONE = 1
        CLASS_CAR_YELLOW = 2
        CLASS_CAR_BLACK = 3
        CLASS_SIGNAL_RED = 4
        CLASS_SIGNAL_ORANGE = 5

    def __init__(self) -> None:
        """
        state class를 초기화하는 함수. 관련 상수는 다 여기서 고치면 된다.
        """
        self._image = None
        self._ranges = None

        self.yolo_model = YOLO(os.path.join(os.path.dirname(__file__), "detector.pt"))

        self.result_yolo = None


        # ================== [ 영상 처리 관련 변수 시작 ] ==================


        # 카메라 이미지의 크기를 나타내는 상수
        self._camera_width = 640  # 높이 픽셀 수
        self._camera_height = 480  # 너비 픽셀 수
        # 연산량을 줄이기 위해 이미지를 줄였을 때의 크기를 나타내는 상수
        # 이때 원본 이미지 크기의 높이와 너비에 비해 각각 8배 작다.
        self._camera_width_s = self._camera_width // 8  # 너비 픽셀 수 (=80)
        self._camera_height_s = self._camera_height // 8  # 높이 픽셀 수 (=60)

        # perspective matrix 생성
        self._persptrans_matrix = self.__get_persptrans_matrix()

        # 흰색 픽셀을 검출할 때의 민감도 설정
        self._white_sensitivity = 30
        # base line ratio는 기준선을 화면의 어느 지점으로 정할지를 결정하는 비율이다.
        # 0.8이면, 이미지의 높이가 1일 때 0.2의 높이를 가지는 선이 기준선이 되는 것이다.
        self._base_line_ratio = 0.8
        self._base_line_ratio_farther = 0.4
        self._base_line_position = int(self._camera_height_s * self._base_line_ratio)
        self._base_line_position_farther = int(self._camera_height_s * self._base_line_ratio_farther)

        ### 차선 판단 관련 상수 ###
        # base line정보의 길이 설정
        self._base_line_data_length = self._camera_width_s
        # 왼쪽 차선의 초기값 설정
        self._pos_l = 0
        # 오른쪽 차선의 초기값 설정
        self._pos_r = self._base_line_data_length - 1
        # position(차선의 가운데에서 벗어난 정도)초기값 설정
        self._pos = 0
        # 직진 구간 판단 범위 설정
        self._center_range = 10
        # 스테이트를 저장하는 변수의 초기값 설정
        self._state = self.State.DRIVE_STATE_NONE
        # 일반적인 차선의 폭 설정
        self._lane_width = 36
        # 차선 폭의 최솟값 설정
        self._lane_width_min = 12
        # 점수 임계치가 가지는 하한선 설정
        self._score_bound_min = -24

        ### 각도, 속도 관련 상수 ###
        self._p_con_p = 2
        self._speed = 5
        self._curve_decel = 5

        # ================== [ 영상 처리 관련 변수 끝 ] ==================
        

        # ================= ANGLE PID =================
        self._angle_prev = 0
        self._angle_i_term = 0

        self._dt = 0.1

        # DEBUGGING
        # I term은 제거해도 될 듯
        # DEBUGGING


        self._K_p = 0.1
        self._K_i = 0
        self._K_d = 0.9
        # ================= ANGLE PID =================

        self._yellow_sensitivity = 5
        # HSV 내 노란색 H값 범위

        self._flag_change_lane = False
        # 차선 변경 분기 플래그

        self._lane_yellow_l = 0
        self._lane_yellow_r = self._base_line_data_length - 1
        # 중앙선 pos 값 초기화

        
        self._flag_angle_decided = False
        self._change_direction_decided = True
        # __change_lane 에서 조향각 결정 flag (최초 1회 실행)
        # __change_lane 에서 차로 변경 방향 결정 flag (최초 1회 실행)

        self._change_to_the_left = True
        # __change_lane 에서 차로 우측으로 변경 / False 시 좌측으로 변경

        self._threshold_dist_middle_lane = 5
        # __change_lane 에서 차량이 중앙선으로부터 확보한 뒤 탈출하도록 하는 threshold

        self._state_change = self.State_change.INIT

        self._dist_car_pixel = 60
        # TO BE TUNED
        # 차선 변경 START state에서 카메라 앵글 중심점 x좌표와 차량 위치 비교용 threshold
   
        self._ROI_width_YOLO = 60
        # 차선 변경 알고리즘 내 YOLO ROI

        self._speed_change = 45
        self._speed_straight = 70
        # 차량 변경 속도

        self._ROI_car_detected = False
        # 차선 변경 시 ROI 내 차량 확인 여부 분기 


        self._car_detection_initialized = False
        # car_detection 가까이서 볼 수 있도록

        self._car_detection_size_threshold = 1000
        # car_detection 차량 이미지 크기 threshold


        self._change_angle_factor = 0.2

        self._target_cls_id = None
        # START_STATE에서 차량 종류 기억

        self.BYPASS_SAFETY_DISTANCE = 17.5
        # 추월 시 가속 안전거리

        self.angle_change_prev = 0
        # 차선 변경PD 제어





        self._lane_stabilized = True
        # 차선 변경 후 차선 추종 알고리즘 안정화 여부 flag

        self._stabilizing_factor = 0
        # 0일 시 그냥 삭제
        # 일반적으로 차선 한 쪽만 감지시 넣는 _lane_width에 안정화를 위해 더해주는 가중치

        self._stabilizer = self._lane_width + self._stabilizing_factor
        # 차선 변경 시 차선 추종 안정화를 위해 좌측 차선 측정 위치에 더해주는 조향 보조 상수

        self._grip_margin = 8
        # grip 차선 범위 margin

        self._bypass_initialized = False
        # bypass 찾기 시작

        self._target_lane_detected = False
        # 차선 변경 시 반대쪽 흰 차선 감지 여부

        self._grip_left_prev = 0
        self._grip_right_prev = self._camera_width_s
        # grip 차선 튐 방지


        self._bird_eye_view = None
        # Bird eye view로 차선 정렬 확인, _pos에 먹어주는 비례 상수 낮추기

        self._alignment_ok = False
        # 차선 정렬 여부 플래그

        self._flag_speed_straight = False
        # 속도 변경 플래그


        # ============================================================
        # pos를 저장할 큐 (크기 20)
        self._pos_history = deque(maxlen=20)
        
        # 곡률 계산 관련 상수
        self._curv_threshold_low = 0.01   # 직선 구간 판단 임계값
        # TO BE TUNED
        self._curv_threshold_high = 0.05  # 급커브 판단 임계값
        self._speed_straight = 45.0           # 직선 구간 속도
        self._speed_curve = 35.0              # 커브 구간 속도
        self._speed_sharp_curve = 30.0        # 급커브 구간 속도
        # ============================================================



        # Logging
        self._enable_logging = False

    def get_value(self, image, ranges):
        # 센서 값 받아오기 위한 메소드
        self._image = image
        self._ranges = ranges

    def __get_persptrans_matrix(self):
        
        # 20250512 준호형 팁
        # 밑에 사다리꼴 튜닝해서 길을 다 바라볼 수 있도록 -> 사다리꼴 모양 튜닝
        # 이 원인은 근데 흰 것만 인식하는데서 발생한 걸로 보임
        # 노란색 어떻게 쓸지 고민해보자
        

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
        Perspective Transform Matrix는 위와 같이 마름모 모양의 영역을 직사각형으로 펼치는 데에 쓰이는 Matrix이다.\n
        이때 좌표는 왼쪽에서 오른쪽으로 가면 x값 증가, 위에서 아래로 내려가면 y값이 증가한다.
        이때 마름모의 왼쪽 위의 좌표부터 순서대로 아래와 같다.\n
        (f_{소실점-왼쪽 아래 좌표 직선}(_road_margin_y),_road_margin_y), (f_{소실점-오른쪽 아래 좌표 직선}(_road_margin_y),_road_margin_y)\n
        (0 - _road_margin_x, _camera_hight), (_camera_width + _road_margin_x, _camera_hight)\n


        Returns:
            Any: Perspective Transform Matrix
        """

        # 이미지에서 도로의 소실점 y값. 이때 y값은 아래로 가면서 증가한다.
        road_vanish_y = 240
        # 아래는 _get_persptrans_matrix 함수의 설명 참고.
        road_margin_x = 400
        road_margin_y = 300  # 소실점보다 아래에 있는 적당한 y값

        # 마름모 왼쪽 변 기울기를 구한다.(dy/dx)가 기울기가 된다.
        dx = self._camera_width / 2 + road_margin_x
        dy = self._camera_height - road_vanish_y

        # 마름모 왼쪽 변 직선과 y =_road_margin_y 인 직선의 교점의 x 좌표를 구한다.
        xr = (dx / dy) * (road_margin_y - road_vanish_y) + (self._camera_width / 2)
        # 마름모는 이미지의 중심 세로선에 대칭이므로
        # 아래와 같이 오른쪽 변 직선과 y =_road_margin_y 의 교점의 x좌표를 구한다.
        xl = self._camera_width - xr

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

        # perspective transform : 항공뷰 구하기
        img = cv2.warpPerspective(
            image,
            self._persptrans_matrix,
            (self._camera_width, self._camera_height),
        )

        # resize : 이미지 크기 줄이기 - 연산량 감소를 위해
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

        '''
        일단 img_white_yellow 만들어 보고 logging 한 다음에 baseline 찍어내보자
        H -> 0 ~ 360 (R: 0, G: 120, B: 240)
        S -> 0 ~ 100
        V -> 0 ~ 100

        그러나! opencv에서는
        H -> 0 ~ 179 (R: 0, G: 59, B: 119)
        S -> 0 ~ 255
        V -> 0 ~ 255
        '''
        # 위 연산을 노란색 픽셀들에 대해서도 수행한다.
        lower_yellow = np.array([29 - self._yellow_sensitivity, 0, 200])
        upper_yellow = np.array([29 + self._yellow_sensitivity, 255, 255])
        img_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)


        base_line_white = img_white[self._base_line_position, :]
        base_line_yellow = img_yellow[self._base_line_position, :]


        self._logging(img_original=img_white + img_yellow)

        self._bird_eye_view = img_white + img_yellow


        # 일단 잡힌 차선들 모두 다 출력

        return base_line_white, base_line_yellow

    def _logging(self, img_original):
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
        차선의 중앙에서 얼마나 떨어져있는지(position)를 바탕으로 조향각(angle)을 반환하는 함수이다.\n

        이 함수는 y=-abs(curr_x - prev_x) 휴리스틱 함수를 이용하여 검출된 차선 중
        이전에 결정된 차선에 가장 가까운 차선의 위치를 고른다.\n

        초기에는 먼 거리에 존재하는 차선도 인식시키기 위해(차량이 보고 있는 차선은 매번 다르므로)
        하한선 개념을 도입하여 차선이 검출될 때마다 하한선을 조금식 올리며 점차 그 간극을 줄여나간다. \n
        """
        # score는 가능한 가장 낮은 점수의 값으로 설정
        score_l, pos_l = -inf, None
        score_r, pos_r = -inf, None


        flag_angle_stabilized = False
        # stabilizer 탈출 분기


        # positions 리스트(또는 np.array)를 순회하여 가장 점수가 큰 위치 결정
        for pos, valid in enumerate(positions):
        # pos -> position의 index , valid -> 내부 값
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

        _flag_both_lanes_detected = False
        # stabilized 판단 시 양 쪽 차선 모두 관측 되는지 여부 flag

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

        # Update Lane Information
        self._pos_l = pos_l
        self._pos_r = pos_r
        # get position
        self._pos = pos_l + pos_r - self._base_line_data_length
        
        
        # Update State
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


        if not self._lane_stabilized:
            # 차선 변경 시 감도 하락 해결 대책
            angle = 0
            pos_lane_array = np.where(positions == 255)

            if pos_lane_array[0].size > 0:

                if self._change_to_the_left:
                    # 좌측 차선으로 차선 변경
                    grip_left_array = pos_lane_array[0]

                    if grip_left_array.size > 0:
                        grip_left = min(grip_left_array)

                        if abs(grip_left - self._grip_left_prev) > (self._camera_width_s // 4):
                        # 좌측 차선 날라가고 추측 차선으로 뛸 때 (threshold 튜닝 필요)
                        # DEBUGGING <각주 말 닦기>
                            angle = 5 

                        self._grip_left_prev = grip_left
                        self._pos = grip_left + grip_left + self._stabilizer - self._base_line_data_length
                        # DEBUGGING
                        # <self._pos 업데이트 하도록 수정해봄>
                        # DEBUGGING


                        angle = self._p_con_p * (self._pos)    
                        
                        # COMMENTED RECORDING
                        # print("grip_left: ", grip_left)
                        # print("grip_right_approx: ", grip_left + self._stabilizer)
                        # print("grip left angle: ", angle)                    
                else:
                    # 우측 차선으로 차선 변경 
                    grip_right_array = pos_lane_array[0]

                    if grip_right_array.size > 0:
                        grip_right = max(grip_right_array)

                        if abs(grip_right - self._grip_right_prev) > (self._camera_width_s // 4):
                        # 좌측 차선 날라가고 추측 차선으로 뛸 때 (threshold 튜닝 필요)
                        # DEBUGGING <각주 말 닦기>
                            angle = 5 

                        self._grip_right_prev = grip_right
                        self._pos = grip_right + grip_right - self._stabilizer - self._base_line_data_length
                        # DEBUGGING
                        # <self._pos 업데이트 하도록 수정해봄>
                        # DEBUGGING

                        angle = self._p_con_p * (self._pos) 

                        # COMMENTED RECORDING
                        # print("grip right angle: ", angle)
            

            # < 분기 조건 후보 2 >
            if not self._alignment_ok and not self._lane_stabilized and self.__lanes_aligned() :
                self._alignment_ok = True
            
            if self._alignment_ok:
                if angle > 2:
                    angle = 2
                else:
                    angle / 2

                if angle < 0.1:
                    self._lane_stabilized = True
                    self._flag_speed_straight = True
                    
                    # COMMENTED RECORDING
                    # print("LANE STABILIZED")
                    
            '''
            # if(self._state == self.State.DRIVE_STATE_NONE and _flag_both_lanes_detected):
            # < 분기 조건 후보 1 -> 잘 동작은 X > 
                print("LANE STABILIZED")
                angle = 0
                self._angle_prev = 0
                self._lane_stabilized = True
            '''
            
        else:
            angle = self._p_con_p * self._pos

        # ===================== 곡률 계산용 deque append =====================
        self._pos_history.append(self._pos)
        # ===================== 곡률 계산용 deque append =====================

        if self._lane_stabilized:

            # PID IMPLEMENTATION TRIAL

            self._angle_i_term = self._angle_i_term + (angle * self._dt)
            angle_d_term = angle - self._angle_prev 

            # COMMENTED RECORDING
            # print("angle_before_PD: ", angle)

            angle = self._K_p * angle \
                    + self._K_i * self._angle_i_term \
                    + self._K_d * angle_d_term
            
            # COMMENTED RECORDING
            # print("angle_after_PD: ", angle)

            # DEBUGGING <여기 아니면 return 직전>
            #self._angle_prev = angle
            # DEBUGGING
            
            # PID IMPLEMENTATION TRIAL

        ranges = self._ranges
        left_ranges = ranges[0:89]
        min_left_ranges = min(left_ranges)
        
        right_ranges = ranges[270:355]
        min_right_ranges = min(right_ranges)

        if min_left_ranges < 1 or min_right_ranges < 1:
        # 차선 변경 시 안전 거리 확보 목적
            print("CAUTION! ABOUT TO CRASH")

            if min_left_ranges < 1:
                angle = 20
                print("TO THE RIGHT")
            else:
                angle = -20   
                print("TO THE LEFT")       
        
        # DEBUGGING <여기 아니면 return 직전>
        self._angle_prev = angle
        # DEBUGGING <여기 아니면 return 직전>

        return angle       


    def __choose_lane_init(self):
        # 차선 변경 절차 수행 후 다시 레인을 따라가기 위한 연산

        # 왼쪽 차선의 초기값 설정
        self._pos_l = 0
        # 오른쪽 차선의 초기값 설정
        self._pos_r = self._base_line_data_length - 1
        # position(차선의 가운데에서 벗어난 정도)초기값 설정
        self._pos = 0
        # 점수 임계치가 가지는 하한선 설정
        self._score_bound_min = -24
        # state
        self._state = self.State.DRIVE_STATE_NONE

    def __change_lane(self, lane_white, lane_yellow):
        
        # < State 예시 >
        # self.State_change.INIT
        # self.State_change.DRIVE_STATE_START 
        # self.State_change.DRIVE_STATE_STRAIGHT
        # self.State_change.DRIVE_STATE_END

        # lane_white는 상시 인식
        # lane_yellow는 점선때문에 인식 안 될때는 일단 대기
        # lane_white, lane_yellow에 np.where 후 평균 때려서 차선 위치 index 추정
        # 차선 위치 index 크기 비교해서 현재 좌, 우측 차선 위치 추정
        # 현재 차량 위치(baseline 위치랑 lane_yellow index 위치 차이 값으로 초기 조향값 설정)
        # 탈출 분기가 고민
        
        # 초안: 위에서 구한 lane_yellow랑 baseline 위치 차이가 threshold 넘기면 
        # 1. 분기 탈출
        # 2. self.__choose_lane_init()

        ranges = self._ranges

        if (self._state_change == self.State_change.INIT):

            pos_white_array = np.where(lane_white == 255)

            if pos_white_array[0].size > 0:
                # pos_white_array이 비지 않았을 때
                lane_white_mean = np.array(pos_white_array).mean()
                pos_current = self._base_line_data_length / 2
                # 현재 차량 위치 (프레임 상)         

                if (lane_white_mean < pos_current):
                    self._change_to_the_left = False
                    # COMMENTED RECORDING
                    # print("change to the right")
                    # 차량 변경 방향 판정
                else:
                    self._change_to_the_left = True
                    # COMMENTED RECORDING
                    # print("change to the left")
            
                self._state_change = self.State_change.DRIVE_STATE_START

            speed = self._speed_change
            # speed 초기화
            angle = 0
            # angle 초기화

        elif(self._state_change == self.State_change.DRIVE_STATE_START):
            
            speed = self._speed_change
            angle = self.Angle_change.DRIVE_STATE_START.value
            # speed 바꿀거면 어케저케.. step도 수정하고 해야된다

            if self._change_to_the_left:
                ranges_right = ranges[270:359]
                min_ranges_right = min(ranges_right)
                if min_ranges_right > 20:
                    min_ranges_right = 20
                angle = 5 + (20 - min_ranges_right) / 2
                # COMMENTED RECORDING
                # print("changing to the left angle: ", -1 * angle)
            else:
                ranges_left = ranges[0:90]
                min_ranges_left = min(ranges_left)
                if min_ranges_left > 20:
                    min_ranges_left = 20
                angle = 5 + (20 - min_ranges_left) / 2
                # COMMENTED RECORDING
                # print("changing to the right angle: ", angle)

            pos_white_array = np.where(lane_white == 255)

            if pos_white_array[0].size > 0:
                
                if self._change_to_the_left:
                    
                    # 왼쪽 구역에 차선 감지 된 거 있으면 탈출 flag 하나 세우기
                    # 오른쪽 차선만 보는 거 방지 안전 장치

                    pos_white_left = min(pos_white_array[0])

                    if(pos_white_left < (self._camera_width_s // 2)):
                        self._target_lane_detected = True
                    
                    if (pos_white_left > 0 and self._target_lane_detected):
                        self._state_change = self.State_change.DRIVE_STATE_END
                        self._target_lane_detected = False

                else:
                    pos_white_right = max(pos_white_array[0])

                    if(pos_white_right > (self._camera_width_s // 2)):
                        self._target_lane_detected = True

                    if (pos_white_right < self._camera_width_s and self._target_lane_detected):
                        self._state_change = self.State_change.DRIVE_STATE_END
                        self._target_lane_detected = False

                    # State 천이

        elif(self._state_change == self.State_change.DRIVE_STATE_END):

            self._flag_change_lane = False
            self._lane_stabilized = False
            # 차선 변경 후 차선 추종 알고리즘 안정화 분기 플래그
            self._state_change = self.State_change.INIT
            self._alignment_ok = False

            angle = 0
            speed = self._speed_change
            # 초기화

        if (self._change_to_the_left):
            angle = angle * (-1)
        # 차선 변경 방향 맞춰주기
        
        self._angle_prev = angle

        return angle, speed
    
    def _YOLO_step(self):
        image = self._image
        self.result_yolo = self.yolo_model(image, verbose=False)
        image_YOLO_plot = self.result_yolo[0].plot()
        
        # cv2.imshow("YOLO", image_YOLO_plot)
        # YOLO 이미지 띄우기

    def car_detected(self):
        result_yolo = self.result_yolo
        ranges = self._ranges

        LIDAR_SAFETY_DISTANCE = self.BYPASS_SAFETY_DISTANCE
        if (not self._car_detection_initialized):
            LIDAR_SAFETY_DISTANCE = 10
            # 최초 차선 변경 지점 도입 안전거리
            # DEBUGGING <굳이 flag 해야 하나?>

        _flag_car_detected = False
        # YOLO 결과 차량이 인식되었는지 flag

        ranges_ROI = np.concatenate([ranges[0:5], ranges[355:360]])

        min_ROI = np.min(ranges_ROI)

        if result_yolo:
            # YOLO 값이 들어왔을 때
            for box in result_yolo[0].boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                center_x = (x1 + x2) / 2
                car_size = abs(x1 - x2) * abs(y1 - y2)
                
                if ((cls_id == self.Class_YOLO.CLASS_CAR_BLACK.value) or (cls_id == self.Class_YOLO.CLASS_CAR_YELLOW.value)):
                    # 노란 차, 검은 차 인식 여부 확인
                    if ((self._camera_width // 2 - self._ROI_width_YOLO) < center_x < (self._camera_width // 2 + self._ROI_width_YOLO) \
                    and (car_size > self._car_detection_size_threshold)):
                        _flag_car_detected = True
                    
        if (min_ROI < LIDAR_SAFETY_DISTANCE) and (_flag_car_detected):
            if (not self._car_detection_initialized):
                self._car_detection_initialized = True

            return True
        else:
            return False

    def __calculate_curv(self):
        """
        pos 큐를 기반으로 도로의 곡률을 계산하는 함수
        Returns:
            float: 계산된 곡률 값 (0에 가까울수록 직선)
        """
        if len(self._pos_history) < 5:
            return 0.0
        
        # 최근 position 값들을 numpy 배열로 변환
        positions = np.array(list(self._pos_history))
        
        # 시간 축 생성 (프레임 단위)
        time_points = np.arange(len(positions))
        
        # 2차 다항식 피팅을 통한 곡률 계산
        try:
            # 2차 다항식 계수 구하기: y = ax^2 + bx + c
            coeffs = np.polyfit(time_points, positions, 2)
            a = coeffs[0]  # 2차 계수
            
            # 곡률은 2차 계수의 절댓값으로 근사
            curv = abs(a) * 2
            
            return curv
        except:
            return 0.0

    def __adjust_speed_by_curv(self):

        curv = self.__calculate_curv()
        
        if curv < self._curv_threshold_low:
            # 직선 구간
            return self._speed_straight
        elif curv < self._curv_threshold_high:
            # 일반 커브 구간
            return self._speed_curve
        else:
            # 급커브 구간
            return self._speed_sharp_curve

    def __lanes_aligned(self):
        # 차선 직선 판정
        img = self._bird_eye_view

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        edges = binary.copy()  # Use raw binary as edges if it's clean

        # Run Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10,
                                minLineLength=10, maxLineGap=5)

        img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
                # Visualize lines (optional)
                cv2.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Check vertical alignment (e.g., angle near ±90)
            vertical_lines = [a for a in angles if abs(a) > 80]
            ratio = len(vertical_lines) / len(angles)
            if ratio > 0.8:
                # COMMENTED RECORDING
                # print("Lanes are mostly vertical — vehicle is likely aligned.")

                # Show the result (if needed)
                # cv2.imshow("Detected Lines", img_color)
                return True
            else:
                # COMMENTED RECORDING
                # print("Lanes are not vertical enough — possible misalignment.")
                return False
        else:
            # COMMENTED RECORDING
            # print("No lines detected — check input or lane visibility.")
            return False

    def step(self):
        image = self._image
        self._YOLO_step()

        lane_white, lane_yellow = self.__detect_lane(image)

        '''
        라이다, YOLO로 차선 변경 분기 마련 후 flag 세우고 검증 필요
        '''

        # 나중에 값 바꿔서 감속 넣어도 돼

        if self.car_detected() and self._lane_stabilized and not (self._flag_change_lane):
            self._flag_change_lane = True
            self._flag_speed_straight = False

        if self._flag_change_lane:
        # _flag_change_lane는 라이다로 제일 가까운 값 threshold보다 작을 때 flag 올려버리자 
        # flag 내린 다음에 self.__change_lane_init() 돌려주고
            angle, speed = self.__change_lane(lane_white, lane_yellow)
           
        else:
            lane_group = lane_white + lane_yellow
            angle = self.__choose_lane(lane_group)

            if not self._lane_stabilized:
                speed = self._speed_change
            else:
                if self._flag_speed_straight:
                    speed = self._speed_straight

                else:
                    speed = self.__adjust_speed_by_curv()
        
        # COMMENTED FOR RECORDING
        # print("FINAL angle: ", angle)
        # print("FINAL speed: ", speed)

        return angle, speed
