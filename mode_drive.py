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


class Drive:
    from enum import Enum

    class State(Enum):
        DRIVE_STATE_NONE = 0
        DRIVE_STATE_LEFT = 1
        DRIVE_STATE_RIGHT = 2

    def __init__(self) -> None:
        """
        state class를 초기화하는 함수. 관련 상수는 다 여기서 고치면 된다.
        """
        self._image = None
        self._ranges = None

        # ================== [ 영상 처리 관련 변수 시작 ] ==================


        # 카메라 이미지의 크기를 나타내는 상수
        self._camera_width = 640  # 높이 픽셀 수
        self._camera_height = 480  # 너비 픽셀 수
        # 연산량을 줄이기 위해 이미지를 줄였을 때의 크기를 나타내는 상수
        # 이때 원본 이미지 크기의 높이와 너비에 비해 각각 8배 작다.
        self._camera_width_s = self._camera_width // 8  # 높이 픽셀 수
        self._camera_height_s = self._camera_height // 8  # 너비 픽셀 수

        # perspective matrix 생성
        self._persptrans_matrix = self.__get_persptrans_matrix()

        # 흰색 픽셀을 검출할 때의 민감도 설정
        self._white_sensitivity = 30
        # base line ratio는 기준선을 화면의 어느 지점으로 정할지를 결정하는 비율이다.
        # 0.8이면, 이미지의 높이가 1일 때 0.2의 높이를 가지는 선이 기준선이 되는 것이다.
        self._base_line_ratio = 0.8
        self._base_line_position = int(self._camera_height_s * self._base_line_ratio)

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
        self._lane_width = 40
        # 차선 폭의 최솟값 설정
        self._lane_width_min = 12
        # 점수 임계치가 가지는 하한선 설정
        self._score_bound_min = -24

        ### 각도, 속도 관련 상수 ###
        self._p_con_p = 2
        self._speed = 5
        self._curve_decel = 5

        # ================== [ 영상 처리 관련 변수 끝 ] ==================

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

        self._change_to_the_right = True
        # __change_lane 에서 차로 우측으로 변경 / False 시 좌측으로 변경

        self._threshold_dist_middle_lane = 5
        # __change_lane 에서 차량이 중앙선으로부터 확보한 뒤 탈출하도록 하는 threshold



        # Logging
        self._enable_logging = True

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
        road_margin_x = 320
        road_margin_y = 280  # 소실점보다 아래에 있는 적당한 y값

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

        # 20250512 수정할 사항
        # image가 노란색 차선도 인식할 수 있도록 바꾸기 HSV 색 공부

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


        # 기준선(base line)위에서 차선이 존재하는 위치만 1이다.
        base_line_white = img_white[self._base_line_position, :]
        base_line_yellow = img_yellow[self._base_line_position, :]


        self._logging(img_original=img_white + img_yellow)
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

        # 차선 결정
        if detect_l:
            # 경우 1: 왼쪽과 오른쪽 차선 둘 다 검출됐을 때
            if detect_r:
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

        angle = self._p_con_p * self._pos
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

    def __change_lane(self, lane_yellow):
        '''
        라이다, YOLO로 차선 변경 분기 마련 후 flag 세우고 검증 필요
        '''

        # score는 가능한 가장 낮은 점수의 값으로 설정
        score_l, pos_l = -inf, None
        score_r, pos_r = -inf, None

        # positions 리스트(또는 np.array)를 순회하여 가장 점수가 큰 위치 결정
        for pos, valid in enumerate(lane_yellow):
        # pos -> position의 index , valid -> 내부 값
            if not valid:
                continue

            # 이전 차선의 위치와 차선 후보의 거리 차를 이용하여 점수를 낸다.
            score_l_cand = -abs(pos - self._lane_yellow_l)
            score_r_cand = -abs(pos - self._lane_yellow_r)

            # 점수가 가장 큰 차선 후보의 인덱스 번호를 저장한다.
            if score_l_cand > score_l:
                score_l = score_l_cand
                pos_l = pos
            if score_r_cand > score_r:
                score_r = score_r_cand
                pos_r = pos
        
                # score는 가능한 가장 낮은 점수의 값으로 설정
        
        if (not self._flag_angle_decided):
        # 최초 1회 실행용 flag (최초 조향각 결정 목적)
            if(pos_l):
            # 노란 차선 감지 시
                pos_yellow_lane = (pos_l + pos_r) // 2
                base_line_middle = self._base_line_data_length // 2

                angle = base_line_middle - pos_yellow_lane
                self._flag_angle_decided = True
            # 노란 차선 감지 '아직' 못 했을 시
            else:
                angle = 0

        else:
        # 조향각 결정 후
            if (pos_yellow_lane == base_line_middle):
            # 차량 중앙선 도달 시
                angle = 0
                # 직진
            elif (abs(pos_yellow_lane - base_line_middle) > self._threshold_dist_middle_lane):
            # 거리 확보된 후 
                self._flag_change_lane = False
                angle = 0

        return angle
         
    def step(self):
        image = self._image

        lane_white, lane_yellow = self.__detect_lane(image)

        '''
        라이다, YOLO로 차선 변경 분기 마련 후 flag 세우고 검증 필요
        '''
        if self._flag_change_lane:
        # _flag_change_lane는 라이다로 제일 가까운 값 threshold보다 작을 때 flag 올려버리자 
        # flag 내린 다음에 self.__change_lane_init() 돌려주고

            angle = self.__change_lane(lane_yellow)
        else:
            lane_group = lane_white + lane_yellow
            angle = self.__choose_lane(lane_group)

        '''
        라이다, YOLO로 차선 변경 분기 마련 후 flag 세우고 검증 필요
        '''
        
        speed = 10.0 # default
        # 나중에 값 바꿔서 감속 넣어도 돼

        return angle, speed
        
