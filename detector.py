import cv2
import numpy as np
from enum import Enum

class MissionType(Enum):
    IDLE = 0x01             # 최초 신호대기 state
    DRIVE = 0x02            # 차선 따라가기 state
    RUBBERCONE = 0x04       # 라바콘 피하기 state
    BYPASS = 0x08           # 차량 피해가기 state

class Detector:
    def __init__(self) -> None:
        
        self._image = None  # detect_mission 메소드에서 받아올 이미지 저장 변수
        self._ranges = None # detect_mission 메소드에서 받아올 이미지 저장 변수

        self._mission_state = MissionType.IDLE 
        # Detector의 현재 state -> 최초 state는 신호대기 (IDLE)     
  
    def detect_mission(self, image, ranges) -> MissionType:

        self._image = image
        self._ranges = ranges
        # 카메라 이미지, 라이다 데이터 받아오기

        if self._mission_state == MissionType.IDLE:
        # 최초 신호 대기 상황
            if self.detect_green_light():
            # YOLO 모델로 판별 후 초록불 들어왔을 시
                self._mission_state = MissionType.DRIVE
                # state를 DRIVE로 천이

        # ===================[ 위 조건문은 최초 시작 분기 ]===================

        elif self._mission_state == MissionType.DRIVE: 
        # 차선 주행 상황
            if self.detect_rubbercone(): 
            # 라바콘 피하기 탐지 시
                self._mission_state = MissionType.RUBBERCONE
                # state를 RUBBERCONE으로 천이
            elif self.detect_bypass():
            # 차량 피해가기 탐지 시
                self._mission_state = MissionType.BYPASS
                # state를 BYPASS로 천이
        
        else:
        # state가 RUBBERCONE, BYPASS일 시 걸리는 분기 => 미션 탈출 분기 
            if self._mission_state == MissionType.RUBBERCONE \
            and self.escape_rubbercone():
            # state가 RUBBERCONE인 상태에서 탈출 조건 충족 시
                self._mission_state = MissionType.DRIVE
                # state를 DRIVE로 천이

            elif self._mission_state == MissionType.BYPASS \
            and self.escape_bypass():
            # state가 BYPASS인 상태에서 탈출 조건 충족 시
                self._mission_state = MissionType.DRIVE
                # state를 DRIVE로 천이

        return self._mission_state
        # 미션 종류 반환

    def detect_green_light(self) -> bool:
        # YOLO 모델 학습시킨 후 화면에 green light가 감지되었을 시 return True
        image = self._image
        ranges = self._ranges
        pass
        
    def detect_rubbercone(self) -> bool:
        # YOLO 모델 학습시킨 후 화면에 라바콘 인식 and 라이다 탐지 거리 중 최솟값이
        # Threshold 보다 작을 때 return True
        image = self._image
        ranges = self._ranges
        pass

    def detect_bypass(self) -> bool:
        # YOLO 모델 학습시킨 후 화면에 차량 인식 and 라이다 탐지 거리 중 최솟값이
        # Threshold 보다 작을 때 return True
        image = self._image
        ranges = self._ranges
        pass
    

    # ===================< 탈출 분기 확인 메소드 시작 >===================

    def escape_rubbercone(self) -> bool:
        # 라바콘이 더이상 감지되는 게 없을 때(최댓값이 아닌 측정값의 개수가 
        # Threshold 보다 작을 때 = 라이다에 걸린 점의 개수가 threshold보다 적을 때) True 반환
        image = self._image
        ranges = self._ranges
        pass

    def escape_bypass(self) -> bool:
        # 라이다 데이터를 봤을 때 더이상 차량과 충돌 여지가 없다는 것이 확인 될 경우 True 반환
        image = self._image
        ranges = self._ranges
        pass

    # ===================< 탈출 분기 확인 메소드 끝 >===================

    

    

    
