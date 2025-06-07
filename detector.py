#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time  # 추가된 import
from enum import Enum
from ultralytics import YOLO
import os

# YOLO 모델 불러오기 (detector.pt 파일은 본 파일과 같은 경로에 있어야 한다.)
model = YOLO(os.path.join(os.path.dirname(__file__), "detector.pt"))

# IDLE: 최초 신호대기 state
# DRIVE: 일반 도로 주행 state
# WAIT: 녹색등 점등 후 직진 state
# RUBBERCONE: 라바콘 회피 주행 state
class MissionType(Enum):
    IDLE = 0x01             
    DRIVE = 0x02            
    WAIT = 0x03             
    RUBBERCONE = 0x04       

# track_drive.py 파일에서 import하여 활용할 미션 감지 클래스
class Detector:
    def __init__(self) -> None:
        
        # 카메라 및 라이다 데이터, 스테이트 초기화 
        self._image = None
        self._ranges = None
        self._mission_state = MissionType.IDLE
        
        # WAIT 상태를 위한 시간 추적 변수 및 직진 지속 시간 정의
        self._wait_start_time = None
        self._wait_duration = 4.0
        
        # Detection 임계값 설정
        self.confidence_threshold = 0.5 
        self.distance_threshold = 2.0    
        self.escape_point_threshold = 10 
        
        # 클래스 ID 매핑 (detector.pt 모델에 따라 조정 필요)
        self.class_names = {
            'signal_GREEN': 0,
            'rubbercone': 1, 
            'signal_RED': 4,
            'signal_ORANGE': 5
        }

    
    def detect_mission(self, image, ranges) -> MissionType:
        """
        미션 감지
        """  
        # 센서 데이터 받아오기
        self._image = image
        self._ranges = np.array(ranges) if not isinstance(ranges, np.ndarray) else ranges

        # 최초 신호대기 state
        if self._mission_state == MissionType.IDLE:            
            if self.detect_green_light():
                self._mission_state = MissionType.WAIT
                self._wait_start_time = time.time()
                print("Green light detected! Transitioning to WAIT state")
         
        # 녹색등 점등 후 직진 state
        elif self._mission_state == MissionType.WAIT:
            if self._wait_start_time is None:
                self._wait_start_time = time.time()
                
            elapsed_time = time.time() - self._wait_start_time
            if elapsed_time < self._wait_duration:
                self._mission_state = MissionType.WAIT

            else:
                self._mission_state = MissionType.DRIVE
                self._wait_start_time = None
                print("WAIT completed! Transitioning to DRIVE state")
        
        # 일반 도로 주행 state
        elif self._mission_state == MissionType.DRIVE:
            if self.detect_rubbercone():
                self._mission_state = MissionType.RUBBERCONE
                print("Rubbercone detected! Transitioning to RUBBERCONE state")

        # 라바콘 회피 주행 state  
        elif self._mission_state == MissionType.RUBBERCONE:
            if self.escape_rubbercone():
                self._mission_state = MissionType.DRIVE
                print("Escaped rubbercone! Transitioning back to DRIVE state")

        return self._mission_state

    def detect_green_light(self) -> bool:
        """
        YOLO 모델로 초록불 감지
        """
        # 이미지 인식되지 않았을 시
        if self._image is None:
            return False
        
        # YOLO 모델로 추론 실행
        try:
            results = model(self._image, verbose=False)
            
            # 결과에서 초록불 탐지 확인
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        
                        # 신뢰도 체크
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # 신뢰도 확보 시 녹색등 점등 판단
                        if (confidence > self.confidence_threshold and 
                            class_id == self.class_names['signal_GREEN']):
                            print(f"Green light detected with confidence: {confidence:.2f}")
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error in detect_green_light: {e}")
            return False
    
    def detect_rubbercone(self) -> bool:
        """
        YOLO 모델로 라바콘 인식 + 라이다 거리 체크
        """
        # 카메라, 라이다 데이터 인식 실패 시
        if self._image is None or self._ranges is None:
            return False
            
        # YOLO로 라바콘 탐지
        try:
            results = model(self._image, verbose=False)
            rubbercone_detected = False
            
            # 결과에서 라바콘 탐지 확인
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:

                        # 신뢰도 체크
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # 신뢰도 확보 시 녹색등 점등 판단
                        if (confidence > self.confidence_threshold and 
                            class_id == self.class_names['rubbercone']):
                            rubbercone_detected = True
                            print(f"Rubbercone detected with confidence: {confidence:.2f}")
                            break
            
            # 라바콘이 화면에 감지되고, 라이다 최솟값이 임계값보다 작을 때 라바콘 미션으로 판단
            if rubbercone_detected and len(self._ranges) > 0:
                valid_ranges = self._ranges[self._ranges > 0]
                if len(valid_ranges) > 0:
                    min_range = np.min(valid_ranges)
                    if min_range < self.distance_threshold:
                        print(f"Rubbercone close! Min distance: {min_range:.2f}m")
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error in detect_rubbercone: {e}")
            return False

    # ===================< 탈출 분기 확인 메소드 시작 >===================
    
    def escape_rubbercone(self) -> bool:
        """
        라바콘 탈출 조건: 라바콘이 더이상 감지되지 않거나, 
        라이다에서 가까운 장애물 포인트가 임계값보다 적을 때
        """
        # 카메라, 라이다 데이터 인식 실패 시
        if self._image is None or self._ranges is None:
            return False
        
        # YOLO로 라바콘 탐지 재확인
        try:
            results = model(self._image, verbose=False)
            rubbercone_detected = False
            
            # 결과에서 라바콘 탐지 확인
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        if (confidence > self.confidence_threshold and 
                            class_id == self.class_names['rubbercone']):
                            rubbercone_detected = True
                            break
            
            # 라바콘이 더 이상 감지되지 않으면 탈출
            if not rubbercone_detected:
                print("No rubbercone detected - escaping")
                return True
            
            # 라이다에서 가까운 포인트 개수 체크
            if len(self._ranges) > 0:
                valid_ranges = self._ranges[self._ranges > 0]
                if len(valid_ranges) > 0:
                    close_points = np.sum(valid_ranges < self.distance_threshold)
                    if close_points < self.escape_point_threshold:
                        print(f"Few close points detected ({close_points}) - escaping rubbercone")
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error in escape_rubbercone: {e}")
            return False

    # ===================< 탈출 분기 확인 메소드 끝 >===================
