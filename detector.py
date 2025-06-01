#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time  # 추가된 import
from enum import Enum
from ultralytics import YOLO
import os

model = YOLO(os.path.join(os.path.dirname(__file__), "detector.pt"))

class MissionType(Enum):
    IDLE = 0x01             # 최초 신호대기 state
    DRIVE = 0x02            # 차선 따라가기 state
    WAIT = 0x03             # 차선 주행 안정화를 위한 임시 state
    RUBBERCONE = 0x04       # 라바콘 피하기 state

class Detector:
    def __init__(self) -> None:
        self._image = None  # detect_mission 메소드에서 받아올 이미지 저장 변수
        self._ranges = None # detect_mission 메소드에서 받아올 라이다 데이터 저장 변수
        self._mission_state = MissionType.IDLE
        
        # WAIT 상태를 위한 시간 추적 변수들 추가
        self._wait_start_time = None
        self._wait_duration = 4.0  # 4초 직진
        
        # Detection 임계값 설정
        self.confidence_threshold = 0.5  # YOLO 신뢰도 임계값
        self.distance_threshold = 2.0    # 라이다 거리 임계값
        self.escape_point_threshold = 10 # 탈출 조건을 위한 라이다 포인트 개수 임계값
        
        # 클래스 ID 매핑 
        self.class_names = {
            'signal_GREEN': 0,
            'rubbercone': 1, 
            'signal_RED': 4,
            'signal_ORANGE': 5
        }

    def detect_mission(self, image, ranges) -> MissionType:
        self._image = image
        # 라이다 데이터를 numpy 배열로 변환
        self._ranges = np.array(ranges) if not isinstance(ranges, np.ndarray) else ranges
        # 카메라 이미지, 라이다 데이터 받아오기

        if self._mission_state == MissionType.IDLE:
            # 최초 신호 대기 상황
            if self.detect_green_light():
                # YOLO 모델로 판별 후 초록불 들어왔을 시
                self._mission_state = MissionType.WAIT
                self._wait_start_time = time.time()  # 대기 시작 시간 기록
                print("Green light detected! Transitioning to WAIT state")

        # ===================[ 위 조건문은 최초 시작 분기 ]===================

        elif self._mission_state == MissionType.WAIT:
            # 차선 주행 안정화를 위한 임시 WAIT 상태 진입
            if self._wait_start_time is None:
                self._wait_start_time = time.time()
                
            elapsed_time = time.time() - self._wait_start_time
            if elapsed_time < self._wait_duration:
                self._mission_state = MissionType.WAIT
                # 임시 상태 4초동안만 실행
            
            else:
                # 4초 경과 후 DRIVE 상태로 전환
                self._mission_state = MissionType.DRIVE
                self._wait_start_time = None  # 시간 변수 초기화
                print("WAIT completed! Transitioning to DRIVE state")

        elif self._mission_state == MissionType.DRIVE:
            # 차선 주행 상황
            if self.detect_rubbercone():
                # 라바콘 상태로 전환
                self._mission_state = MissionType.RUBBERCONE
                print("Rubbercone detected! Transitioning to RUBBERCONE state")
                
        elif self._mission_state == MissionType.RUBBERCONE:
            # state가 RUBBERCONE일 시 걸리는 분기 => 미션 탈출 분기
            if self.escape_rubbercone():
                # 탈출 조건 충족 시
                self._mission_state = MissionType.DRIVE
                print("Escaped rubbercone! Transitioning back to DRIVE state")

        return self._mission_state
        # 미션 종류 반환
    
    def get_control_values(self):
        """
        현재 상태에 따른 제어값 반환 (angle, speed)
        """
        if self._mission_state == MissionType.WAIT:
            return 0, 20  # 직진, 속도 20
        elif self._mission_state == MissionType.DRIVE:
            # 차선 추종 로직 
            return 0, 20  # 기본값
        elif self._mission_state == MissionType.RUBBERCONE:
            # 라바콘 회피 로직 
            return 0, 20   # 기본값
        else:  # IDLE
            return 0, 0   # 정지

    def detect_green_light(self) -> bool:
        """
        YOLO 모델로 초록불 감지
        """
        if self._image is None:
            return False
            
        try:
            # YOLO 모델로 추론 실행
            results = model(self._image)
            
            # 결과에서 초록불 탐지 확인
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 신뢰도 체크
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
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
        if self._image is None or self._ranges is None:
            return False
            
        try:
            # YOLO로 라바콘 탐지
            results = model(self._image)
            rubbercone_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        if (confidence > self.confidence_threshold and 
                            class_id == self.class_names['rubbercone']):
                            rubbercone_detected = True
                            print(f"Rubbercone detected with confidence: {confidence:.2f}")
                            break
            
            # 라바콘이 화면에 감지되고, 라이다 최솟값이 임계값보다 작을 때
            if rubbercone_detected and len(self._ranges) > 0:
                # 유효한 라이다 데이터만 필터링 (0보다 큰 값)
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
        if self._image is None or self._ranges is None:
            return False
            
        try:
            # YOLO로 라바콘 탐지 재확인
            results = model(self._image)
            rubbercone_detected = False
            
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
                valid_ranges = self._ranges[self._ranges > 0]  # 유효한 데이터만
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
    
    def get_current_state(self) -> MissionType:
        """현재 미션 상태 반환"""
        return self._mission_state
    
    def reset_state(self):
        """상태 초기화"""
        self._mission_state = MissionType.IDLE
        self._wait_start_time = None
        print("Detector state reset to IDLE")

