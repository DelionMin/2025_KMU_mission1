#!/usr/bin/env python3
import cv2
import rclpy
import numpy as np
import time 
from enum import Enum
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory

# YOLO 모델 불러오기 (detector.pt 파일은 본 파일과 같은 경로에 있어야 한다.)

model = YOLO("/home/xytron/xycar_ws/src/xycar_drive/xycar_drive/detector.pt")

'''
미션 상태 설명:
WAIT: 신호등 인식하여 파란불이 인식되면 RUBBERCONE State로 분기 
RUBBERCONE: RubberCone 코스 통과 후 RubberCone이 하나 이하로 보일 경우 LANE_DRIVE로 분기
LANE_DRIVE: 차선 인식 주행 및 추월 주행
'''



class MissionType(Enum):
  WAIT = 0x01
  RUBBERCONE = 0x02 
  LANE_DRIVE = 0x03
         
# track_drive.py 파일에서 import하여 활용할 미션 감지 클래스
class Detector:
  def __init__(self) -> None:
      
    # 카메라 및 라이다 데이터, 스테이트 초기화 
    self._image = None
    self._ranges = None

    # DEBUGGING < TESTING DESIGNATED MISSION>
    self._mission_state = MissionType.WAIT

    # CHANGE THIS VARIABLE TO DESIGNATE A PARTICULAR MISSION TO TEST
    # DEBUGGING < TESTING DESIGNATED MISSION>
    
    self.confidence_threshold = 0.6

    # 카메라 exposure 조절
    os.system('v4l2-ctl -d /dev/videoCAM -c auto_exposure=1')
    os.system(f'v4l2-ctl -d /dev/videoCAM -c exposure_time_absolute={110}')

    # 클래스 ID 매핑 (detector.pt 모델에 따라 조정 필요)
    self.class_names = {
      'signal_red': 0,
      'signal_yellow': 1,
      'signal_green': 2,
      'rubbercone': 3
    }

    # 디버깅 및 상태 추적용 변수
    self._detection_count = 0
    self._last_detection_time = time.time()

  
  def detect_mission(self, image, ranges) -> MissionType:
    """
    미션 감지 - 현재 상태에 따라 다음 상태로 전환 결정
    """  
    # 센서 데이터 받아오기
    self._image = image
    self._ranges = np.array(ranges) if not isinstance(ranges, np.ndarray) else ranges
    
    # 이미지가 없으면 현재 상태 유지
    if self._image is None:
        print("No image received! Maintaining current state")
        return self._mission_state

    # 현재 상태에 따른 처리
    if self._mission_state == MissionType.WAIT:
      if self._detect_green_light():
         self._mission_state = MissionType.RUBBERCONE
         print("Green light detected! Transitioning to RUBBERCONE STATE")  
      else:
         print("Green light not detected! Stay at WAIT STATE") 

    elif self._mission_state == MissionType.RUBBERCONE:
      rubbercone_number = self._detect_rubbercone()

      if rubbercone_number <= 1:
         self._mission_state = MissionType.LANE_DRIVE
         #  print(f"RUBBERCONE {rubbercone_number} detected! Transitioning to LANE_DRIVE STATE")
      else:
          #  print(f"RUBBERCONE {rubbercone_number} detected! Stay at RUBBERCONE STATE")
        pass
    elif self._mission_state == MissionType.LANE_DRIVE:
         print("Lane driving mode active")
         # LANE_DRIVE 상태에서는 계속 유지 (필요시 추가 로직 구현)

    return self._mission_state
  
  def get_current_state(self) -> MissionType:
    """
    현재 미션 상태 반환
    """
    return self._mission_state
  
  def set_state(self, new_state: MissionType):
    """
    강제로 상태 변경 (테스트 또는 특수 상황용)
    """
    old_state = self._mission_state
    self._mission_state = new_state
    print(f"State manually changed: {old_state} -> {new_state}")
  
  def _detect_green_light(self) -> bool:
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
                        class_id == self.class_names['signal_green']):
                        print(f"Green light detected with confidence: {confidence:.2f}")
                        return True
        
        return False
        
    except Exception as e:
        print(f"Error in _detect_green_light: {e}")
        return False
    
  def _detect_rubbercone(self) -> int:
    """
    YOLO 모델로 rubbercone 감지, rubbercone 갯수 반환
    """
    # 이미지가 없으면 0 반환
    if self._image is None:
        return 0
    
    rubbercone_count = 0
    detected_boxes = []  # 디버깅용 박스 정보 저장
    
    try:
        # YOLO 모델로 추론 실행
        results = model(self._image, verbose=False)
        
        # 결과에서 rubbercone 탐지 확인
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    
                    # 신뢰도 체크
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # 신뢰도 확보 시 rubbercone 카운트 증가
                    if (confidence > self.confidence_threshold and 
                        class_id == self.class_names['rubbercone']):
                        rubbercone_count += 1
                        
                        # 바운딩 박스 정보 저장 (디버깅용)
                        xyxy = box.xyxy[0].cpu().numpy()
                        detected_boxes.append({
                            'confidence': confidence,
                            'bbox': xyxy,
                            'center': [(xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2]
                        })
                        
                        #print(f"Rubbercone detected with confidence: {confidence:.2f}")
        
        # 디버깅 정보 출력
        if rubbercone_count > 0:
            #print(f"Total rubbercone count: {rubbercone_count}")
            # 추가 디버깅: 각 러버콘의 위치 정보
            for i, box_info in enumerate(detected_boxes):
                center = box_info['center']
                #print(f"  Rubbercone {i+1}: center=({center[0]:.1f}, {center[1]:.1f}), conf={box_info['confidence']:.2f}")
        
        return rubbercone_count
        
    except Exception as e:
        print(f"Error in _detect_rubbercone: {e}")
        return 0


  # ===================< 탈출 분기 확인 메소드 끝 >===================
