#!/usr/bin/env python
# -*- coding: utf-8 -*- 2
#=============================================
# 본 프로그램은 2025 제8회 국민대 자율주행 경진대회에서
# 예선과제를 수행하기 위한 파일입니다. 
# 예선과제 수행 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, rospy, time, os, math
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt


# 커스텀 모듈 import 시작 ======================
from detector import Detector, MissionType
from mode_drive import Drive
from mode_rubbercone import Rubbercone


# 커스텀 모듈 import 끝   ======================

# mode_drive 안정성을 위한 임시 state 정의 시작  ======================
class Wait:
    def __init__(self):
        pass
    
    def get_value(self, image, ranges):
        pass
    
    def step(self):
        return 0, 20  # angle=0, speed=10
# mode_drive 안정성을 위한 임시 state 정의 끝   ======================

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None  # 라이다 데이터를 담을 변수
motor = None  # 모터노드
motor_msg = XycarMotor()  # 모터 토픽 메시지
Fix_Speed = 20  # 모터 속도 고정 상수값 
new_angle = 0  # 모터 조향각 초기값
new_speed = Fix_Speed  # 모터 속도 초기값
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 

#=============================================
# 라이다 스캔정보로 그림을 그리기 위한 변수
#=============================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_aspect('equal')
lidar_points, = ax.plot([], [], 'bo')

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8") 
    
    # image 기본 인코딩 bgr
   

#=============================================
# 콜백함수 - 라이다 토픽을 받아서 처리하는 콜백함수
#=============================================
def lidar_callback(data):
    global ranges    
    ranges = data.ranges[0:360]
	
#=============================================
# 모터로 토픽을 발행하는 함수 
#=============================================
def drive(angle, speed):
    motor_msg.angle = float(angle)
    motor_msg.speed = float(speed)
    motor.publish(motor_msg)
             
#=============================================
# 실질적인 메인 함수 
#=============================================
def start():

    global motor, image, ranges
    
    print("Start program --------------")

    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    rospy.Subscriber("/usb_cam/image_raw", Image, usbcam_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
        
    #=========================================
    # 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Ready ----------")
    
    #=========================================
    # 라이다 스캔정보에 대한 시각화 준비를 합니다.
    #=========================================
    plt.ion()
    plt.show()
    print("Lidar Visualizer Ready ----------")
    
    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")


	
    #=========================================
    # 루프 전 프레임 관련 설정 
    #=========================================

    detector = Detector()
    # 미션 탐지 Detector 클래스 Instantiation 

    #=========================================
    # 메인 루프 
    #=========================================

    # Missions
    mission_mapping = {
        MissionType.IDLE: None,
        MissionType.DRIVE: Drive(),
        MissionType.WAIT: Wait(),
        MissionType.RUBBERCONE: Rubbercone(),   
    }

    detected_mission_prev = None
    # 미션 변경 여부 체크용 
    mission_changed = False
    # 미션 변경 여부 flag

    angle = 0
    speed = 0
    # IDLE state를 위한 초기화

	
    while not rospy.is_shutdown():

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", image)
        cv2.imshow("gray", gray)
        # 현재 들어오고 있는 카메라 이미지 출력 창 띄우기

        if ranges is not None:  
            angles = np.linspace(0,2*np.pi, len(ranges))+np.pi/2
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)

            lidar_points.set_data(x, y)
            fig.canvas.draw_idle()
            plt.pause(0.01)
        # 라이다 출력 창 띄우기

        # ==================[ 수정 시작 ]================== 

        detected_mission = detector.detect_mission(image, ranges)
        # Detector method의 detector instance의 detect_mission 메소드로 미션 탐지


        if(detected_mission != detected_mission_prev):
            mode_to_execute = mission_mapping[detected_mission]
            detected_mission_prev = detected_mission
        # 탐지된 미션이 바뀔 때만 주행 클래스를 Instantiation 하기 위한 조건문


        if mode_to_execute:
        # mode_to_excute가 None이 아닐 시 (=IDLE state가 아닐 시)
            
            mode_to_execute.get_value(image, ranges)
            # 센서 값 업데이트 
            angle, speed = mode_to_execute.step()
            # 조향각, 속도 결정


        drive(angle, speed)
        # 조향각, 속도로 실제 주행 수행

        # ==================[ 수정 끝  ]================== 
        time.sleep(0.1)
        
        cv2.waitKey(1)

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
