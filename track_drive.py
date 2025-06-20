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

class Wait:
    def __init__(self):
        pass
    
    def get_value(self, image, ranges):
        pass
    
    def step(self):
        return 0, 45  # angle=0, speed=45

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
'''
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_aspect('equal')
lidar_points, = ax.plot([], [], 'bo')
'''
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

    # 미션 탐지 객체 Detector 클래스로 Instantiation 
    detector = Detector()

    #=========================================
    # 메인 루프 
    #=========================================

    # 미션, 클래스 맵핑 (스테이트 머신 구현)
    mission_mapping = {
        MissionType.IDLE: None,
        MissionType.DRIVE: Drive(),
        MissionType.WAIT: Wait(),
        MissionType.RUBBERCONE: Rubbercone(),   
    }

    # 미션 변경 여부 체크용 버퍼, flag
    detected_mission_prev = None 
    mission_changed = False

    # IDLE state를 위한 초기화
    angle = 0
    speed = 0

    # < 메인 루프 - 스테이트 머신>
    # - Detector 클래스의 detector 인스턴스 내 detect_mission 메소드로 미션 탐지
    # - 이전 미션 스테이트와 탐지된 미션 스테이트를 비교한 뒤 두 스테이트가 
    # 일치하지 않는 경우에만 주행 클래스를 Instantiate 한다.
    while not rospy.is_shutdown():

        detected_mission = detector.detect_mission(image, ranges)

        if(detected_mission != detected_mission_prev):
            mode_to_execute = mission_mapping[detected_mission]
            detected_mission_prev = detected_mission
       
        if mode_to_execute:
            
            mode_to_execute.get_value(image, ranges)
            angle, speed = mode_to_execute.step()

        drive(angle, speed)

        time.sleep(0.1)
        cv2.waitKey(1)

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
