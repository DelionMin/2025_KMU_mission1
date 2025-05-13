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
import matplotlib.pyplot as plt
import apriltag


#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================

image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None                # 라이다 데이터를 담을 변수
motor = None                 # 모터노드
motor_msg = XycarMotor()     # 모터 토픽 메시지
Fix_Speed = 10               # 모터 속도 고정 상수값 
new_angle = 0                # 모터 조향각 초기값
new_speed = Fix_Speed        # 모터 속도 초기값
bridge = CvBridge()          # OpenCV 함수를 사용하기 위한 브릿지 


#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
#=============================================

def usbcam_callback(data):

    global image

    image = bridge.imgmsg_to_cv2(data, "bgr8") 

    

    # image 기본 인코딩 bgr

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
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)

    #=========================================
    # 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다.
    #=========================================

    rospy.wait_for_message("/usb_cam/image_raw", Image)
    print("Camera Ready --------------")
    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")

    #=========================================
    # 메인 루프 
    #=========================================
    # ✅ Apriltag Detector 초기화
    options = apriltag.DetectorOptions(families="tag36h11")  # 또는 "tag36h11"
    detector = apriltag.Detector(options)


    # 초기화

    case = 1
    parallel = True
    distance = 1000
    k=160
    center_y=5000

    while not rospy.is_shutdown():
        if image.size == 0:
            continue

        # 흑백 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ✅ AprilTag 검출
        detections = detector.detect(gray)

        # ✅ 검출된 태그 시각화
        for det in detections:
            corners = det.corners.astype(int)
            for i in range(4):
                cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.putText(image, f"ID:{det.tag_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"[INFO] ID: {det.tag_id}, center: {det.center}")
            center_y=cy

        # ✅ 영상 출력
        cv2.imshow("Camera View", image)





        if case==1: #일반주행 모드
            
            angle = 0
            speed = 10

            if parallel == False: # 평행상수가 false이면
                case=2

            if center_y < k:      # 평형조건 만족 + 만약 ar과의 거리가 임계값 미만이면 주차모드
                case=3
            

        elif case==2:             #차선 교정 모드

            if distance<k:
                case=3

           # if 가로선 기울어짐

          #  else 안 기울어짐

        elif case==3:             #후진 모드
            if 가로선 기울어짐
                angle = 0
                speed = -10
                drive(angle, speed)
                time.sleep(3)
                case=1
            #else안기울어지면
            #drive(적당한 값)
            #exit(0)

        print(case)



        drive(angle, speed)

        # Default 제어 명령, 추후에 삭제 예정인데 형식은 그대로 가져다 쓰자


        # ==================[ 수정 끝  ]================== 

        time.sleep(0.1)

        

        cv2.waitKey(1)


#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================

if __name__ == '__main__':

    start()