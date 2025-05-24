import numpy as np

class Rubbercone:
    def __init__(self):
        # 장애물이 6m 이내이면 gap 탐색, 그 외 speed=6 직진
        self._DIST_THRESHOLD = 6.0

        # 속도 설정
        self._SPEED_DEFAULT = 6
        self._SPEED_GAP = 3

        # Gap 탐색 파라미터
        self._GAP_THRESH = 2.4  # distance>=1.0 -> 열려있는 구간
        self._STEER_SCALE = 2.3

        # 이미지, 라이다 저장
        self._image = None
        self._ranges = None

    def get_value(self, image, ranges):
        """
        외부(메인 루프)에서 카메라 영상(image)과 라이다 거리배열(ranges) 전달
        """
        self._image = image
        self._ranges = ranges

    def step(self):
        """
        1) 전방 180도 구하기 (0~89, 270~359)
          - 최종 배열 길이=180
          - 인덱스 0 => 왼쪽 끝, 89 => 중앙, 179 => 오른쪽 끝
        2) dist_min > 7 => angle=0, speed=6 (직진)
        3) dist_min <= 7 => gap 탐색 => (angle, speed=4)
        4) 반환: (angle, speed)
        """
        if self._ranges is None:
            # 라이다 준비 안 됨 => 임의로 speed=6, angle=0
            return 0, 6

        front_ranges = self._get_front_ranges()
        dist_min = min(front_ranges)
        if dist_min > self._DIST_THRESHOLD:
            # 장애물이 7m보다 멀면 -> 직진
            return 0, self._SPEED_DEFAULT
        else:
            # 7m 이내 -> gap 탐색
            gap_idx = self._find_largest_gap(front_ranges)
            # 중앙 인덱스 = 89
            angle = (gap_idx - 89) * self._STEER_SCALE
            speed = self._SPEED_GAP
            print(angle)
            return angle, speed

    def _get_front_ranges(self):
        """
        왼쪽(0~89) + 오른쪽(270~359) 를
        '왼쪽끝 -> 중앙 -> 오른쪽끝' 순서로 합쳐 전방 180도 배열 생성
        - left_segment: [0~89], reverse() => (89->0)
        - right_segment: [270~359], reverse() => (359->270)
        - front_ranges = left_segment + right_segment
        => 길이=90+90=180
        => index 0 = 실제 왼쪽끝, index 89=중앙, index179=오른쪽끝
        """
        left_segment = list(self._ranges[0:90])    # 0~89
        left_segment.reverse()                     # 89->...->0
        right_segment = list(self._ranges[270:360])# 270~359
        right_segment.reverse()                    # 359->...->270

        front_ranges = left_segment + right_segment  # len=180
        return front_ranges

    def _find_largest_gap(self, front_ranges):
        """
        front_ranges(길이=180) 내에서
        distance >= self._GAP_THRESH 인 연속 구간 중 가장 긴 곳을 찾고, 중앙 인덱스 반환
        front_ranges[0] = 왼쪽끝, 179=오른쪽끝, 89=중앙
        """
        threshold = self._GAP_THRESH
        max_gap_size = 0
        best_center = 89  # 중앙(왼쪽=0, 오른쪽=179)
        i = 0
        length = len(front_ranges)

        while i < length:
            if front_ranges[i] >= threshold:
                start_i = i
                while i < length and front_ranges[i] >= threshold:
                    i += 1
                end_i = i - 1
                gap_size = end_i - start_i + 1
                if gap_size > max_gap_size:
                    max_gap_size = gap_size
                    best_center = (start_i + end_i)//2
            i += 1

        return best_center
