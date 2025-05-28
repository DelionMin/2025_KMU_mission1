import numpy as np

class Rubbercone:
    def __init__(self):
        # 거리별 다단계 속도 제어
        self._DIST_THRESHOLD_FAR = 10.0   # 10m 이상: 최고속도
        self._DIST_THRESHOLD_MID = 6.0    # 6-10m: 중간속도  
        self._DIST_THRESHOLD_NEAR = 3.0   # 3-6m: 저속
        
        # 속도 설정 (매우 공격적으로)
        self._SPEED_MAX = 20        # 최고속도 (장애물 멀 때)
        self._SPEED_HIGH = 15       # 고속 (중간거리)
        self._SPEED_MID = 10        # 중속 (가까운 거리)
        self._SPEED_SLOW = 7.0        # 저속 (매우 가까울 때)
        
        # Gap 탐색 파라미터 (더 공격적인 조향)
        self._GAP_THRESH = 2.4      # gap 판단 기준을 더 엄격하게
        self._STEER_SCALE = 4.0     # 기본 조향 강도를 대폭 증가
        self._STEER_SCALE_HIGH = 2.5    # 고속일 때 조향 (안정성 고려)
        self._STEER_SCALE_MID = 4.0     # 중속일 때 조향
        self._STEER_SCALE_SLOW = 5.0    # 저속일 때 조향 (더 공격적)
        
        # 이미지, 라이다 저장
        self._image = None
        self._ranges = None
        
        # 성능 최적화를 위한 미리 계산된 인덱스
        self._left_indices = np.arange(89, -1, -1)   # [89, 88, ..., 1, 0]
        self._right_indices = np.arange(359, 269, -1)  # [359, 358, ..., 271, 270]
        
    def get_value(self, image, ranges):
        """
        외부(메인 루프)에서 카메라 영상(image)과 라이다 거리배열(ranges) 전달
        """
        self._image = image
        # numpy array로 변환하여 벡터화 연산 활용
        if not isinstance(ranges, np.ndarray):
            self._ranges = np.array(ranges)
        else:
            self._ranges = ranges
    
    def step(self):
        """
        거리별 다단계 속도 제어로 안전하면서도 빠른 주행
        1) 전방 180도 구하기 (0~89, 270~359)
        2) 거리에 따른 속도 및 조향 결정:
           - 10m 이상: 최고속도로 직진
           - 6-10m: 고속으로 직진  
           - 3-6m: 중속으로 gap 탐색
           - 3m 미만: 저속으로 신중한 gap 탐색
        """
        if self._ranges is None:
            return 0, self._SPEED_MAX
        
        front_ranges = self._get_front_ranges()
        dist_min = np.min(front_ranges)
        
        # 거리별 다단계 제어
        if dist_min > self._DIST_THRESHOLD_FAR:
            # 10m 이상: 최고속도 직진
            return 0, self._SPEED_MAX
        elif dist_min > self._DIST_THRESHOLD_MID:
            # 6-10m: 고속 직진 (약간의 조향 허용)
            gap_idx = self._find_largest_gap(front_ranges)
            angle = (gap_idx - 89) * self._STEER_SCALE_HIGH  # 고속용 조향각도
            return angle, self._SPEED_HIGH
        elif dist_min > self._DIST_THRESHOLD_NEAR:
            # 3-6m: 중속으로 공격적 gap 탐색
            gap_idx = self._find_largest_gap(front_ranges)
            angle = (gap_idx - 89) * self._STEER_SCALE_MID   # 중속용 조향각도
            return angle, self._SPEED_MID
        else:
            # 3m 미만: 저속으로 매우 공격적인 gap 탐색
            gap_idx = self._find_largest_gap(front_ranges)
            angle = (gap_idx - 89) * self._STEER_SCALE_SLOW  # 저속용 최대 조향각도
            print(f"Aggressive turn! dist={dist_min:.2f}, angle={angle:.2f}")
            return angle, self._SPEED_SLOW
    
    def _get_front_ranges(self):
        """
        numpy 벡터화 연산을 사용하여 전방 180도 배열 생성
        왼쪽(0~89) + 오른쪽(270~359) 를
        '왼쪽끝 -> 중앙 -> 오른쪽끝' 순서로 합쳐 전방 180도 배열 생성
        """
        # 벡터화 연산으로 한번에 추출 및 뒤집기
        left_segment = self._ranges[self._left_indices]   # [89->0] 순서
        right_segment = self._ranges[self._right_indices] # [359->270] 순서
        
        # concatenate로 합치기 (더 빠름)
        front_ranges = np.concatenate([left_segment, right_segment])
        return front_ranges
    
    def _find_largest_gap(self, front_ranges):
        """
        numpy 벡터화 연산을 사용하여 가장 큰 gap 찾기
        front_ranges(길이=180) 내에서
        distance >= self._GAP_THRESH 인 연속 구간 중 가장 긴 곳을 찾고, 중앙 인덱스 반환
        """
        threshold = self._GAP_THRESH
        
        # 벡터화 연산으로 threshold 이상인 위치 찾기
        valid_mask = front_ranges >= threshold
        
        if not np.any(valid_mask):
            # gap이 없으면 중앙 반환
            return 89
        
        # 연속된 구간 찾기 (diff를 이용한 효율적인 방법)
        # valid_mask의 변화점 찾기
        padded = np.concatenate([[False], valid_mask, [False]])
        diff = np.diff(padded.astype(int))
        
        starts = np.where(diff == 1)[0]   # gap 시작점들
        ends = np.where(diff == -1)[0] - 1  # gap 끝점들
        
        if len(starts) == 0:
            return 89
        
        # 각 gap의 크기 계산
        gap_sizes = ends - starts + 1
        
        # 가장 큰 gap 찾기
        max_gap_idx = np.argmax(gap_sizes)
        best_start = starts[max_gap_idx]
        best_end = ends[max_gap_idx]
        best_center = (best_start + best_end) // 2
        
        return best_center
