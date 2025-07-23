import numpy as np

class Rubbercone:
    """
    2-D LiDAR 기반 장애물 회피·속도 계획 클래스.  
    · step() → (steer_angle[deg], speed[m/s]) 반환
    """

    # ───────────────────── 초기화 ────────────────────────────────
    def __init__(self):
        # 거리 임계값 [m]
        self._DIST_THRESHOLD_FAR  = 1.0   # >1.0m → 최고속
        self._DIST_THRESHOLD_MID  = 0.6   # >0.6m → 고속
        self._DIST_THRESHOLD_NEAR = 0.4   # >0.4m → 중속

        # 속도 [m/s]
        self._SPEED_MAX  = 10
        self._SPEED_HIGH = 10
        self._SPEED_MID  = 11
        self._SPEED_SLOW = 11             # 현재 적용되고 있는 파라미터

        # Gap 파라미터
        self._GAP_THRESH_LOW   = 0.01
        self._GAP_THRESH       = 0.7
        self._MIN_GAP_PTS      = 10       # LiDAR 포인트 ~개 이하는 ‘좁은 틈’으로 간주

        self._STEER_SCALE_HIGH = 1.0
        self._STEER_SCALE_MID  = 1.3
        self._STEER_SCALE_SLOW = 1.5      # 현재 적용되고 있는 파라미터

        # 전방 인덱스(약 200도)
        self._left_indices  = np.arange(391, 252, -1)   # 391‥253 (139)
        self._right_indices = np.arange(252, 113, -1)   # 252‥114 (139)

        self._centre_front =  (len(self._left_indices) + len(self._right_indices)) // 2

        # 버퍼
        self._image  = None
        self._ranges = None

    # ───────────────────── 입력 수신 ─────────────────────────────
    def get_value(self, image, ranges):
        self._image  = image
        self._ranges = np.asarray(ranges)

    # ───────────────────── 주행 결정 ────────────────────────────
    def step(self):
        if self._ranges is None:
            return 0.0, self._SPEED_MAX

        front = self._get_front_ranges()
               
        # ─── dist_min 계산: 오류 값(<= 1 cm) 무시 ───
        valid_front = front[front > self._GAP_THRESH_LOW]   # 0.01 m 이하는 제외
        if valid_front.size == 0:                           # 전부 오류면 failsafe
            dist_min = np.inf
        else:
            dist_min = valid_front.min()

        c = self._centre_front

        # print(f"{dist_min}")       # 디버깅용

        if dist_min > self._DIST_THRESHOLD_FAR:
            return 0.0, self._SPEED_MAX

        gap_idx = self._find_largest_gap(front)
        offset  = gap_idx - c

        if dist_min > self._DIST_THRESHOLD_MID:
            angle = offset * self._STEER_SCALE_HIGH
            return angle, self._SPEED_HIGH
        elif dist_min > self._DIST_THRESHOLD_NEAR:
            angle = offset * self._STEER_SCALE_MID
            return angle, self._SPEED_MID
        else:
            angle = offset * self._STEER_SCALE_SLOW
            return angle, self._SPEED_SLOW

    # ───────────────────── 헬퍼 ────────────────────────────────
    def _get_front_ranges(self):
        """
        정면 약 200도를 반환한다
        """

        left  = self._ranges[self._left_indices]
        right = self._ranges[self._right_indices]
        return np.concatenate([left, right])

    def _find_largest_gap(self, front_ranges):
        """
        distance ≥ _GAP_THRESH(m) 인 연속 구간 중
        가장 긴 구간의 중앙 인덱스를 반환.
        """

        valid_mask = (front_ranges >= self._GAP_THRESH) | (front_ranges <= self._GAP_THRESH_LOW)   

        if not valid_mask.any():
            return len(front_ranges) // 2         

        padded = np.concatenate([[False], valid_mask, [False]])
        diff   = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0] - 1
        sizes  = ends - starts + 1

        mask   = sizes >= self._MIN_GAP_PTS          # 갭 최소치 추가
        starts, ends, sizes = starts[mask], ends[mask], sizes[mask]
        
        if sizes.size == 0:                          # 후보가 사라지면
            return len(front_ranges)//2              # 그냥 중앙 유지

        k = np.argmax(sizes)
        start_idx, end_idx = starts[k], ends[k]

        # 최대 갭 출력

        # print(f"[GAP] start={start_idx:3d}, end={end_idx:3d}")       

        # ───────────────────────────────────────────────────────────────────────────────────

        # 갭 전부 한번에 출력
        
        # print(list(zip(starts, ends, sizes)))   
        
        return (start_idx + end_idx) // 2
