import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def bezier(t, P0, P1, P2, P3): # 베지에 곡선 리턴 함수
    return (1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1 + 3*(1 - t) * t**2 * P2 + t**3 * P3

class DraggableControlPoints:
    def __init__(self, ax, P0, P3, theta0_rad, d0_init, d1_init): #상수들 초기화

        self.ax = ax              # 그래프 축 객체
        self.P0 = P0              # 시작점 좌표
        self.P3 = P3              # 끝점 좌표
        self.theta0 = theta0_rad  # 시작점의 방향 각도
        self.theta1 = np.pi / 2   # y축 방향 고정 (90도)


        self.d0 = d0_init         # 시작점 → P1 거리
        self.d1 = d1_init         # 끝점 → P2 거리


        self.update_dirs_and_points()   # 제어점 위치 업데이트 실행

        self.dragging_point = None      # 현재 드래그 중인 점

        self.ts = np.linspace(0,1,100)  # 0~1 사이의 숫자를 100등분한 배열


        # 그래프 곡선 객체 / plot 함수는 리스트를 반환하므로 ,을 붙여 unpacking 하였음
        self.curve, = ax.plot([], [], 'b-', label='Bezier Curve')

        # 4개의 점을 찍는 코드 / x좌표 4개, y좌표 4개, 색상 4개
        # s는 점의 크기, picker은 클릭 가능한 크기
        self.points = ax.scatter([self.P0[0], self.P1[0], self.P2[0], self.P3[0]],
                                [self.P0[1], self.P1[1], self.P2[1], self.P3[1]],
                                c=['green','red','red','magenta'], s=100, picker=5)

        # 점들에 라벨을 붙이는 코드
        self.labels = []

        # 시작점
        # x좌표, y좌표, 라벨 내용, 글씨 크기, 색상
        self.labels.append(ax.text(self.P0[0], self.P0[1], 'P0(Start)', fontsize=9, color='green'))

        # P1 제어점
        self.labels.append(ax.text(self.P1[0], self.P1[1], 'P1(Control)', fontsize=9, color='red'))

        # P2 제어점
        self.labels.append(ax.text(self.P2[0], self.P2[1], 'P2(Control)', fontsize=9, color='red'))

        # 종료점
        self.labels.append(ax.text(self.P3[0], self.P3[1], 'P3(End)', fontsize=9, color='magenta'))

        # P0 점의 방향 벡터 화살표를 그림
        # x좌표, y좌표, x증분, y증분, 색상, 머리 너비, true는 길이에 머리도 포함
        self.arr0 = ax.arrow(self.P0[0], self.P0[1], self.dir0[0], self.dir0[1],
                             color='green', head_width=0.3, length_includes_head=True)

        # P3 점의 방향 벡터 화살표를 그림
        self.arr1 = ax.arrow(self.P3[0], self.P3[1], self.dir1[0], self.dir1[1],
                             color='magenta', head_width=0.3, length_includes_head=True)

        # 베지에 곡선 초기값 로딩
        self.update_curve()

        # 마우스 버튼을 눌렀을 때 이벤트
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)

        # 마우스 버튼을 뗄 있을 때 이벤트
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

        # 마우스 버튼을 움직일 때 이벤트
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_dirs_and_points(self): # 제어점 업데이트
        # 각도를 방향벡터로 변환
        self.dir0 = np.array([np.cos(self.theta0), np.sin(self.theta0)])
        self.dir1 = np.array([np.cos(self.theta1), np.sin(self.theta1)])

        # P1과 P2 좌표 계산
        self.P1 = self.P0 + self.d0 * self.dir0
        self.P2 = self.P3 + self.d1 * self.dir1

    def update_curve(self): # 베지에 곡선 업데이트
        
        # 0부터 1까지 100등분 한 점에 대해 베지에 메소드를 적용해 점들을 계산
        curve_points = np.array([bezier(t, self.P0, self.P1, self.P2, self.P3) for t in self.ts])
        
        # 곡선의 점들의 데이터를 업데이트
        self.curve.set_data(curve_points[:,0], curve_points[:,1])

        # 네개의 점들의 위치를 업데이트
        self.points.set_offsets([self.P0, self.P1, self.P2, self.P3])

        # 라벨 위치 업데이트
        self.labels[0].set_position((self.P0[0], self.P0[1]))
        self.labels[1].set_position((self.P1[0], self.P1[1]))
        self.labels[2].set_position((self.P2[0], self.P2[1]))

        # 기존 방향 벡터 제거
        self.arr0.remove()
        self.arr1.remove()

        # 방향 벡터 업데이트
        self.arr0 = self.ax.arrow(self.P0[0], self.P0[1], self.dir0[0], self.dir0[1],
                                 color='green', head_width=0.3, length_includes_head=True)
        self.arr1 = self.ax.arrow(self.P3[0], self.P3[1], self.dir1[0], self.dir1[1],
                                 color='magenta', head_width=0.3, length_includes_head=True)


        # 변화가 있는 경우 redraw
        self.ax.figure.canvas.draw_idle()

    def on_press(self, event):      # 드래그 시작
        if event.inaxes != self.ax: # 스크롤바에서 발생한 이벤트가 아닐때 cancel
            return

        mouse_xy = np.array([event.xdata, event.ydata])  # 클릭 위치를 저장
        dist_P1 = np.linalg.norm(mouse_xy - self.P1)     # 클릭점~ 제어점 1의 거리
        dist_P2 = np.linalg.norm(mouse_xy - self.P2)     # 클릭점~ 제어점 2의 거리
        threshold = 0.5                                  # drag의 threshold

        if dist_P1 < threshold:         # 제어점 1과의 거리가 threshold 미만이면
            self.dragging_point = 'P1'  # p1을 그래그 포인트로 지정
        elif dist_P2 < threshold:       # 제어점 2과의 거리가 threshold 미만이면
            self.dragging_point = 'P2'  # p2을 그래그 포인트로 지정

    def on_motion(self, event):           # 마우스 따라 값 갱신
        if self.dragging_point is None:   # 드래그 값 없으면 cancel
            return
        if event.inaxes != self.ax:       # 스크롤바에서 발생한 이벤트가 아닐때 cancel
            return

        mouse_xy = np.array([event.xdata, event.ydata]) # 마우스의 위치를 저장

        if self.dragging_point == 'P1':             # p1을 드래그 중이라면
            vec = mouse_xy - self.P0                # P0~ 마우스 벡터 계산
            proj_length = np.dot(vec, self.dir0)    # dir0에 정사영
            self.d0 = proj_length                   # 정사영한 길이로 d0 update
            self.P1 = self.P0 + self.d0 * self.dir0 # P1 위치 update

        elif self.dragging_point == 'P2':            # p1을 드래그 중이라면
            vec = mouse_xy - self.P3                 # p3 ~ 마우스 벡터 계산
            proj_length = np.dot(vec, self.dir1)     # dir1에 정사영
            self.d1 = proj_length                    # 정사영한 길이로 d1 update
            self.P2 = self.P3 + self.d1 * self.dir1  # P2 위치 update

        self.update_curve() # 곡선 업데이트

    def on_release(self, event): # 드래그 종료
        self.dragging_point = None

    def update_d0(self, val): # 슬라이더로 d0 update
        self.d0 = val
        self.P1 = self.P0 + self.d0 * self.dir0
        self.update_curve() # 곡선 업데이트

    def update_d1(self, val): # 슬라이더로 d1 update
        self.d1 = val
        self.P2 = self.P3 + self.d1 * self.dir1
        self.update_curve() # 곡선 업데이트

    def update_theta0(self, val): # 슬라이더로 theta update
        self.theta0 = np.deg2rad(val)
        self.update_dirs_and_points()
        self.update_curve() # 곡선 업데이트

def plot_bezier_interactive(P0, P3, theta0_deg, d0=3, d1=3): # 메인 메소드
    P0 = np.array(P0)   # P0점 
    P3 = np.array(P3)   # P3점

    fig, ax = plt.subplots(figsize=(16,12))     # 차트 생성
    plt.subplots_adjust(left=0.1, bottom=0.35)  # 차트 레이아웃

    # 제목설정
    ax.set_title(f'Bezier Curve with Fixed End Angle (90°) and Adjustable Start Angle')

    ax.axis('equal')    # x축 y축 scale 같게
    ax.grid(True)       # 격자선 표시

    # 위에서 정의한 클래스 객체 생성
    draggable = DraggableControlPoints(ax, P0, P3, np.deg2rad(theta0_deg), d0, d1)

    
    # d1 Distance 슬라이더
    ax_d1 = plt.axes([0.1, 0.25, 0.8, 0.03])
    slider_d1 = Slider(ax_d1, 'd1 Distance', -20.0, 20.0, valinit=d0)
    slider_d1.on_changed(draggable.update_d0)

    # 각도 슬라이더
    ax_theta0 = plt.axes([0.1, 0.20, 0.8, 0.03])
    slider_theta0 = Slider(ax_theta0, 'd1 Angle (yaw)', -180, 180, valinit=theta0_deg)
    slider_theta0.on_changed(draggable.update_theta0)

    # d2 Distance 슬라이더
    ax_d2 = plt.axes([0.1, 0.15, 0.8, 0.03])
    slider_d2 = Slider(ax_d2, 'd2 Distance', -20.0, 20.0, valinit=d1)
    slider_d2.on_changed(draggable.update_d1)


    # 시작점 위치 조정 슬라이더 (X, Y)
    ax_P0x = plt.axes([0.1, 0.05, 0.35, 0.03])
    ax_P0y = plt.axes([0.55, 0.05, 0.35, 0.03])

    slider_P0x = Slider(ax_P0x, 'P0 X', -20.0, 20.0, valinit=P0[0])
    slider_P0y = Slider(ax_P0y, 'P0 Y', -20.0, 20.0, valinit=P0[1])


    def update_P0(val):     # 슬라이더로 p0위치 조정
        x = slider_P0x.val
        y = slider_P0y.val
        draggable.P0 = np.array([x, y])
        draggable.update_dirs_and_points()
        draggable.update_curve()

    slider_P0x.on_changed(update_P0)
    slider_P0y.on_changed(update_P0)

    plt.legend()
    plt.show()

if __name__ == "__main__": # main method 실행
    plot_bezier_interactive(P0=[-5, -5], P3=[0, 0], theta0_deg=60)
