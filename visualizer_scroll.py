import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def bezier(t, P0, P1, P2, P3):
    return (1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1 + 3*(1 - t) * t**2 * P2 + t**3 * P3

class DraggableControlPoints:
    def __init__(self, ax, P0, P3, theta0_rad, d0_init, d1_init):
        self.ax = ax
        self.P0 = P0
        self.P3 = P3
        self.theta0 = theta0_rad
        self.theta1 = np.pi / 2  # y축 방향 고정 (90도)

        self.d0 = d0_init
        self.d1 = d1_init

        self.update_dirs_and_points()

        self.dragging_point = None

        self.ts = np.linspace(0,1,100)

        self.curve, = ax.plot([], [], 'b-', label='Bezier Curve')
        self.points = ax.scatter([self.P0[0], self.P1[0], self.P2[0], self.P3[0]],
                                [self.P0[1], self.P1[1], self.P2[1], self.P3[1]],
                                c=['green','red','red','magenta'], s=100, picker=5)

        self.labels = []
        self.labels.append(ax.text(self.P0[0], self.P0[1], 'P0(Start)', fontsize=9, color='green'))
        self.labels.append(ax.text(self.P1[0], self.P1[1], 'P1(Control)', fontsize=9, color='red'))
        self.labels.append(ax.text(self.P2[0], self.P2[1], 'P2(Control)', fontsize=9, color='red'))
        self.labels.append(ax.text(self.P3[0], self.P3[1], 'P3(End)', fontsize=9, color='magenta'))

        self.arr0 = ax.arrow(self.P0[0], self.P0[1], self.dir0[0], self.dir0[1],
                             color='green', head_width=0.3, length_includes_head=True)
        self.arr1 = ax.arrow(self.P3[0], self.P3[1], self.dir1[0], self.dir1[1],
                             color='magenta', head_width=0.3, length_includes_head=True)

        self.update_curve()

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_dirs_and_points(self):
        self.dir0 = np.array([np.cos(self.theta0), np.sin(self.theta0)])
        self.dir1 = np.array([np.cos(self.theta1), np.sin(self.theta1)])

        self.P1 = self.P0 + self.d0 * self.dir0
        self.P2 = self.P3 + self.d1 * self.dir1

    def update_curve(self):
        curve_points = np.array([bezier(t, self.P0, self.P1, self.P2, self.P3) for t in self.ts])
        self.curve.set_data(curve_points[:,0], curve_points[:,1])
        self.points.set_offsets([self.P0, self.P1, self.P2, self.P3])

        self.labels[0].set_position((self.P0[0], self.P0[1]))
        self.labels[1].set_position((self.P1[0], self.P1[1]))
        self.labels[2].set_position((self.P2[0], self.P2[1]))

        self.arr0.remove()
        self.arr1.remove()
        self.arr0 = self.ax.arrow(self.P0[0], self.P0[1], self.dir0[0], self.dir0[1],
                                 color='green', head_width=0.3, length_includes_head=True)
        self.arr1 = self.ax.arrow(self.P3[0], self.P3[1], self.dir1[0], self.dir1[1],
                                 color='magenta', head_width=0.3, length_includes_head=True)

        self.ax.figure.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        mouse_xy = np.array([event.xdata, event.ydata])
        dist_P1 = np.linalg.norm(mouse_xy - self.P1)
        dist_P2 = np.linalg.norm(mouse_xy - self.P2)
        threshold = 0.5

        if dist_P1 < threshold:
            self.dragging_point = 'P1'
        elif dist_P2 < threshold:
            self.dragging_point = 'P2'

    def on_motion(self, event):
        if self.dragging_point is None:
            return
        if event.inaxes != self.ax:
            return

        mouse_xy = np.array([event.xdata, event.ydata])

        if self.dragging_point == 'P1':
            vec = mouse_xy - self.P0
            proj_length = np.dot(vec, self.dir0)
            self.d0 = proj_length
            self.P1 = self.P0 + self.d0 * self.dir0

        elif self.dragging_point == 'P2':
            vec = mouse_xy - self.P3
            proj_length = np.dot(vec, self.dir1)
            self.d1 = proj_length
            self.P2 = self.P3 + self.d1 * self.dir1

        self.update_curve()

    def on_release(self, event):
        self.dragging_point = None

    def update_d0(self, val):
        self.d0 = val
        self.P1 = self.P0 + self.d0 * self.dir0
        self.update_curve()

    def update_d1(self, val):
        self.d1 = val
        self.P2 = self.P3 + self.d1 * self.dir1
        self.update_curve()

    def update_theta0(self, val):
        self.theta0 = np.deg2rad(val)
        self.update_dirs_and_points()
        self.update_curve()

def plot_bezier_interactive(P0, P3, theta0_deg, d0=3, d1=3):
    P0 = np.array(P0)
    P3 = np.array(P3)

    fig, ax = plt.subplots(figsize=(16,12))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    ax.set_title(f'Bezier Curve with Fixed End Angle (90°) and Adjustable Start Angle')
    ax.axis('equal')
    ax.grid(True)

    draggable = DraggableControlPoints(ax, P0, P3, np.deg2rad(theta0_deg), d0, d1)

    
    # d0 Distance 슬라이더 위치를 0.25로 (세번째 슬라이더 위치로 변경)
    ax_d0 = plt.axes([0.1, 0.25, 0.8, 0.03])
    slider_d0 = Slider(ax_d0, 'd1 Distance', -20.0, 20.0, valinit=d0)
    slider_d0.on_changed(draggable.update_d0)

    # 각도 슬라이더 그대로
    ax_theta0 = plt.axes([0.1, 0.20, 0.8, 0.03])
    slider_theta0 = Slider(ax_theta0, 'd1 Angle (yaw)', -180, 180, valinit=theta0_deg)
    slider_theta0.on_changed(draggable.update_theta0)

    # d1 Distance 슬라이더 위치를 0.15로 (첫번째 슬라이더 위치로 변경)
    ax_d1 = plt.axes([0.1, 0.15, 0.8, 0.03])
    slider_d1 = Slider(ax_d1, 'd2 Distance', -20.0, 20.0, valinit=d1)
    slider_d1.on_changed(draggable.update_d1)


    # 시작점 위치 조정 슬라이더 (X, Y)
    ax_P0x = plt.axes([0.1, 0.05, 0.35, 0.03])
    ax_P0y = plt.axes([0.55, 0.05, 0.35, 0.03])

    slider_P0x = Slider(ax_P0x, 'P0 X', -20.0, 20.0, valinit=P0[0])
    slider_P0y = Slider(ax_P0y, 'P0 Y', -20.0, 20.0, valinit=P0[1])

    def update_P0(val):
        x = slider_P0x.val
        y = slider_P0y.val
        draggable.P0 = np.array([x, y])
        draggable.update_dirs_and_points()
        draggable.update_curve()

    slider_P0x.on_changed(update_P0)
    slider_P0y.on_changed(update_P0)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_bezier_interactive(P0=[-5, -5], P3=[0, 0], theta0_deg=60)

"""
✅ 1. 시작점과 끝점의 방향(각도) 고정 및 조정 가능
P0의 방향(θ₀)을 슬라이더로 실시간 조정할 수 있으며,

P3의 방향은 고정되어 있어 끝점의 방향 조건이 명확한 제약 조건으로 작동합니다.

이는 실제 로봇 경로 생성, 곡선 연결, 인터폴레이션 등에서 자주 필요한 제약 조건입니다.

✅ 2. 거리(d₀, d₁)를 직관적으로 조절 가능
슬라이더를 통해 제어점과의 거리(d0, d1)를 조정함으로써 곡선의 굴곡(곡률)을 실시간 제어할 수 있습니다.

이 접근 방식은 곡률을 조정할 때 수치적으로 안정적이며 직관적입니다.

✅ 3. 제어점 드래그 기능으로 곡선 형상 직접 수정 가능
P1, P2 제어점을 마우스로 직접 드래그하여 위치 조절이 가능하며, 이 동작은 d0, d1을 업데이트하여 일관된 방향 벡터를 유지합니다.

따라서 위치 조절과 방향 조건의 연동을 유지한 채로 직관적인 수정이 가능합니다.

✅ 4. 슬라이더로 시작점 P0도 조절 가능
P0의 위치를 슬라이더로 조정하면, 시작 각도 theta0과 거리 d0에 따라 자동으로 P1이 재계산되어, 전체 곡선의 시작점 조건을 유지하며 움직임을 시뮬레이션할 수 있습니다.

✅ 5. 구현 구조가 명확하고 확장 가능
DraggableControlPoints 클래스를 중심으로, 데이터(제어점, 각도, 거리)와 동작(업데이트, 드래그, 슬라이더 연동)이 잘 분리되어 있어:

다양한 조건 제약을 추가하기 쉽고,

향후 3차원 곡선이나 다중 곡선 연결 등으로 확장하기도 유리합니다.

✅ 6. 시각적 피드백이 빠르고 직관적
각 제어점의 이름, 방향 벡터(화살표), 색상 등을 다르게 설정하여 사용자가 구조를 쉽게 이해할 수 있도록 시각적 단서를 제공합니다.

실시간 draw_idle()을 통해 빠른 피드백이 제공됩니다.

💡 요약
이 코드는 **곡선의 시작 조건(위치 + 방향)**과 **끝 조건(위치 + 방향)**을 유지하면서도 곡선의 형태를 다양한 방식으로 직관적이고 실시간으로 조정할 수 있게 하여, 인터랙티브한 곡선 설계 도구로 매우 유용합니다.
"""
