import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def bezier(t, P0, P1, P2, P3):
    return (1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1 + 3*(1 - t) * t**2 * P2 + t**3 * P3

class DraggableControlPoints:
    def __init__(self, ax, P0, P3, theta0_rad, theta1_rad, d0_init, d1_init):
        self.ax = ax
        self.P0 = P0
        self.P3 = P3
        self.theta0 = theta0_rad
        self.theta1 = theta1_rad

        self.d0 = d0_init
        self.d1 = d1_init

        self.dir0 = np.array([np.cos(self.theta0), np.sin(self.theta0)])
        self.dir1 = np.array([np.cos(self.theta1), np.sin(self.theta1)])

        self.P1 = self.P0 + self.d0 * self.dir0
        self.P2 = self.P3 + self.d1 * self.dir1

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

    def update_curve(self):
        curve_points = np.array([bezier(t, self.P0, self.P1, self.P2, self.P3) for t in self.ts])
        self.curve.set_data(curve_points[:,0], curve_points[:,1])
        self.points.set_offsets([self.P0, self.P1, self.P2, self.P3])

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
            # 시작점 기준 벡터
            vec = mouse_xy - self.P0
            # dir0 단위벡터와 벡터의 내적으로 거리값 추출 (음수 포함)
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

def plot_bezier_interactive(P0, P3, theta0_deg, angle_diff_deg, d0=3, d1=3):
    P0 = np.array(P0)
    P3 = np.array(P3)
    theta0 = np.deg2rad(theta0_deg)
    theta1 = theta0 + np.deg2rad(angle_diff_deg)

    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    ax.set_title(f'Bezier Curve with Fixed Tangents\nStart angle: {theta0_deg}°, Angle diff: {angle_diff_deg}°')
    ax.axis('equal')
    ax.grid(True)

    draggable = DraggableControlPoints(ax, P0, P3, theta0, theta1, d0, d1)

    ax_d0 = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_d1 = plt.axes([0.1, 0.15, 0.8, 0.03])

    slider_d0 = Slider(ax_d0, 'Distance d0', -20.0, 20.0, valinit=d0)
    slider_d1 = Slider(ax_d1, 'Distance d1', -20.0, 20.0, valinit=d1)

    slider_d0.on_changed(draggable.update_d0)
    slider_d1.on_changed(draggable.update_d1)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_bezier_interactive(P0=[0,0], P3=[10,5], theta0_deg=45, angle_diff_deg=50)
