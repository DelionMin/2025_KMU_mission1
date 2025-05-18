import numpy as np
import matplotlib.pyplot as plt

def bezier(t, P0, P1, P2, P3):
    return (1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1 + 3*(1 - t) * t**2 * P2 + t**3 * P3

def plot_bezier_with_fixed_tangents(P0, P3, theta0_deg, angle_diff_deg, d0=3, d1=3):
    """
    P0, P3: ndarray or list-like, shape (2,) - 시작점, 끝점 좌표
    theta0_deg: float - 시작점 접선 각도 (도 단위)
    angle_diff_deg: float - 끝점 접선 각도 - 시작점 접선 각도 (도 단위)
    d0, d1: float - 시작점/끝점에서 제어점까지 거리 (기본값 3)
    """
    P0 = np.array(P0)
    P3 = np.array(P3)
    theta0 = np.deg2rad(theta0_deg)
    theta1 = theta0 + np.deg2rad(angle_diff_deg)

    # 제어점 계산
    P1 = P0 + d0 * np.array([np.cos(theta0), np.sin(theta0)])
    P2 = P3 - d1 * np.array([np.cos(theta1), np.sin(theta1)])

    # 베지어 곡선 생성
    ts = np.linspace(0, 1, 100)
    curve = np.array([bezier(t, P0, P1, P2, P3) for t in ts])

    # 플롯
    plt.figure(figsize=(8,6))
    plt.plot(curve[:,0], curve[:,1], 'b-', label='Bezier Curve')
    plt.scatter([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]],
                c=['green','red','red','magenta'], s=100, label='Points')
    plt.text(P0[0], P0[1], 'P0(Start)', fontsize=9, color='green')
    plt.text(P1[0], P1[1], 'P1(Control)', fontsize=9, color='red')
    plt.text(P2[0], P2[1], 'P2(Control)', fontsize=9, color='red')
    plt.text(P3[0], P3[1], 'P3(End)', fontsize=9, color='magenta')

    # 접선 화살표
    plt.arrow(P0[0], P0[1], np.cos(theta0), np.sin(theta0), color='green', head_width=0.3, length_includes_head=True)
    plt.arrow(P3[0], P3[1], np.cos(theta1), np.sin(theta1), color='magenta', head_width=0.3, length_includes_head=True)

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f'Bezier Curve with Fixed Tangents\nStart angle: {theta0_deg}°, Angle diff: {angle_diff_deg}°')
    plt.show()

# 예시 호출
if __name__ == "__main__":
    plot_bezier_with_fixed_tangents(P0=[0,0], P3=[10,5], theta0_deg=45, angle_diff_deg=45)
