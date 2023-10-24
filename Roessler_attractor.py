import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

N_trajectories = 20

def roessler_deriv(xyz, t0, a=0.2, b=0.2, c=5.7):
    x, y, z = xyz
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

np.random.seed(1)
x0 = -15 + 30 * np.random.random((N_trajectories, 3))

t = np.linspace(0, 100, 10000)
x_t = np.asarray([odeint(roessler_deriv, x0i, t) for x0i in x0])

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')

colors = plt.cm.rainbow(np.linspace(0, 1, N_trajectories))
lines = [ax.plot([], [], [], '-', c=c, lw=0.5)[0] for c in colors]
pts = [ax.plot([], [], [], ' ', markersize=0)[0] for _ in range(N_trajectories)]

ax.set_xlim((-30, 30))
ax.set_ylim((-30, 30))
ax.set_zlim((0, 30))
ax.set_facecolor((0, 0, 0, 1))

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

def animate(i):
    for line, xi in zip(lines, x_t):
        x, y, z = xi.T
        line.set_data(x[:i], y[:i])
        line.set_3d_properties(z[:i])
    return lines

# Mengurangi interval menjadi setengah (10 menjadi 5) untuk mempercepat animasi.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=x_t[0].shape[0], interval=5, blit=True)

plt.show()
