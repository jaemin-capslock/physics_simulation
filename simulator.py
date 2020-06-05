#% matplotlib inline - configure this option for google colab (graphs won't display otherwise)
#the calculation part was based on https://en.wikipedia.org/wiki/Projectile_motion#Solution_by_numerical_integration

from math import pi, radians, degrees, sin, cos, atan, sqrt, sinh, cosh, asinh
import numpy as np
from scipy.integrate import quadrature
from scipy.optimize import newton
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import matplotlib.ticker as ticker
#from google.colab import widgets
from IPython.display import HTML

print("Welcome to the fancy projectile Calculator!")

print("What's the initial velocity of the baseball that you'd like it to be launched? ")

V_eigen: float = float(input("km/h"))  # Initial velocity (km/h)
matplotlib.rcParams['animation.embed_limit'] = 2 ** 64
g = 9.81  # Acceleration due to gravity (m/s^2)
print("Sounds fair. What about the angle?")
psi = float(input())  # Launch angle (deg.)
print("Alright! Here are the reviews for the constants that are going to be used.")
c = 0.47  # Drag coefficient (spherical projectile)
r = 0.0366  # Radius of projectile (m)
m = 0.145  # Mass of projectile (kg)
rho_air = 1.29  # Air density (kg/m^3)
a = pi * r ** 2.0  # Cross-sectional area of projectile (m^2)
print("The initial velocity of this baseball will be " + str(V_eigen) + " .")

print("The launch angle of it will be " + str(psi) + " .")

print("The drag coefficient we'll be using will be " + str(c) + " .")
print("Proceed? [Y/N]")
answer = input()
if answer != "Y":
    print("Not going to proceed. Goodbye..")
    exit
V_0 = V_eigen / 3.6
grid = widgets.Grid(2, 3, header_row=True, header_column=True)
psi = radians(psi)
interval = 0.005
x_0 = 0.0
u_0 = V_0 * cos(psi)
y_0 = 0.0
v_0 = V_0 * sin(psi)

mu = 0.5 * c * rho_air * a / m
Q_0 = asinh(v_0 / u_0)
A = g / (mu * u_0 ** 2.0) + (Q_0 + 0.5 * sinh(2.0 * Q_0))


def lam(Q):
    return A - (Q + 0.5 * sinh(2.0 * Q))


def u_s(Q):
    return sqrt(g / mu) / sqrt(lam(Q))


def v_s(Q):
    return sqrt(g / mu) * sinh(Q) / sqrt(lam(Q))


def f_t(Q):
    return cosh(Q) / sqrt(lam(Q))


def f_x(Q):
    return cosh(Q) / lam(Q)


def f_y(Q):
    return sinh(2.0 * Q) / lam(Q)


def t_s(Q):
    return - quadrature(f_t, Q_0, Q, vec_func=False)[0] / sqrt(g * mu)


def x_s(Q):
    return x_0 - quadrature(f_x, Q_0, Q, vec_func=False)[0] / mu


def y_s(Q):
    return y_0 - quadrature(f_y, Q_0, Q, vec_func=False)[0] / (2.0 * mu)


def y_s_p(Q):
    return -(1.0 / (2.0 * mu)) * sinh(2.0 * Q) / lam(Q)


Q_T_est = asinh(-v_0 / u_0)  # Initial estimate for Newton's method
Q_T = newton(y_s, Q_T_est, y_s_p)
T = t_s(Q_T)

R = x_s(Q_T)
H = y_s(0.0)
t_vec = np.vectorize(t_s)
x_vec = np.vectorize(x_s)
y_vec = np.vectorize(y_s)
u_vec = np.vectorize(u_s)
v_vec = np.vectorize(v_s)

N = 300
psi_T = degrees(atan(sinh(Q_T)))
Q = np.arcsinh(np.tan(np.radians(np.linspace(degrees(psi), psi_T, N))))
t = t_vec(Q)
x = x_vec(Q)
y = y_vec(Q)
u = u_vec(Q)
v = v_vec(Q)


def get_intervals():
    t_flight = t[-1]
    intervals = []
    t_init = 0
    while t_init < t_flight:
        intervals.append(t_init)
        t_init += T / N
    return intervals


def update_position_circle(i, circle, intervals):
    t = intervals[i]
    x_pos = x[i]
    y_pos = y[i]
    circle.center = x_pos, y_pos
    return circle,


intervals = get_intervals()
xmin = 0
xmax = x[np.argmax(x)]
ymin = 0
ymax = y[np.argmax(y)]
plotmax = max(xmax, ymax)
fig = plt.gcf()
fig.set_size_inches(10.5, 7)
ax = plt.axes(xlim=(xmin, xmax + xmax * 0.15), ylim=(ymin, ymax + ymax * 0.3))
ax.set_aspect('equal')
ax.text(plotmax / 3, ymax / 2, 'Initial Velocity = ' + str(V_eigen) + 'km/h')
ax.text(plotmax / 3, (ymax / 2) - 5, "Launch Angle = " + str(np.round(degrees(psi))) + " degrees")
line, = ax.plot(x[np.argmax(y)], y[np.argmax(y)], 'ro')
ax.set_title(r'Animated Projectile motion with air resistance')
xstart, xend = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(xstart, xend, (np.round(xmax / 10))))
ystart, yend = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(ystart, yend, np.round(ymax / 7)))
ax.plot(x, y)
ax.grid(b=True)
'''
all of the sections containing grids are used for neat google colab display environment. They can be modified at your choice.

'''
rad = plotmax / 40

circle = plt.Circle((xmin, ymin), rad, color="c")
ax.add_patch(circle)
anim = animation.FuncAnimation(fig, update_position_circle, fargs=(circle, intervals),
                               frames=len(intervals) - 1, interval=1,
                               repeat=False, blit=True)

with grid.output_to(0, 0):
    print("Results : ")
    print("Total time of flight = :    {:.3f}".format(T))
    print("Total distance of flight = :    {:.3f}".format(R))
    print("Max height of flight = :    {:.3f}".format(H))
with grid.output_to(0, 1):
    plt.title("Projectile Motion with Air Resistance")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    ax.grid(b=True)
    plt.plot(x, y)

with grid.output_to(1, 0):
    fig, ax = plt.subplots()
    line, = ax.plot(t, v, 'b-', label='v')
    ax.set_title(r'Vertical velocity component')
    ax.grid(b=True)
    ax.legend()
    ax.set_xlabel('t (s)')
    ax.set_ylabel('v (m/s)')
with grid.output_to(1, 1):
    fig, ax = plt.subplots()
    line, = ax.plot(t, u, 'b-', label='u')
    ax.set_title(r'Horizontal velocity component')
    ax.grid(b=True)
    ax.legend()
    ax.set_xlabel('t (s)')
    ax.set_ylabel('u (m/s)')
    plt.show()
with grid.output_to(1, 2):
    fig, ax = plt.subplots()
    v_cul = np.sqrt(u ** 2 + v ** 2)
    line, = ax.plot(t, v_cul, 'b-', label='u')
    ax.set_title(r'Culmulative velocity component')
    ax.grid(b=True)
    ax.legend()
    ax.set_xlabel('t (s)')
    ax.set_ylabel('v (m/s)')
    plt.show()

with grid.output_to(0, 2):
    print("Made by Jaemin Ko :)")
rc('animation', html="jshtml")
anim
