#!/usr/bin/python3.6

from ins_solver import Mesh_80x320, Mesh_40x160, Mesh_20x80, Mesh_10x40
from ins_solver import IE_80x320, IE_40x160, IE_20x80, IE_10x40
from ins_solver import Mesh_160x320, Mesh_80x160, Mesh_40x80, Mesh_20x40, Mesh_10x20
from ins_solver import IE_160x320, IE_80x160, IE_40x80, IE_20x40, IE_10x20
from ins_solver import Mesh_160x160, Mesh_80x80, Mesh_40x40, Mesh_20x20, Mesh_10x10
from ins_solver import IE_160x160, IE_80x80, IE_40x40, IE_20x20, IE_10x10
from ins_solver import to_np_array, x_y_coords
from ins_solver import BConds_Part3
from ins_solver import triple

from matplotlib.pyplot import figure, figaspect, savefig, show, title, legend, close, plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from numpy import sum, abs, ceil, floor, sign, log2, sqrt, isnan, array, ones, delete
from numpy.linalg import lstsq

from os import mkdir
from time import clock

def plot_mesh(mesh, name):
    x, y = x_y_coords(mesh)
    p, u, v = to_np_array(mesh)

    num_figs = 4

    fig = figure(figsize=figaspect(1.0 / num_figs))
    ax1 = fig.add_subplot(1, num_figs, 1, projection='3d')
    ax1.plot_surface(x, y, p, cmap=cm.gist_heat)
    ax1.contour(x, y, p)
    ax1.set_title("Pressure")
    ax2 = fig.add_subplot(1, num_figs, 2, projection='3d')
    ax2.plot_surface(x, y, u, cmap=cm.gist_heat)
    ax2.contour(x, y, u)
    ax2.set_title("u")
    ax3 = fig.add_subplot(1, num_figs, 3, projection='3d')
    ax3.plot_surface(x, y, v, cmap=cm.gist_heat)
    ax3.contour(x, y, v)
    ax3.set_title("v")
    ax4 = fig.add_subplot(1, num_figs, 4)
    x = delete(x, list(range(0, x.shape[0], 2)), axis=0)
    y = delete(y, list(range(0, y.shape[0], 2)), axis=0)
    u = delete(u, list(range(0, u.shape[0], 2)), axis=0)
    v = delete(v, list(range(0, v.shape[0], 2)), axis=0)
    x = delete(x, list(range(0, x.shape[0], 2)), axis=0)
    y = delete(y, list(range(0, y.shape[0], 2)), axis=0)
    u = delete(u, list(range(0, u.shape[0], 2)), axis=0)
    v = delete(v, list(range(0, v.shape[0], 2)), axis=0)

    x = delete(x, list(range(0, x.shape[1], 2)), axis=1)
    y = delete(y, list(range(0, y.shape[1], 2)), axis=1)
    u = delete(u, list(range(0, u.shape[1], 2)), axis=1)
    v = delete(v, list(range(0, v.shape[1], 2)), axis=1)
    x = delete(x, list(range(0, x.shape[1], 2)), axis=1)
    y = delete(y, list(range(0, y.shape[1], 2)), axis=1)
    u = delete(u, list(range(0, u.shape[1], 2)), axis=1)
    v = delete(v, list(range(0, v.shape[1], 2)), axis=1)
    ax4.quiver(x, y, u, v, scale=8.0, cmap=cm.gist_heat)
    #ax4.streamplot(x=x[:, 0], y=y[0], u=u, v=v, cmap=cm.gist_heat)
    ax4.set_title("velocity")

    fig.suptitle(name)
    return fig

def filter_vel(mesh, max_vel):
    for i in range(type(mesh).x_dim()):
        for j in range(type(mesh).y_dim()):
            t = mesh[i, j]
            p, u, v = t[0], t[1], t[2]
            vel = sqrt(u * u + v * v)
            if vel <= max_vel:
                c = 1.0 / max_vel
                mesh[i, j] = t * c

def secant_zeros(data):
    """
    Returns the approximate positions of the zeros using linear interpolation
    data is a 1D array
    """
    zeros = []
    for i in range(1, len(data)):
        if data[i] == 0:
            zeros.append(float(i))
        elif sign(data[i]) != sign(data[i - 1]):
            # At least one zero is between these two points
            # Since we're working at the cell scale, dx = 1
            dydx = data[i] - data[i - 1]
            dx = data[i] / dydx
            zeros.append(i - dx)
    return zeros

def vortex_strength(mesh, x_i, y_j):
    """
    Estimates the magnitude of the curl of the velocity at x_i, y_j in cell coordinates
    """
    _, u, v = to_np_array(mesh)
    i = int(x_i)
    right_weight = x_i - i
    left_weight = 1.0 - right_weight
    u_y = u[i] * left_weight + u[i + 1] * right_weight
    # Use FV to approximate the derivatives at the interpolated cell centers
    u_y_flux = (u_y[1:] + u_y[:-1]) / 2.0
    u_y_deriv = (u_y_flux[1:] - u_y_flux[:-1]) / mesh.dy()
    j = int(y_j)
    above_weight = y_j - j
    below_weight = 1.0 - above_weight
    v_x = v[:, j] * below_weight + v[:, j + 1] * above_weight
    v_x_flux = (v_x[1:] + v_x[:-1]) / 2.0
    v_x_deriv = (v_x_flux[1:] - v_x_flux[:-1]) / 2.0
    # Not certain about this offset... TODO Work it out
    # A wrong offset shouldn't substantially affect the answer,
    # since the vortices are substantially larger than the cells
    some_offset = -2
    dvdx = (v_x_deriv[i + some_offset] * left_weight
            + v_x_deriv[i + 1 + some_offset] * right_weight)
    dudy = (u_y_deriv[j + some_offset] * below_weight
            + u_y_deriv[j + 1 + some_offset] * above_weight)
    return sqrt(dvdx * dvdx + dudy * dudy)

def find_vortex_centers(mesh):
    _, u, v = to_np_array(mesh)
    # First pass - x = 0.5
    # a factor of two off, but not an issue for this method
    y_u_strip = u[len(u) // 2] + u[len(u) // 2 - 1]
    y_j_vals = secant_zeros(y_u_strip)
    for y_j in y_j_vals:
        j = int(y_j)
        upper_weight = y_j - j
        lower_weight = 1.0 - upper_weight
        x_v_strip = v[:, j] * lower_weight + v[:, j + 1] * upper_weight
        x_i_vals = secant_zeros(x_v_strip)
        for x_i in x_i_vals:
            x_c = mesh.dx() / 2.0 + mesh.dx() * x_i
            y_c = mesh.dy() / 2.0 + mesh.dy() * y_j
            s = vortex_strength(mesh, x_i, y_j)
            print("Vortex center at ({}, {}), with strength {}".format(x_c, y_c, s))

Solver = IE_40x160
bc = BConds_Part3(wall_vel = 1.0, reynolds = 250.0, y_max = 4.0, relax = 1.25, beta = 0.75, diffuse = 32.0)
ie = Solver(bc)

prev_delta = float('inf')
ts = 0

while prev_delta > 1e-12:
    prev_delta = ie.timestep(0.1).l2_norm()
    ts += 1
    if ts % 10 == 0:
        print(ts, prev_delta)

mesh = ie.mesh()

plot_mesh(mesh, "Original")

find_vortex_centers(mesh)

def plot_u_sym(mesh, m_name):
    x, y = x_y_coords(mesh)
    P, u, v = to_np_array(mesh)
    u = 0.5 * (u[len(x) // 2] + u[len(x) // 2 - 1])
    u = abs(u)
    u_deriv = u[:-1] - u[1:]
    label = "u for x={} on the {} mesh".format(0.5 * (x[len(x) // 2, 0] + x[len(x) // 2 - 1, 0]), m_name)
    plot(y[len(x) // 2], u, label=label)
    plot(y[len(x) // 2, :-1], u_deriv, label="u deriv")
    legend()

figure()
plot_u_sym(mesh, "Rescaled 0.1")

find_vortex_centers(mesh)

filter_vel(mesh, 0.1)
plot_mesh(mesh, "Rescaled 0.1")

figure()
plot_u_sym(mesh, "Rescaled 0.1")

filter_vel(mesh, 0.1)
plot_mesh(mesh, "Rescaled 0.01")

filter_vel(mesh, 0.1)
plot_mesh(mesh, "Rescaled 0.001")

filter_vel(mesh, 0.1)
plot_mesh(mesh, "Rescaled 0.0001")

filter_vel(mesh, 0.1)
plot_mesh(mesh, "Rescaled 0.0001")


show()
