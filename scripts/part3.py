#!/usr/bin/python3.6

from ins_solver import Mesh_160x160, Mesh_80x80, Mesh_40x40, Mesh_20x20, Mesh_10x10
from ins_solver import IE_160x160, IE_80x80, IE_40x40, IE_20x20, IE_10x10
from ins_solver import to_np_array, x_y_coords
from ins_solver import BConds_Part3
from ins_solver import triple

from matplotlib.pyplot import figure, figaspect, savefig, show, title, legend, close, plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from numpy import sum, abs, ceil, floor, log2, sqrt, isnan, array, delete
from numpy.linalg import lstsq

from os import mkdir
from time import clock

def plot_mesh(mesh, name, plot_vectors = False):
    x, y = x_y_coords(mesh)
    p, u, v = to_np_array(mesh)

    if plot_vectors is True:
        num_figs = 4
    elif isinstance(plot_vectors, bool):
        num_figs = 3
    else:
        num_figs = plot_vectors

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
    if plot_vectors is True:
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

def plot_u_sym(mesh, m_name):
    x, y = x_y_coords(mesh)
    P, u, v = to_np_array(mesh)
    label = "u for x={} on the {} mesh".format(0.5 * (x[len(u) // 2, 0] + x[len(u) // 2 - 1, 0]), m_name)
    plot(y[len(u) // 2], 0.5 * (u[len(u) // 2] + u[len(u) // 2 - 1]),
         label=label)

def prob3_1():
    # Some experimenation showed that a diffusion constant
    # of 128.0 removes most of the ringing in the result
    bc = BConds_Part3(reynolds = 100.0, diffuse=128.0)
    ie_ts = IE_20x20(bc)
    delta_P = []
    delta_u = []
    delta_v = []
    times = []
    prev_delta = float('inf')
    for ts in range(600):
        prev_delta = ie_ts.timestep(0.05)
        delta_P.append(log2(abs(prev_delta[0])))
        delta_u.append(log2(abs(prev_delta[1])))
        delta_v.append(log2(abs(prev_delta[2])))
        times.append(ts)
    plot_mesh(ie_ts.mesh(), "Zero Steady State", False)
    figure()
    times = array(times)
    plot(times, delta_P, label="Delta P")
    plot(times, delta_u, label="Delta u")
    plot(times, delta_v, label="Delta v")
    legend()
    title("Convergence to zero")
    show()

def avg_value(mesh_src, mesh_bounds, bound_i, bound_j):
    x_min, y_min = mesh_bounds.x_min(bound_i), mesh_bounds.y_min(bound_j)
    x_max, y_max = mesh_bounds.x_max(bound_i), mesh_bounds.y_max(bound_j)
    i_min, j_min = round(x_min / mesh_src.dx()), round(y_min / mesh_src.dy())
    i_max, j_max = round(x_max / mesh_src.dx()), round(y_max / mesh_src.dy())

    tot = triple(0.0, 0.0, 0.0)
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            tot += mesh_src[i, j]
    tot /= (i_max - i_min) * (j_max - j_min)
    return tot

def grid_convergence():
    # 32 is too large a diffusion term for IE_10x10, so reduce it here
    bc = BConds_Part3(wall_vel = 1.0, reynolds = 100.0, diffuse=16.0, relax=1.0)
    ie_ts = [IE_10x10(bc), IE_20x20(bc), IE_40x40(bc), IE_80x80(bc), IE_160x160(bc)]
    mesh_names = ["10x10", "20x20", "40x40", "80x80", "160x160"]
    for ts, m_name in zip(ie_ts, mesh_names):
        print("Convergence test for {}".format(type(ts)))
        prev_delta = float('inf')
        start = clock()
        i = 0
        while prev_delta > 1e-9:
            prev_delta = ts.timestep(0.05).l2_norm()
            i += 1
        end = clock()
        print("{} s to converge in {} iterations for {}".format(end - start, i, m_name))
        plot_u_sym(ts.mesh(), m_name)

    legend()
    show()
    sol_mesh = ie_ts[-1].mesh()
    errors = []
    for ts, m_name in zip(ie_ts[:-1], mesh_names[:-1]):
        err = triple(0.0, 0.0, 0.0)
        mesh = ts.mesh()
        # Ignore the corners to avoid effects from ringing
        # for i in range(type(mesh).x_dim() // 10,
        #                type(mesh).x_dim() - type(mesh).x_dim() // 10):
        #     for j in range(type(mesh).y_dim() // 10,
        #                    type(mesh).y_dim() - type(mesh).y_dim() // 10):
        for i in range(type(mesh).x_dim()):
            for j in range(type(mesh).y_dim()):
                fine_val = avg_value(sol_mesh, mesh, i, j)
                err_delta = (mesh[i, j] - fine_val) * mesh.dx() * mesh.dy()
                mesh[i, j] = err_delta
                err[0] += abs(err_delta[0])
                err[1] += abs(err_delta[1])
                err[2] += abs(err_delta[2])

        # for i in range(type(mesh).x_dim()):
        #     for j in range(type(mesh).y_dim() // 10):
        #         mesh[i, j] = triple()
        #         mesh[i, type(mesh).y_dim() - j - 1] = triple()
        # for i in range(type(mesh).x_dim() // 10):
        #     for j in range(type(mesh).y_dim()):
        #         mesh[i, j] = triple()
        #         mesh[type(mesh).x_dim() - i - 1, j] = triple()
        plot_mesh(mesh, m_name)
        errors.append(err)

    show()
    e_cur = errors[0]
    v_names = ["Pressure", "u velocity", "v velocity"]
    for e_next, m_name in zip(errors[1:], mesh_names[1:-1]):
        for i in range(3):
            order = log2(abs(e_cur[i] / e_next[i]))
            print("{} Estimated order of convergence of {}: {}".format(m_name, v_names[i], order))
        e_cur = e_next

def prob3_2():
    # Overrelaxing by a factor of 1.5 has given me the best results
    bc_1 = BConds_Part3(wall_vel = 1.0, reynolds = 100.0, diffuse=128.0, relax=1.5)
    ie_1_ts = IE_20x20(bc_1)
    bc_m1 = BConds_Part3(wall_vel = -1.0, reynolds = 100.0, diffuse=128.0, relax=1.5)
    ie_m1_ts = IE_20x20(bc_m1)
    delta_norm = []
    times = []
    prev_delta = triple(float('inf'), float('inf'), float('inf'))
    ts = 0
    while prev_delta.l2_norm() > 1e-6:
        ie_m1_ts.timestep(0.05)
        prev_delta = ie_1_ts.timestep(0.05)
        delta_norm.append(log2(sqrt(prev_delta.l2_norm())))
        ts += 1
        times.append(ts)
    fig = plot_mesh(ie_1_ts.mesh(), "U=1.0 Steady State", 5)
    ax4 = fig.add_subplot(1, 5, 4)

    x, y = x_y_coords(ie_1_ts.mesh())
    _, u, _ = to_np_array(ie_1_ts.mesh())
    _, u_m1, _ = to_np_array(ie_m1_ts.mesh())
    ax4.plot(y[len(u) // 2], 0.5 * (u[len(u) // 2] + u[len(u) // 2 - 1]))
    ax4.set_title("u along x={}".format(0.5 * (x[len(u) // 2, 0] + x[len(u) // 2 - 1, 0])))

    ax5 = fig.add_subplot(1, 5, 5, projection='3d')
    ax5.plot_surface(x, y, u + u_m1[::-1, :], cmap=cm.gist_heat)
    ax5.set_title("Symmetry test of u")

    figure()
    times = array(times)
    plot(times, delta_norm, label="Max Delta L2 Norm")
    legend()
    title("Convergence to steady state")
    show()

    grid_convergence()

bc_1 = BConds_Part3(wall_vel = 1.0, reynolds = 100.0, diffuse=128.0, relax=1.5)
ie_1_ts = IE_20x20(bc_1)
plot_mesh(ie_1_ts.mesh(), "Initial conditions", True)

prob3_1()
prob3_2()
