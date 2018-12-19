#!/usr/bin/python3.6

from numpy import sum, log2, sqrt

from ins_solver import Mesh_160x160, Mesh_80x80, Mesh_40x40, Mesh_20x20, Mesh_10x10
from ins_solver import RK1_160x160, RK1_80x80, RK1_40x40, RK1_20x20, RK1_10x10
from ins_solver import RK4_3_160x160, RK4_3_80x80, RK4_3_40x40, RK4_3_20x20, RK4_3_10x10
from ins_solver import IE_160x160, IE_80x80, IE_40x40, IE_20x20, IE_10x10
from ins_solver import IE_160x192, IE_80x96, IE_40x48, IE_20x24, IE_10x12
from ins_solver import IE_160x240, IE_80x120, IE_40x60, IE_20x30, IE_10x15
from ins_solver import IE_1280x2560, IE_640x1280, IE_320x640, IE_160x320, IE_80x160, IE_40x80, IE_20x40, IE_10x20
from ins_solver import IE_160x640, IE_80x320, IE_40x160, IE_20x80, IE_10x40
from ins_solver import to_np_array, x_y_coords
from ins_solver import BConds_Part1, BConds_Part3
from ins_solver import INSAssembly1, INSAssembly3
from ins_solver import Jacobian, triple

from matplotlib.pyplot import figure, contour, show, clabel, title, figaspect, savefig, close
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import delete

from os import mkdir
from time import clock

def plot_errs(MeshT, plot=True):
    m_zero = MeshT()
    m_init = MeshT()
    m_fi = MeshT()
    m_fi_soln = MeshT()

    bc = BConds_Part1(P_0 = 1.0, u_0 = 1.0, v_0 = 1.0, beta = 1.0)
    sd = INSAssembly1(bc)

    bc.init_mesh(m_init)
    sd.flux_assembly(m_zero, m_init, m_fi, 0.0, 1.0)
    bc.flux_soln(m_fi_soln)

    x, y = x_y_coords(m_fi)

    p_fi, u_fi, v_fi = to_np_array(m_fi)
    p_fi_soln, u_fi_soln, v_fi_soln = to_np_array(m_fi_soln)
    p_fi_err, u_fi_err, v_fi_err = (p_fi - p_fi_soln, u_fi - u_fi_soln, v_fi - v_fi_soln)

    if plot:
        # fig = figure()
        # ax = fig.gca(projection="3d")
        # ax.plot_surface(x, y, p_fi, cmap=cm.gist_heat)
        # cs = contour(x, y, p_fi)
        # clabel(cs)
        # title("P FI {}x{} Computed".format(p_fi.shape[0], p_fi.shape[1]))

        # fig = figure()
        # ax = fig.gca(projection="3d")
        # ax.plot_surface(x, y, p_fi_soln, cmap=cm.gist_heat)
        # cs = contour(x, y, p_fi_soln)
        # clabel(cs)
        # title("P FI {}x{} Solution".format(p_fi.shape[0], p_fi.shape[1]))

        fig = figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(x, y, p_fi_err, cmap=cm.gist_heat)
        cs = ax.contour(x, y, p_fi_err, linewidths=2.0, linestyles="solid")
        title("P FI {}x{} Error".format(p_fi.shape[0], p_fi.shape[1]))

        fig = figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(x, y, u_fi, cmap=cm.gist_heat)
        cs = contour(x, y, u_fi)
        clabel(cs)
        title("U FI {}x{} Computed".format(u_fi.shape[0], u_fi.shape[1]))

        fig = figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(x, y, u_fi_soln, cmap=cm.gist_heat)
        cs = contour(x, y, u_fi_soln)
        clabel(cs)
        title("U FI {}x{} Solution".format(u_fi_soln.shape[0], u_fi_soln.shape[1]))

        fig = figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(x, y, u_fi_err, cmap=cm.gist_heat)
        cs = ax.contour(x, y, u_fi_err)
        title("U FI {}x{} Error".format(u_fi.shape[0], u_fi.shape[1]))

        # fig = figure()0
        # ax = fig.gca(projection="3d")
        # ax.plot_surface(x, y, v_fi, cmap=cm.gist_heat)
        # cs = contour(x, y, v_fi)
        # clabel(cs)
        # title("V FI {}x{} Computed".format(v_fi.shape[0], v_fi.shape[1]))

        # fig = figure()
        # ax = fig.gca(projection="3d")
        # ax.plot_surface(x, y, v_fi_soln, cmap=cm.gist_heat)
        # cs = contour(x, y, v_fi_soln)
        # clabel(cs)
        # title("V FI {}x{} Solution".format(v_fi_soln.shape[0], v_fi_soln.shape[1]))

        fig = figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(x, y, v_fi_err, cmap=cm.gist_heat)
        cs = ax.contour(x, y, v_fi_err)
        title("V FI {}x{} Error".format(v_fi.shape[0], v_fi.shape[1]))

    p_fi_l2_err = sqrt(sum(p_fi_err * p_fi_err) * m_zero.dx() * m_zero.dy())
    print("Pressure Flux Integral L2 error: {}".format(p_fi_l2_err))
    u_fi_l2_err = sqrt(sum(u_fi_err * u_fi_err) * m_zero.dx() * m_zero.dy())
    print("Velocity U Flux Integral L2 error: {}".format(u_fi_l2_err))
    v_fi_l2_err = sqrt(sum(v_fi_err * v_fi_err) * m_zero.dx() * m_zero.dy())
    print("Velocity V Flux Integral L2 error: {}".format(v_fi_l2_err))
    return p_fi_l2_err, u_fi_l2_err, v_fi_l2_err

print("10x10")
p_err_010, u_err_010, v_err_010 = plot_errs(Mesh_10x10, False)
print("20x20")
p_err_020, u_err_020, v_err_020 = plot_errs(Mesh_20x20, False)
print("20 Convergence Rate Pressure: {} u: {} v: {}".format(log2(p_err_010 / p_err_020),
                                                            log2(u_err_010 / u_err_020),
                                                            log2(v_err_010 / v_err_020)))
print("40x40")
p_err_040, u_err_040, v_err_040 = plot_errs(Mesh_40x40, False)
print("40 Convergence Rate Pressure: {} u: {} v: {}".format(log2(p_err_020 / p_err_040),
                                                            log2(u_err_020 / u_err_040),
                                                            log2(v_err_020 / v_err_040)))
print("80x80")
p_err_080, u_err_080, v_err_080 = plot_errs(Mesh_80x80, False)
print("80 Convergence Rate Pressure: {} u: {} v: {}".format(log2(p_err_040 / p_err_080),
                                                            log2(u_err_040 / u_err_080),
                                                            log2(v_err_040 / v_err_080)))
print("160x160")
p_err_160, u_err_160, v_err_160 = plot_errs(Mesh_160x160, True)
print("160 Convergence Rate Pressure: {} u: {} v: {}".format(log2(p_err_080 / p_err_160),
                                                             log2(u_err_080 / u_err_160),
                                                             log2(v_err_080 / v_err_160)))

def test_jacobian(MeshT):
    bc = BConds_Part3(P_0 = 1.0, u_0 = 1.0, v_0 = 1.0,
                      beta = 1.0, reynolds = 1.0, diffuse = 1.0)
    sd = INSAssembly3(bc)

    center = (MeshT.x_dim() // 2 - 2, MeshT.y_dim() // 2 + 3)

    m_orig = MeshT()
    bc.init_mesh(m_orig)

    epsilon = triple(1e-6, 1e-6, 1e-6)

    m_delta = MeshT()

    m_delta[center[0], center[1]] = epsilon

    m_next = MeshT()
    bc.init_mesh(m_next)
    m_next[center[0], center[1]] += epsilon

    offsets = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]

    for i_off, j_off in offsets:
        i, j = (center[0] + i_off, center[1] + j_off)

        # j_f_U_m1_approx = sd.jacobian_x_p1(m_orig, i, j - 1, 0.0) * m_delta[i - 1, j]
        # j_f_U_approx = sd.jacobian_x_0(m_orig, i, j, 0.0) * m_delta[i, j]
        # j_f_U_p1_approx = sd.jacobian_x_p1(m_orig, i, j, 0.0) * m_delta[i + 1, j]

        # j_g_U_m1_approx = sd.jacobian_y_p1(m_orig, i, j - 1, 0.0) * m_delta[i, j - 1]
        # j_g_U_approx = sd.jacobian_y_0(m_orig, i, j, 0.0) * m_delta[i, j]
        # j_g_U_p1_approx = sd.jacobian_y_p1(m_orig, i, j, 0.0) * m_delta[i, j + 1]

        # delta_U = ((j_f_U_m1_approx + j_f_U_approx + j_f_U_p1_approx)
        #            + (j_g_U_m1_approx + j_g_U_approx + j_g_U_p1_approx))

        # Note that we need the negative to account for the flux integral returning
        # the negative of the calculation
        delta_U = -((sd.Dx_m1(m_orig, i, j, 0.0) * m_delta[i - 1, j] +
                     sd.Dx_0(m_orig, i, j, 0.0) * m_delta[i, j] +
                     sd.Dx_p1(m_orig, i, j, 0.0) * m_delta[i + 1, j]) +
                    (sd.Dy_m1(m_orig, i, j, 0.0) * m_delta[i, j - 1] +
                     sd.Dy_0(m_orig, i, j, 0.0) * m_delta[i, j] +
                     sd.Dy_p1(m_orig, i, j, 0.0) * m_delta[i, j + 1]))

        orig_fi = sd.flux_integral(m_orig, i, j, 0.0)
        approx_t = orig_fi + delta_U

        err = sd.flux_integral(m_next, i, j, 0.0) - approx_t
        print("Linear Approximation Error at {}, {}:".format(i, j))
        print(err)

test_jacobian(Mesh_20x20)

#show()

def plot_mesh(mesh, name):
    x, y = x_y_coords(mesh)
    p, u, v = to_np_array(mesh)
    fig = figure(figsize=figaspect(1.0 / 4.0))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.plot_surface(x, y, p, cmap=cm.gist_heat)
    ax1.contour(x, y, p)
    ax1.set_title("Pressure")
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.plot_surface(x, y, u, cmap=cm.gist_heat)
    ax2.contour(x, y, u)
    ax2.set_title("u")
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3.plot_surface(x, y, v, cmap=cm.gist_heat)
    ax3.contour(x, y, v)
    ax3.set_title("v")
    ax4 = fig.add_subplot(1, 4, 4)
    x = delete(x, list(range(0, x.shape[0], 2)), axis=0)
    y = delete(y, list(range(0, y.shape[0], 2)), axis=0)
    u = delete(u, list(range(0, u.shape[0], 2)), axis=0)
    v = delete(v, list(range(0, v.shape[0], 2)), axis=0)
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
    x = delete(x, list(range(0, x.shape[1], 2)), axis=1)
    y = delete(y, list(range(0, y.shape[1], 2)), axis=1)
    u = delete(u, list(range(0, u.shape[1], 2)), axis=1)
    v = delete(v, list(range(0, v.shape[1], 2)), axis=1)
    ax4.quiver(x, y, u, v, scale=16.0, cmap=cm.gist_heat)
    #ax4.streamplot(x=x[:, 0], y=y[0], u=u, v=v, cmap=cm.gist_heat)
    ax4.set_title("velocity")
    fig.suptitle(name)
    return fig

beta  = 0.125
relax = 1.75
diffuse = 0.0
reynolds = 250.0
y_1 = 2.0

bc1 = BConds_Part3(wall_vel = 1.0, beta = beta, reynolds = reynolds,
                   relax = relax, diffuse = diffuse, y_max = y_1)
bcm1 = BConds_Part3(wall_vel = -1.0, beta = beta, reynolds = reynolds,
                    relax = relax, diffuse = diffuse, y_max = y_1)
ie_1_ts = IE_80x160(bc1)
ie_m1_ts = IE_80x160(bcm1)
#ie_ts = RK4_3_10x10(bc)

def figfile(ts):
    dirname = ("../report"
               + "/h{}_80x160_u1_re{}_beta{}_relax{}_diffuse{}".format(int(y_1), int(reynolds),
                                                                      beta, relax, diffuse)
               .replace(".", "_"))
    return dirname + "/t{}.png".format(ts)

accum_delta = 0.0
prev_delta = float('inf')
i = 0

start_t = clock()

while prev_delta > 1e-12:
    prev_delta = ie_1_ts.timestep(0.05)
    #ie_m1_ts.timestep(0.05)
    accum_delta += prev_delta
    i += 1
    if accum_delta > 0.25:
        name = "Part 3 IE Timestep at t={:.6}".format(ie_1_ts.time())
        f = plot_mesh(ie_1_ts.mesh(), name + ", u=1.0")
        savefig(figfile(i), dpi=800)
        close(f)
        #plot_mesh(ie_m1_ts.mesh(), name + ", u=-1.0")
        print(name)
        accum_delta = 0.0
    print(i, accum_delta)

tot_t = clock() - start_t

print("Total running time for {} iterations to converge: {}".format(i, tot_t))

name = "Part 3 IE Timestep at t={:.6}".format(ie_1_ts.time())
plot_mesh(ie_1_ts.mesh(), name + " u=1.0")
savefig(figfile(i), dpi=800)
#plot_mesh(ie_m1_ts.mesh(), name + " u=-1.0")
print(name)
print(i, accum_delta)

x, y = x_y_coords(ie_1_ts.mesh())
_, u_1, v_1 = to_np_array(ie_1_ts.mesh())
_, u_m1_r, v_m1_r = to_np_array(ie_m1_ts.mesh())

u_m1 = u_m1_r[::-1, :]
v_m1 = v_m1_r[::-1, :]

fig = figure(figsize=figaspect(1.0 / 3.0))
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.plot_surface(x, y, u_1, cmap=cm.gist_heat)
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_surface(x, y, u_m1, cmap=cm.gist_heat)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot_surface(x, y, u_1 + u_m1, cmap=cm.gist_heat)

fig = figure(figsize=figaspect(1.0 / 3.0))
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.plot_surface(x, y, v_1, cmap=cm.gist_heat)
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_surface(x, y, v_m1, cmap=cm.gist_heat)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot_surface(x, y, v_1 - v_m1, cmap=cm.gist_heat)

show()
