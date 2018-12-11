#!/usr/bin/python3.6

from numpy import sum, log2, sqrt

from ins_solver import Mesh_160x160, Mesh_80x80, Mesh_40x40, Mesh_20x20, Mesh_10x10
from ins_solver import RK1_160x160, RK1_80x80, RK1_40x40, RK1_20x20, RK1_10x10
from ins_solver import to_np_array, x_y_coords
from ins_solver import BConds_Part1
from ins_solver import INSAssembly

from matplotlib.pyplot import figure, contour, show, clabel, title
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_errs(MeshT, plot=True):
    m_zero = MeshT()
    m_init = MeshT()
    m_fi = MeshT()
    m_fi_soln = MeshT()

    bc = BConds_Part1(P_0 = 1.0, u_0 = 1.0, v_0 = 1.0, beta = 1.0)
    sd = INSAssembly(bc)

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
p_err_080, u_err_080, v_err_080 = plot_errs(Mesh_80x80, True)
print("80 Convergence Rate Pressure: {} u: {} v: {}".format(log2(p_err_040 / p_err_080),
                                                            log2(u_err_040 / u_err_080),
                                                            log2(v_err_040 / v_err_080)))
print("160x160")
p_err_160, u_err_160, v_err_160 = plot_errs(Mesh_160x160, True)
print("160 Convergence Rate Pressure: {} u: {} v: {}".format(log2(p_err_080 / p_err_160),
                                                             log2(u_err_080 / u_err_160),
                                                             log2(v_err_080 / v_err_160)))

show()

bc = BConds_Part1()
ts = RK1_160x160(bc)
print("Getting x, y, coordinates")
x, y = x_y_coords(ts.mesh())
print("Getting values")
p, u, v = to_np_array(ts.mesh())
m = Mesh_160x160(bc)
p_e, u_e, v_e = to_np_array(m)
print("P_0", bc.P_0())
print("u_0", bc.u_0())
print("v_0", bc.v_0())
print("Beta", bc.beta())
print("x_min", bc.x_min())
print("x_max", bc.x_max())
print("y_min", bc.y_min())
print("y_max", bc.y_max())
print("Computed", p)
print("Expected", p_e)
print("Error", p - p_e)
assembly = ts.space_assembly()
print(assembly)
bc_ref = assembly.boundaries()
print("P_0", bc_ref.P_0())
print("u_0", bc_ref.u_0())
print("v_0", bc_ref.v_0())
print("Beta", bc_ref.beta())
print("x_min", bc_ref.x_min())
print("x_max", bc_ref.x_max())
print("y_min", bc_ref.y_min())
print("y_max", bc_ref.y_max())
while ts.time() < 0.000025:
    p, u, v = to_np_array(ts.mesh())
    print(ts.time(), p)
    fig = figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x, y, p, cmap=cm.gist_heat)
    ax.contour(x, y, p)
    title("Part 1 RK1 Timestep at t={:.6}".format(ts.time()))

    ts.timestep(0.8)

show()
