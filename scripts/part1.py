#!/usr/bin/python3.6

from ins_solver import Mesh_40x40, to_np_array
from ins_solver import BConds_Part1
from ins_solver import INSAssembly

from matplotlib.pyplot import figure, contour, show, clabel, title

m_zero = Mesh_40x40()
m_init = Mesh_40x40()
m_fi = Mesh_40x40()
m_fi_soln = Mesh_40x40()

bc = BConds_Part1()
sd = INSAssembly(bc)

bc.init_mesh(m_init)
sd.flux_assembly(m_zero, m_init, m_fi, 0.0, 1.0)
bc.flux_soln(m_fi_soln)

p_fi, u_fi, v_fi = to_np_array(m_fi)
p_fi_soln, u_fi_soln, v_fi_soln = to_np_array(m_fi_soln)
p_fi_err, u_fi_err, v_fi_err = (p_fi - p_fi_soln, u_fi - u_fi_soln, v_fi - v_fi_soln)

figure(); cs = contour(p_fi); clabel(cs); title("P FI Computed")
figure(); cs = contour(p_fi_soln); clabel(cs); title("P FI Solution")
figure(); cs = contour(p_fi_err); clabel(cs); title("P FI Error")

figure(); cs = contour(u_fi); clabel(cs); title("U FI Computed")
figure(); cs = contour(u_fi_soln); clabel(cs); title("U FI Solution")
figure(); cs = contour(u_fi_err); clabel(cs); title("U FI Error")

figure(); cs = contour(v_fi); clabel(cs); title("V FI Computed")
figure(); cs = contour(v_fi_soln); clabel(cs); title("V FI Solution")
figure(); cs = contour(v_fi_err); clabel(cs); title("V FI Error")

show()
