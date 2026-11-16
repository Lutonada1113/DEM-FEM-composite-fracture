# -*- coding: utf-8 -*-
"""FEM reference numerical driver for the verification benchmark.

Use run_benchmark.py from the repository root. Automatic visualization was
removed for this distribution. Model and iteration kernels are retained.
See docs/reproducibility.md for archived-result provenance.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] + sys.path
import dsolve


BASE = os.path.dirname(os.path.abspath(__file__))

# ========================== 材料与加载参数 ==========================
E11 = 171.0e3     # MPa
E22 = 9.08e3
G12 = 5.29e3
NU12 = 0.32
GC = 0.277        # N/mm (Gc22)
L = 0.01          # mm
U_MAX = 0.05      # mm
N_STEPS = 50

LAYUPS = {1: [0.0, -22.5, 0.0, 22.5], 2: [0.0, 45.0, -45.0, 90.0]}
REGION_BOUNDS = [-0.5, -0.25, 0.0, 0.25, 0.5]
BETA = {'A1': 300.0, 'A2': 300.0, 'A3': 0.2}
TENSORS = ['A1', 'A2', 'A3']
MESHES = [1, 2]

STAG_TOL = 5e-3
STAG_EPS = 1e-6
MAX_STAG = 30
STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 3


def layup_theta(layup_deg, x_centroid):
    """按单元形心 x 分区, 返回逐单元纤维角 (弧度)."""
    x = np.asarray(x_centroid, dtype=np.float64)
    angles = [np.deg2rad(a) for a in layup_deg]
    out = np.zeros_like(x)
    for i in range(4):
        m = (x >= REGION_BOUNDS[i] - 1e-12) & (x < REGION_BOUNDS[i + 1] + 1e-12)
        out[m] = angles[i]
    return out


def build_local_Q():
    nu21 = NU12 * E22 / E11
    den = 1.0 - NU12 * nu21
    return (E11 / den, E22 / den, NU12 * E22 / den, G12)


def rotate_Q(q11, q22, q12, q66, theta):
    """局部 Q -> 全局 Q (工程剪应变约定)."""
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2, cs = c * c, s * s, c * s
    Qloc = np.array([[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, q66]])
    row1 = np.stack([c2, s2, cs], axis=-1)
    row2 = np.stack([s2, c2, -cs], axis=-1)
    row3 = np.stack([-2 * cs, 2 * cs, c2 - s2], axis=-1)
    T = np.stack([row1, row2, row3], axis=-2)
    return np.einsum('...ki,...kl,...lj->...ij', T, Qloc, T)


def local_driving_components(eps, q11, q22, q12, q66, theta):
    """论文式(20)-(22), 纤维坐标系三分量."""
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2, cs = c * c, s * s, c * s
    e11 = c2 * exx + cs * gxy + s2 * eyy
    e22 = s2 * exx - cs * gxy + c2 * eyy
    g12 = -2.0 * cs * exx + 2.0 * cs * eyy + (c2 - s2) * gxy
    e11p = np.maximum(e11, 0.0)
    e22p = np.maximum(e22, 0.0)
    s11p = q11 * e11p + q12 * e22p
    s22p = q12 * e11p + q22 * e22p
    tau12 = q66 * g12
    return 0.5 * e11p * s11p, 0.5 * e22p * s22p, 0.5 * g12 * tau12


class FEMDisplacementSolver:
    """CST 平面应力求解器, 每单元 (1-d)^2 退化, 逐单元 Q."""
    def __init__(self, node_coords, tri_conn, Q_elems):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.Q = np.asarray(Q_elems, dtype=np.float64)
        self.Nn = self.node_coords.shape[0]
        self.Ne = self.tri_conn.shape[0]
        self._precompute()

    def _precompute(self):
        nc = self.node_coords
        conn = self.tri_conn
        p0, p1, p2 = nc[conn[:, 0]], nc[conn[:, 1]], nc[conn[:, 2]]
        x0, y0 = p0[:, 0], p0[:, 1]
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        self.area = np.abs(area2) / 2.0
        denom = np.where(np.abs(area2) < 1e-12, 1e-12, area2)
        dNdx = np.stack([(y1 - y2) / denom, (y2 - y0) / denom, (y0 - y1) / denom], axis=1)
        dNdy = np.stack([(x2 - x1) / denom, (x0 - x2) / denom, (x1 - x0) / denom], axis=1)
        self.centroid = np.stack([(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0], axis=1)
        B = np.zeros((self.Ne, 3, 6))
        B[:, 0, 0::2] = dNdx
        B[:, 1, 1::2] = dNdy
        B[:, 2, 0::2] = dNdy
        B[:, 2, 1::2] = dNdx
        self.B = B

    def solve(self, d_gauss, U):
        Nn, Ne = self.Nn, self.Ne
        D = np.maximum((1.0 - d_gauss) ** 2, 1e-10)
        QB = np.einsum('eij,ejk->eik', self.Q * D[:, None, None], self.B)
        Ke = np.einsum('eij,ejk->eik', self.B.transpose(0, 2, 1), QB)
        Ke = Ke * self.area[:, None, None]
        conn = self.tri_conn
        dof_map = np.zeros((Ne, 6), dtype=np.int64)
        dof_map[:, 0::2] = conn * 2
        dof_map[:, 1::2] = conn * 2 + 1
        r_all = np.repeat(dof_map, 6, axis=1).ravel()
        c_all = np.tile(dof_map, (1, 6)).ravel()
        from scipy.sparse import coo_matrix
        K = coo_matrix((Ke.reshape(-1), (r_all, c_all)), shape=(2 * Nn, 2 * Nn)).tocsr()

        fix = {}
        tol = 1e-6
        y = self.node_coords[:, 1]
        bot = np.where(np.abs(y + 0.5) < tol)[0]
        top = np.where(np.abs(y - 0.5) < tol)[0]
        for i in bot:
            fix[2 * i] = 0.0
            fix[2 * i + 1] = 0.0
        for i in top:
            fix[2 * i + 1] = U
        rhs = np.zeros(2 * Nn)
        fix_dofs = np.array(list(fix.keys()), dtype=int)
        fix_vals = np.array(list(fix.values()), dtype=float)
        K = K.tolil()
        K[fix_dofs, :] = 0.0
        K[fix_dofs, fix_dofs] = 1.0
        K = K.tocsr()
        rhs[fix_dofs] = fix_vals
        from scipy.sparse.linalg import spsolve, cg
        try:
            u = spsolve(K, rhs)
        except Exception:
            u, _ = cg(K, rhs, tol=1e-10, maxiter=10000)

        eps = self.compute_strain(u)
        sigma = np.einsum('eij,ej->ei', self.Q, eps) * D[:, None]
        force = 0.0
        for e in range(Ne):
            n0, n1, n2 = conn[e]
            for a, b in ((n0, n1), (n1, n2), (n2, n0)):
                if abs(y[a] - 0.5) < tol and abs(y[b] - 0.5) < tol:
                    force += sigma[e, 1] * abs(self.node_coords[a, 0] - self.node_coords[b, 0])
        return u, force

    def compute_strain(self, u):
        ue = np.stack([u[self.tri_conn * 2], u[self.tri_conn * 2 + 1]], axis=-1)
        ux, uy = ue[:, :, 0], ue[:, :, 1]
        B = self.B
        exx = np.sum(B[:, 0, 0::2] * ux, axis=1)
        eyy = np.sum(B[:, 1, 1::2] * uy, axis=1)
        gxy = np.sum(B[:, 2, 0::2] * ux, axis=1) + np.sum(B[:, 2, 1::2] * uy, axis=1)
        return np.stack([exx, eyy, gxy], axis=-1)


class AnisoPhaseFieldSolver:
    """AT2 相场 FEM, 逐单元纤维角, 结构张量 A1/A2/A3."""
    def __init__(self, node_coords, tri_conn, Gc, l, beta, theta_elems, tensor_type='A1'):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.Gc = Gc
        self.l = l
        self.beta = beta
        self.theta = np.asarray(theta_elems, dtype=np.float64)
        self.tensor_type = tensor_type.upper()
        self.Nn = self.node_coords.shape[0]
        self.Ne = self.tri_conn.shape[0]
        self._precompute()

    def _precompute(self):
        coords = self.node_coords
        conn = self.tri_conn
        p0 = coords[conn[:, 0]]
        p1 = coords[conn[:, 1]]
        p2 = coords[conn[:, 2]]
        x0, y0 = p0[:, 0], p0[:, 1]
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        self.area = np.maximum(np.abs(area2) / 2.0, 1e-12)
        denom = np.where(np.abs(area2) < 1e-12, 1e-12, area2)
        self.dNdx = np.stack([(y1 - y2) / denom, (y2 - y0) / denom, (y0 - y1) / denom], axis=1)
        self.dNdy = np.stack([(x2 - x1) / denom, (x0 - x2) / denom, (x1 - x0) / denom], axis=1)
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        self.dN_dot_a = c[:, None] * self.dNdx + s[:, None] * self.dNdy

    def solve(self, H_gauss):
        Nn, Ne = self.Nn, self.Ne
        Gc, l, beta = self.Gc, self.l, self.beta
        dN_dot_a = self.dN_dot_a
        grad = (self.dNdx[:, :, None] * self.dNdx[:, None, :] +
                self.dNdy[:, :, None] * self.dNdy[:, None, :])
        aniso = dN_dot_a[:, :, None] * dN_dot_a[:, None, :]
        if self.tensor_type in ('A1', 'A2'):
            K_aniso = (Gc * l) * self.area[:, None, None] * (grad + beta * aniso)
        elif self.tensor_type == 'A3':
            K_aniso = (Gc * l) * self.area[:, None, None] * (
                beta * grad + (1.0 - beta) * aniso)
        else:
            raise ValueError(self.tensor_type)
        Ni = np.full((Ne, 3), 1.0 / 3.0)
        M_e = (self.area[:, None, None] * Ni[:, :, None] * Ni[:, None, :]
               * (Gc / l + 2.0 * H_gauss)[:, None, None])
        K_e = K_aniso + M_e
        f_e = 2.0 * H_gauss[:, None] * self.area[:, None] * Ni
        conn = self.tri_conn
        rows = np.repeat(conn.ravel(), 3)
        cols = np.tile(conn, (1, 3)).ravel()
        from scipy.sparse import coo_matrix
        K = coo_matrix((K_e.reshape(-1), (rows, cols)), shape=(Nn, Nn)).tocsr()
        rhs = np.bincount(conn.ravel(), weights=f_e.ravel(), minlength=Nn)
        from scipy.sparse.linalg import spsolve, cg
        try:
            d = spsolve(K, rhs)
        except Exception:
            d, _ = cg(K, rhs, tol=1e-8, maxiter=5000)
        return np.clip(d, 0.0, 1.0)


def write_step_times(path, times):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'seconds'])
        for step in sorted(times):
            w.writerow([step, f'{times[step]:.2f}'])




def min_edge(node_coords, tri_conn):
    p0 = node_coords[tri_conn[:, 0]]
    p1 = node_coords[tri_conn[:, 1]]
    p2 = node_coords[tri_conn[:, 2]]
    e = np.minimum(np.minimum(np.linalg.norm(p1 - p0, axis=1),
                              np.linalg.norm(p2 - p1, axis=1)),
                   np.linalg.norm(p0 - p2, axis=1))
    return float(e.min())


def run_case(mesh_idx, tensor, args):
    tag = f'm{mesh_idx}_{tensor}'
    case_dir = os.path.join(BASE, f'fem_{tag}')
    os.makedirs(case_dir, exist_ok=True)
    force_file = os.path.join(case_dir, f'force_fem_{tag}.csv')
    results_file = os.path.join(case_dir, f'results_fem_{tag}.csv')
    time_file = os.path.join(case_dir, f'time_fem_{tag}.csv')

    layup = LAYUPS[mesh_idx]
    beta = BETA[tensor]
    node_coords, tri_conn = dsolve.read_gmsh_mesh(
        os.path.join(BASE, f'verification{mesh_idx}.msh'))
    Nn, Ne = node_coords.shape[0], tri_conn.shape[0]
    lm = args.lm if args.lm is not None else min_edge(node_coords, tri_conn)
    gc = GC * (1.0 - (lm / L) * (1.0 - beta / 1000.0))
    print(f'\n===== FEM: 铺层 {layup}, 张量 {tensor}, beta={beta} =====')
    print(f'网格: {Nn} 节点, {Ne} 单元;  有效 Gc: {GC:.4f} -> {gc:.4f} '
          f'N/mm (lm={lm:.2e})')

    centroid = (node_coords[tri_conn].mean(axis=1))
    theta_elems = layup_theta(layup, centroid[:, 0])
    q11, q22, q12, q66 = build_local_Q()
    Q_elems = rotate_Q(q11, q22, q12, q66, theta_elems)
    fem_solver = FEMDisplacementSolver(node_coords, tri_conn, Q_elems)
    pf_solver = AnisoPhaseFieldSolver(node_coords, tri_conn, gc, L, beta,
                                      theta_elems, tensor_type=tensor)

    d_node = np.zeros(Nn)
    d_gauss = np.zeros(Ne)
    H1 = np.zeros(Ne)
    H2 = np.zeros(Ne)
    H3 = np.zeros(Ne)
    force_max = 0.0
    stop_extra_left = None
    step_times = {}

    with open(force_file, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'U', 'force'])
    with open(results_file, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'node', 'x', 'y', 'u_x', 'u_y', 'd'])
    write_step_times(time_file, {})

    for step in range(1, args.steps + 1):
        step_t0 = time.time()
        U_cur = U_MAX * step / N_STEPS
        print(f'\n========== {tag} step {step}/{args.steps}, U={U_cur:.5f} ==========')
        d_prev_node = d_node.copy()
        u_prev = None
        failed = False
        best_metric = 1e30
        stagnate = 0
        for stag in range(1, MAX_STAG + 1):
            t0 = time.time()
            u_nodes, force_now = fem_solver.solve(d_gauss, U_cur)
            if not np.all(np.isfinite(u_nodes)):
                print(f'  [stag {stag}] 位移解发散, 停止计算')
                failed = True
                break
            eps = fem_solver.compute_strain(u_nodes)
            psi1, psi2, psi3 = local_driving_components(eps, q11, q22, q12, q66,
                                                        theta_elems)
            H1, H2, H3, H_total = dsolve.update_history(H1, H2, H3, psi1, psi2, psi3)
            d_new = pf_solver.solve(H_total)
            if not np.all(np.isfinite(d_new)):
                print(f'  [stag {stag}] 相场解发散, 停止计算')
                failed = True
                break
            d_new = np.maximum(d_new, d_prev_node)
            d_node = d_new
            d_gauss = d_new[tri_conn].mean(axis=1)
            if u_prev is None:
                du_ratio = 1.0
            else:
                du_ratio = np.linalg.norm(u_nodes - u_prev) / (np.linalg.norm(u_nodes) + STAG_EPS)
            dd_ratio = np.linalg.norm(d_new - d_prev_node) / (np.linalg.norm(d_new) + STAG_EPS)
            stag_metric = float(du_ratio + dd_ratio)
            u_prev = u_nodes.copy()
            d_prev_node = d_new.copy()
            print(f'  [stag {stag}] ||Δu||/||u||={du_ratio:.3e}, ||Δd||/||d||={dd_ratio:.3e}, '
                  f'收敛指标={stag_metric:.3e}, d_max={d_new.max():.4f} ({time.time()-t0:.1f}s)')
            if force_max > 0 and force_now < STOP_FRACTION * force_max:
                print('  反力骤降至峰值 50% 以下, 结束交错')
                break
            if stag_metric < STAG_TOL:
                break
            if stag_metric < best_metric:
                best_metric = stag_metric
                stagnate = 0
            else:
                stagnate += 1
                if stagnate >= 15:
                    print(f'  [stag {stag}] 收敛停滞 {stagnate} 次, 结束交错')
                    break
        if failed:
            break

        u_nodes, force = fem_solver.solve(d_gauss, U_cur)
        if not np.all(np.isfinite(u_nodes)) or not np.isfinite(force):
            print('  位移解发散, 停止算例')
            break
        with open(force_file, 'a', newline='') as f:
            csv.writer(f).writerow([step, U_cur, f'{force:.8f}'])
        force_max = max(force_max, force)
        print(f'  反力 F={force:.6f} N (max={force_max:.6f})')

        with open(results_file, 'a', newline='') as f:
            w = csv.writer(f)
            for i in range(Nn):
                w.writerow([step, i, node_coords[i, 0], node_coords[i, 1],
                            f'{u_nodes[2*i]:.8f}', f'{u_nodes[2*i+1]:.8f}',
                            f'{d_node[i]:.8f}'])
        step_time = time.time() - step_t0
        step_times[step] = step_time
        write_step_times(time_file, step_times)
        print(f'  本步耗时 {step_time:.1f}s, 累计 {sum(step_times.values()):.1f}s')

        if (stop_extra_left is None and force_max > 0
                and force < STOP_FRACTION * force_max):
            stop_extra_left = STOP_EXTRA_STEPS
            print(f'  反力降至峰值 {STOP_FRACTION:.0%} 以下, 再算 {STOP_EXTRA_STEPS} 步后停止')
        elif stop_extra_left is not None:
            stop_extra_left -= 1
            if stop_extra_left < 0:
                print(f'{tag}: 破坏完成, 停止 (step {step})')
                break
    print(f'{tag}: 完成, 总耗时 {sum(step_times.values()):.1f}s')
    return force_file




def main():
    parser = argparse.ArgumentParser(description='verification - 位移有限元版 (有效 Gc)')
    parser.add_argument('--mesh', type=int, choices=MESHES, default=None)
    parser.add_argument('--tensor', type=str, choices=TENSORS, default=None)
    parser.add_argument('--steps', type=int, default=N_STEPS)
    parser.add_argument('--lm', type=float, default=None,
                        help='有效 Gc 修正的最小单元尺寸 (默认取网格最小边长)')
    args = parser.parse_args()

    if args.mesh and args.tensor:
        combos = [(args.mesh, args.tensor)]
    elif args.mesh:
        combos = [(args.mesh, t) for t in TENSORS]
    else:
        combos = [(m, t) for m in MESHES for t in TENSORS]
    for m, t in combos:
        run_case(m, t, args)


if __name__ == '__main__':
    main()
