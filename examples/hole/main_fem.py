# -*- coding: utf-8 -*-
"""FEM reference numerical driver for the hole benchmark.

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

# ========================== 材料与加载参数 (论文表 6) ==========================
E11 = 47.4e3       # MPa
E22 = 16.2e3
G12 = 7.0e3
NU12 = 0.26
GC = 0.622         # N/mm (单一 Gc, 不用有效 Gc 修正)
L = 0.4            # mm (长度尺度, 用户指定)
BETA = 300.0
THETA = np.deg2rad(90.0)   # 纤维方向与 x 轴夹角 = 90° (竖直)
U_MAX = 1.0        # mm
N_STEPS = 100

# 几何与边界
XMIN, XMAX = -40.0, 40.0
YMIN, YMAX = -9.0, 9.0
FIX_X, FIX_Y = XMIN, YMIN     # 左下角固定 u_y, 消除刚体位移

CASES = ['C', 'S', 'T']
MSH = {c: os.path.join(BASE, f'{c}.msh') for c in CASES}
HOLE_NAME = {'C': 'Circle R=2.5', 'S': 'Square a=4.432', 'T': 'Triangle'}

STAG_TOL = 5e-3
STAG_EPS = 1e-8
MAX_STAG = 120
STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 3


def write_step_times(path, times):
    """写入每步耗时 CSV (表头: step,seconds); 每步后整表重写, 保证与当前步数一致."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'seconds'])
        for step in sorted(times):
            w.writerow([step, f'{times[step]:.2f}'])




# ========================== 材料数据 ==========================
def build_material_data():
    nu21 = NU12 * E22 / E11
    den = 1.0 - NU12 * nu21
    q11, q22, q12, q66 = E11 / den, E22 / den, NU12 * E22 / den, G12
    # theta=90° 的全局 Q (T^T Q_loc T): 1<->2 互换, 剪切不变
    Q = np.array([[q22, q12, 0.0], [q12, q11, 0.0], [0.0, 0.0, q66]])
    return dict(Q=Q, q11=q11, q22=q22, q12=q12, q66=q66)


def local_driving_components(eps, q):
    """论文式(20)-(22), theta=90° (纤维竖直).
    全局应变 -> 纤维坐标: e11=eyy, e22=exx, g12=-gxy (工程剪应变)."""
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    e11 = eyy
    e22 = exx
    g12 = -gxy
    e11p = np.maximum(e11, 0.0)
    e22p = np.maximum(e22, 0.0)
    s11p = q['q11'] * e11p + q['q12'] * e22p
    s22p = q['q12'] * e11p + q['q22'] * e22p
    tau12 = q['q66'] * g12
    psi_teedf = 0.5 * e11p * s11p
    psi_teedm = 0.5 * e22p * s22p
    psi_tsedm = 0.5 * g12 * tau12
    return psi_teedf, psi_teedm, psi_tsedm


# ========================== 位移 FEM (CST) ==========================
class FEMDisplacementSolver:
    """常应变三角形平面应力线弹性求解器, 每单元 (1-d)^2 刚度退化."""
    def __init__(self, node_coords, tri_conn, Q):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.Q = np.asarray(Q, dtype=np.float64)
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
        B = np.zeros((self.Ne, 3, 6))
        B[:, 0, 0::2] = dNdx
        B[:, 1, 1::2] = dNdy
        B[:, 2, 0::2] = dNdy
        B[:, 2, 1::2] = dNdx
        self.B = B

    def solve(self, d_gauss, U):
        """d_gauss: [Ne] 单元损伤; U: 右端位移. 返回 (u_nodes[2N], force_right)."""
        Nn, Ne = self.Nn, self.Ne
        D = np.maximum((1.0 - d_gauss) ** 2, 1e-10)
        QB = np.einsum('ij,ejk->eik', self.Q, self.B) * D[:, None, None]
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

        # 边界条件: 左端 u_x=0, 右端 u_x=U, 左下角 u_y=0
        fix = {}
        tol = 1e-6
        x = self.node_coords[:, 0]
        y = self.node_coords[:, 1]
        left = np.where(np.abs(x - XMIN) < tol)[0]
        right = np.where(np.abs(x - XMAX) < tol)[0]
        for i in left:
            fix[2 * i] = 0.0
        for i in right:
            fix[2 * i] = U
        corner = int(np.argmin((x - FIX_X) ** 2 + (y - FIX_Y) ** 2))
        fix[2 * corner + 1] = 0.0

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

        # 右端反力 (sigma_xx 沿右边界积分)
        eps = self.compute_strain(u)
        sigma = np.einsum('ij,ej->ei', self.Q, eps) * D[:, None]
        force = 0.0
        for e in range(Ne):
            n0, n1, n2 = conn[e]
            for a, b in ((n0, n1), (n1, n2), (n2, n0)):
                if abs(x[a] - XMAX) < tol and abs(x[b] - XMAX) < tol:
                    force += sigma[e, 0] * abs(y[a] - y[b])
        return u, force

    def compute_strain(self, u):
        ue = np.stack([u[self.tri_conn * 2], u[self.tri_conn * 2 + 1]], axis=-1)
        ux, uy = ue[:, :, 0], ue[:, :, 1]
        B = self.B
        exx = np.sum(B[:, 0, 0::2] * ux, axis=1)
        eyy = np.sum(B[:, 1, 1::2] * uy, axis=1)
        gxy = np.sum(B[:, 2, 0::2] * ux, axis=1) + np.sum(B[:, 2, 1::2] * uy, axis=1)
        return np.stack([exx, eyy, gxy], axis=-1)


# ========================== 相场求解器 (AT2, A1 各向异性) ==========================
class AnisoPhaseFieldSolver:
    """AT2 相场 FEM, theta=90° (a=(0,1)), Gc/beta 全场常数."""
    def __init__(self, node_coords, tri_conn, Gc, l, beta):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.l = l
        self.Gc = Gc
        self.beta = beta
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

    def solve(self, H_gauss):
        Nn, Ne = self.Nn, self.Ne
        Gc, l, beta = self.Gc, self.l, self.beta
        dN = np.stack([self.dNdx, self.dNdy], axis=-1)
        dN_dot_a = self.dNdy      # theta=90°, a=(0,1)
        grad = np.einsum('eik,ejk->eij', dN, dN)
        aniso = dN_dot_a[:, :, None] * dN_dot_a[:, None, :]
        K_aniso = (Gc * l) * self.area[:, None, None] * (
            grad + beta * aniso)
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


# ========================== 绘图 ==========================


# ========================== 主流程 ==========================
def run_case(case, args):
    case_dir = os.path.join(BASE, f'fem_{case}')
    os.makedirs(case_dir, exist_ok=True)
    force_file = os.path.join(case_dir, f'force_fem_{case}.csv')
    results_file = os.path.join(case_dir, f'results_fem_{case}.csv')
    time_file = os.path.join(case_dir, f'time_fem_{case}.csv')

    print(f'\n===== FEM: {HOLE_NAME[case]} 孔, Gc={GC} N/mm, l={L}, beta={BETA} =====')
    node_coords, tri_conn = dsolve.read_gmsh_mesh(MSH[case])
    Nn, Ne = node_coords.shape[0], tri_conn.shape[0]
    print(f'网格: {Nn} 节点, {Ne} 单元')

    q = build_material_data()
    fem_solver = FEMDisplacementSolver(node_coords, tri_conn, q['Q'])
    pf_solver = AnisoPhaseFieldSolver(node_coords, tri_conn, GC, L, BETA)

    d_gauss = np.zeros(Ne)
    d_node = np.zeros(Nn)
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
        print(f'\n========== {case} step {step}/{args.steps}, U={U_cur:.5f} ==========')
        d_prev_node = d_node.copy()
        u_prev = None

        converged = False
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
            psi1, psi2, psi3 = local_driving_components(eps, q)
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
                print(f'  [stag {stag}] 反力骤降至峰值 {STOP_FRACTION:.0%} 以下, 结束交错')
                converged = True
                break
            if stag_metric < STAG_TOL:
                converged = True
                break
            if stag_metric < best_metric:
                best_metric = stag_metric
                stagnate = 0
            else:
                stagnate += 1
                if stagnate >= 15:
                    print(f'  [stag {stag}] 收敛指标停滞 {stagnate} 次, 结束交错')
                    converged = True
                    break
        if failed:
            break

        u_nodes, force = fem_solver.solve(d_gauss, U_cur)
        if not np.all(np.isfinite(u_nodes)) or not np.isfinite(force):
            print('  最终位移解发散, 停止算例')
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
        total_time = sum(step_times.values())
        print(f'  本步耗时 {step_time:.1f}s, 累计总耗时 {total_time:.1f}s '
              f'({total_time / 60.0:.1f} min)')

        # 破坏停止判据: 反力降至峰值 50% 以下后再算 3 步
        if (stop_extra_left is None and force_max > 0
                and force < STOP_FRACTION * force_max):
            stop_extra_left = STOP_EXTRA_STEPS
            print(f'  反力 {force:.2f} 降至峰值 {force_max:.2f} 的 {STOP_FRACTION:.0%} 以下, '
                  f'再算 {STOP_EXTRA_STEPS} 步后停止')
        elif stop_extra_left is not None:
            stop_extra_left -= 1
            if stop_extra_left < 0:
                print(f'{case}: 破坏完成, 停止 (step {step}), '
                      f'总耗时 {sum(step_times.values()):.1f}s')
                break
    print(f'{case}: 完成, 累计总耗时 {sum(step_times.values()):.1f}s '
          f'({sum(step_times.values()) / 60.0:.1f} min), 记录于 {os.path.basename(time_file)}')
    return force_file




def main():
    parser = argparse.ArgumentParser(description='三种孔形板 - 位移 FEM 版')
    parser.add_argument('--case', choices=CASES, default=None)
    parser.add_argument('--steps', type=int, default=N_STEPS)
    args = parser.parse_args()

    cases = [args.case] if args.case else CASES
    for case in cases:
        run_case(case, args)


if __name__ == '__main__':
    main()
