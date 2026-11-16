# -*- coding: utf-8 -*-
"""FEM reference numerical driver for the interface benchmark.

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
E11 = 171.42e3      # MPa
E22 = 9.08e3
G12 = 5.29e3
NU12 = 0.32
E_IF = 9.08e3       # 界面各向同性
NU_IF = 0.32
G_IF = E_IF / (2.0 * (1.0 + NU_IF))
GC_INTRA = 0.277    # N/mm (Gc22)
GC_IF = {'weak': 0.0277, 'strong1': 0.277, 'strong2': 2.77}
L = 0.0065
BETA_INTRA = 300.0
BETA_IF = 0.0
U_MAX = 0.075
N_STEPS = 50

# ========================== 网格 ==========================
# 用户上传的 inp 已含几何开缝 (楔形窄缝), 直接转换得到 interface_new.msh
MSH_FILE = os.path.join(BASE, 'interface_new.msh')

STAG_TOL = 1e-3
STAG_EPS = 1e-8
MAX_STAG = 120
STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 2

CASES = ['weak', 'strong1', 'strong2']
# 是否在反力降至峰值一定比例后停止: weak 需要跑满全部加载步, strong 断后停止
STOP_ON_FORCE_DROP = {'weak': False, 'strong1': True, 'strong2': True}


# ========================== 材料数据 (按区域) ==========================
def build_material_data(node_coords, tri_conn):
    p0 = node_coords[tri_conn[:, 0]]
    p1 = node_coords[tri_conn[:, 1]]
    p2 = node_coords[tri_conn[:, 2]]
    centroid = (p0 + p1 + p2) / 3.0
    region = np.where(np.abs(centroid[:, 0]) <= 0.01, 1, 0)

    nu21 = NU12 * E22 / E11
    den = 1.0 - NU12 * nu21
    q11, q22, q12, q66 = E11 / den, E22 / den, NU12 * E22 / den, G12
    den_if = 1.0 - NU_IF ** 2
    q11i = q22i = E_IF / den_if
    q12i = NU_IF * E_IF / den_if
    q66i = G_IF

    Q = np.zeros((tri_conn.shape[0], 3, 3))
    m = region == 0
    i = region == 1
    Q[m] = [[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, q66]]
    Q[i] = [[q11i, q12i, 0.0], [q12i, q22i, 0.0], [0.0, 0.0, q66i]]
    Gc = np.where(region == 0, GC_INTRA, GC_IF['weak'])
    beta = np.where(region == 0, BETA_INTRA, BETA_IF)
    return dict(Q=Q, Gc=Gc, beta=beta, region=region, centroid=centroid)


def local_driving_components(eps, Q_elems):
    """论文式(20)-(22), theta=0; 界面元素 Q 为各向同性自动退化为拉伸分解."""
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    e11p = np.maximum(exx, 0.0)
    e22p = np.maximum(eyy, 0.0)
    s11p = Q_elems[:, 0, 0] * e11p + Q_elems[:, 0, 1] * e22p
    s22p = Q_elems[:, 1, 0] * e11p + Q_elems[:, 1, 1] * e22p
    tau12 = Q_elems[:, 2, 2] * gxy
    psi_teedf = 0.5 * e11p * s11p
    psi_teedm = 0.5 * e22p * s22p
    psi_tsedm = 0.5 * gxy * tau12
    return psi_teedf, psi_teedm, psi_tsedm


# ========================== 位移 FEM (CST) ==========================
class FEMDisplacementSolver:
    """常应变三角形平面应力线弹性求解器, 支持每单元 (1-d)^2 刚度退化."""
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
        # B 矩阵 [3, 6] 每单元
        B = np.zeros((self.Ne, 3, 6))
        B[:, 0, 0::2] = dNdx
        B[:, 1, 1::2] = dNdy
        B[:, 2, 0::2] = dNdy
        B[:, 2, 1::2] = dNdx
        self.B = B
        self.area2 = area2

    def solve(self, d_gauss, U, fix_x_node=None):
        """d_gauss: [Ne] 单元损伤; U: 上边位移. 返回 (u_nodes[2N], force_top)."""
        Nn, Ne = self.Nn, self.Ne
        # 刚度下限, 避免 d=1 单元使刚度矩阵严格奇异
        D = np.maximum((1.0 - d_gauss) ** 2, 1e-10)
        # K_e = area * B^T ((1-d)^2 Q) B
        QB = np.einsum('eij,ejk->eik', self.Q * D[:, None, None], self.B)
        Ke = np.einsum('eij,ejk->eik', self.B.transpose(0, 2, 1), QB)
        Ke = Ke * self.area[:, None, None]

        conn = self.tri_conn
        # 展开到 2N DOF
        dof_map = np.zeros((Ne, 6), dtype=np.int64)
        dof_map[:, 0::2] = conn * 2
        dof_map[:, 1::2] = conn * 2 + 1
        r_all = np.repeat(dof_map, 6, axis=1).ravel()
        c_all = np.tile(dof_map, (1, 6)).ravel()
        v_all = Ke.reshape(-1)
        from scipy.sparse import coo_matrix
        K = coo_matrix((v_all, (r_all, c_all)), shape=(2 * Nn, 2 * Nn)).tocsr()

        # 边界条件
        fix = {}
        tol = 1e-6
        y = self.node_coords[:, 1]
        bot = np.where(np.abs(y + 0.5) < tol)[0]
        top = np.where(np.abs(y - 0.5) < tol)[0]
        for i in bot:
            fix[2 * i + 1] = 0.0
        for i in top:
            fix[2 * i + 1] = U
        if fix_x_node is None:
            fix_x_node = int(bot[np.argmin(np.abs(self.node_coords[bot, 0]))])
        fix[2 * fix_x_node] = 0.0

        rhs = np.zeros(2 * Nn)
        fix_dofs = np.array(list(fix.keys()), dtype=int)
        fix_vals = np.array(list(fix.values()), dtype=float)
        # 只清零固定行 (保留列), 固定位移对内部方程的影响保留在 K 中 (spsolve 可解非对称)
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

        # 上边反力
        eps = self.compute_strain(u)
        sigma = np.einsum('eij,ej->ei', self.Q, eps) * D[:, None]
        force = 0.0
        nc = self.node_coords
        for e in range(Ne):
            n0, n1, n2 = conn[e]
            for a, b in ((n0, n1), (n1, n2), (n2, n0)):
                if abs(nc[a, 1] - 0.5) < tol and abs(nc[b, 1] - 0.5) < tol:
                    force += sigma[e, 1] * abs(nc[a, 0] - nc[b, 0])
        return u, force

    def compute_strain(self, u):
        ue = np.stack([u[self.tri_conn * 2], u[self.tri_conn * 2 + 1]], axis=-1)
        ux, uy = ue[:, :, 0], ue[:, :, 1]
        B = self.B
        exx = np.sum(B[:, 0, 0::2] * ux, axis=1)
        eyy = np.sum(B[:, 1, 1::2] * uy, axis=1)
        gxy = np.sum(B[:, 2, 0::2] * ux, axis=1) + np.sum(B[:, 2, 1::2] * uy, axis=1)
        return np.stack([exx, eyy, gxy], axis=-1)


# ========================== 相场求解器 (Gc/beta 随空间变化) ==========================
class InterfacePhaseFieldSolver:
    """AT2 相场 FEM, 每单元 Gc 与 beta (界面 beta=0 各向同性)."""
    def __init__(self, node_coords, tri_conn, Gc_elems, l, beta_elems):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.l = l
        self.Gc = np.asarray(Gc_elems, dtype=np.float64)
        self.beta = np.asarray(beta_elems, dtype=np.float64)
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
        dN_dot_a = self.dNdx   # theta=0, a=(1,0)
        grad = np.einsum('eik,ejk->eij', dN, dN)
        aniso = dN_dot_a[:, :, None] * dN_dot_a[:, None, :]
        K_aniso = (Gc * l)[:, None, None] * self.area[:, None, None] * (
            grad + beta[:, None, None] * aniso)
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

    print(f'\n===== FEM: {case}, 界面 Gc={GC_IF[case]} N/mm =====')
    node_coords, tri_conn = dsolve.read_gmsh_mesh(MSH_FILE)
    Nn, Ne = node_coords.shape[0], tri_conn.shape[0]
    print(f'网格: {Nn} 节点, {Ne} 单元')

    mdata = build_material_data(node_coords, tri_conn)
    mdata['Gc'] = np.where(mdata['region'] == 0, GC_INTRA, GC_IF[case])

    fem_solver = FEMDisplacementSolver(node_coords, tri_conn, mdata['Q'])
    pf_solver = InterfacePhaseFieldSolver(node_coords, tri_conn, mdata['Gc'], L, mdata['beta'])

    d_node = np.zeros(Nn)
    d_gauss = np.zeros(Ne)
    H1 = np.zeros(Ne)
    H2 = np.zeros(Ne)
    H3 = np.zeros(Ne)
    force_max = 0.0
    stop_extra_left = None

    with open(force_file, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'U', 'force'])
    with open(results_file, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'node', 'x', 'y', 'u_x', 'u_y', 'd'])

    for step in range(1, args.steps + 1):
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
                print(f'  [stag {stag}] 位移解发散 (裂纹贯穿/刚度奇异), 停止计算')
                failed = True
                break
            eps = fem_solver.compute_strain(u_nodes)
            psi1, psi2, psi3 = local_driving_components(eps, mdata['Q'])
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
            # 结构失效: 裂纹贯穿且反力骤降, 提前结束交错
            if force_max > 0 and force_now < STOP_FRACTION * force_max:
                print(f'  [stag {stag}] 反力骤降至峰值 {STOP_FRACTION:.0%} 以下 (结构失效), 结束交错')
                converged = True
                break
            if stag_metric < STAG_TOL:
                converged = True
                break
            # 不稳定裂纹扩展时指标可能振荡, 若长时间不改善则接受当前解
            if stag_metric < best_metric:
                best_metric = stag_metric
                stagnate = 0
            else:
                stagnate += 1
                if stagnate >= 15:
                    print(f'  [stag {stag}] 收敛指标停滞 {stagnate} 次 (不稳定扩展), 结束交错')
                    converged = True
                    break
        if failed:
            break   # 停止整个算例

        u_nodes, force = fem_solver.solve(d_gauss, U_cur)
        if not np.all(np.isfinite(u_nodes)) or not np.isfinite(force):
            print('  最终位移解发散 (结构完全破坏), 停止算例')
            break
        force_rows = [step, U_cur, force]
        with open(force_file, 'a', newline='') as f:
            csv.writer(f).writerow([step, U_cur, f'{force:.8f}'])
        force_max = max(force_max, force)
        print(f'  反力 F={force:.6f} N (max={force_max:.6f})')


        with open(results_file, 'a', newline='') as f:
            w = csv.writer(f)
            for i in range(Nn):
                w.writerow([step, i, node_coords[i, 0], node_coords[i, 1],
                            f'{u_nodes[2*i]:.8f}', f'{u_nodes[2*i+1]:.8f}', f'{d_node[i]:.8f}'])

        # 破坏停止判据: 仅对 strong 工况, 反力降至峰值 50% 以下后停止
        if (stop_extra_left is None and force_max > 0
                and STOP_ON_FORCE_DROP[case]
                and force < STOP_FRACTION * force_max):
            stop_extra_left = STOP_EXTRA_STEPS
            print(f'  反力 {force:.2f} 降至峰值 {force_max:.2f} 的 {STOP_FRACTION:.0%} 以下, '
                  f'再算 {STOP_EXTRA_STEPS} 步后停止')
        elif stop_extra_left is not None:
            stop_extra_left -= 1
            if stop_extra_left < 0:
                print(f'{case}: 破坏完成, 停止 (step {step})')
                break
    return force_file


def main():
    parser = argparse.ArgumentParser(description='interface 算例 - 位移 FEM 版')
    parser.add_argument('--case', choices=CASES, default=None)
    parser.add_argument('--steps', type=int, default=N_STEPS)
    args = parser.parse_args()

    cases = [args.case] if args.case else CASES
    for case in cases:
        run_case(case, args)

    # 三合一载荷-位移图


if __name__ == '__main__':
    main()
