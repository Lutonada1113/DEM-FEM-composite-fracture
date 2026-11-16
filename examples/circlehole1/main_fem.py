# -*- coding: utf-8 -*-
"""FEM reference numerical driver for the circlehole1 benchmark.

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

# ========================== 材料与加载参数 (论文表 4) ==========================
E11 = 26.5e3       # MPa
E22 = 2.6e3
G12 = 1.3e3
NU12 = 0.35
GC = 0.622         # N/mm (Gc_I)
L = 0.416          # mm (长度尺度)
LM_DEFAULT = 0.15  # mm (有效 Gc 修正的最小网格尺寸, DEM main.py 同值)
BETA = 300.0
U_MAX = 2.0        # mm
N_STEPS = 50

ANGLES = [0, 15, 30, 45, 60]

# ========================== 网格尺寸 ==========================
# 网格尺寸 (mm) -> msh 文件 (0.15 为原来的网格)
MESH_DEFAULT = 0.15
MESH_FILES = {
    0.1: 'circlehole-m0.1.msh',
    0.15: 'circlehole-m0.15.msh',
    0.2: 'circlehole-m0.2.msh',
    0.3: 'circlehole-m0.3.msh',
    0.5: 'circlehole-m0.5.msh',
    0.8: 'circlehole-m0.8.msh',
    1.2: 'circlehole-m1.2.msh',
    2.0: 'circlehole-m2.msh',
}


def mesh_key(mesh):
    """把命令行给的网格尺寸归一到 MESH_FILES 的键."""
    for k in MESH_FILES:
        if abs(float(mesh) - k) < 1e-9:
            return k
    raise ValueError(f'不支持的网格尺寸 {mesh} mm, 可选: {sorted(MESH_FILES)}')


def mesh_file(mesh):
    """给定网格尺寸返回 msh 文件路径 (兼容旧的 circlehole.msh)."""
    k = mesh_key(mesh)
    path = os.path.join(BASE, MESH_FILES[k])
    if not os.path.exists(path) and abs(k - 0.15) < 1e-9:
        path = os.path.join(BASE, 'circlehole.msh')      # 旧文件名兜底
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到网格文件: {path}')
    return path


def case_dir_name(angle, mesh):
    """结果文件夹名: 0.15 沿用 fem_alpha{angle}, 其他网格加 -m{尺寸} 后缀."""
    k = mesh_key(mesh)
    if abs(k - 0.15) < 1e-9:
        return f'fem_alpha{angle}'
    return f'fem_alpha{angle}-m{k:g}'

# 几何与边界
XMIN, XMAX = -40.0, 40.0
YMIN, YMAX = -9.0, 9.0
FIX_X, FIX_Y = XMIN, YMIN     # 左下角固定 u_y, 消除刚体位移
WIDTH = YMAX - YMIN           # 用于换算破坏应力 (F / (18x1) )

STAG_TOL = 5e-3
STAG_EPS = 1e-8
MAX_STAG = 120
STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 1


# ========================== 材料数据 ==========================
def build_local_Q():
    nu21 = NU12 * E22 / E11
    den = 1.0 - NU12 * nu21
    return E11 / den, E22 / den, NU12 * E22 / den, G12


def rotate_Q(q11, q22, q12, q66, theta):
    """局部 Q -> 全局 Q (工程剪应变约定), theta 为数组 (弧度)."""
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
    """论文式(20)-(22): 全局应变先旋转到纤维坐标系, 再取三个驱动分量."""
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


# ========================== 位移 FEM (CST, 平面应力) ==========================
class FEMDisplacementSolver:
    """常应变三角形平面应力求解器, 每单元 (1-d)^2 刚度退化."""

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
        B = np.zeros((self.Ne, 3, 6))
        B[:, 0, 0::2] = dNdx
        B[:, 1, 1::2] = dNdy
        B[:, 2, 0::2] = dNdy
        B[:, 2, 1::2] = dNdx
        self.B = B
        # 自由度映射与右端边界单元 (用于反力图)
        dof_map = np.zeros((self.Ne, 6), dtype=np.int64)
        dof_map[:, 0::2] = conn * 2
        dof_map[:, 1::2] = conn * 2 + 1
        self.dof_map = dof_map
        tol = 1e-6
        x, y = nc[:, 0], nc[:, 1]
        self.left_nodes = np.where(np.abs(x - XMIN) < tol)[0]
        self.right_nodes = np.where(np.abs(x - XMAX) < tol)[0]
        self.corner = int(np.argmin((x - FIX_X) ** 2 + (y - FIX_Y) ** 2))

    def solve(self, d_gauss, U):
        """d_gauss: [Ne] 单元损伤; U: 右端位移. 返回 (u_nodes[2N], force_right)."""
        Nn, Ne = self.Nn, self.Ne
        D = np.maximum((1.0 - d_gauss) ** 2, 1e-10)
        QB = np.einsum('eij,ejk->eik', self.Q * D[:, None, None], self.B)
        Ke = np.einsum('eij,ejk->eik', self.B.transpose(0, 2, 1), QB)
        Ke = Ke * self.area[:, None, None]
        dof_map = self.dof_map
        r_all = np.repeat(dof_map, 6, axis=1).ravel()
        c_all = np.tile(dof_map, (1, 6)).ravel()
        from scipy.sparse import coo_matrix
        K = coo_matrix((Ke.reshape(-1), (r_all, c_all)),
                       shape=(2 * Nn, 2 * Nn)).tocsr()

        # 边界条件: 左端 u_x=0, 右端 u_x=U, 左下角 u_y=0
        fix = {}
        for i in self.left_nodes:
            fix[2 * i] = 0.0
        for i in self.right_nodes:
            fix[2 * i] = U
        fix[2 * self.corner + 1] = 0.0

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

        # 右端反力: 沿 x=XMAX 边界对 sigma_xx 积分
        eps = self.compute_strain(u)
        sigma = np.einsum('eij,ej->ei', self.Q, eps) * D[:, None]
        conn = self.tri_conn
        x = self.node_coords[:, 0]
        tol = 1e-6
        force = 0.0
        for e in range(Ne):
            n0, n1, n2 = conn[e]
            for a, b in ((n0, n1), (n1, n2), (n2, n0)):
                if abs(x[a] - XMAX) < tol and abs(x[b] - XMAX) < tol:
                    force += sigma[e, 0] * abs(self.node_coords[a, 1]
                                               - self.node_coords[b, 1])
        return u, force

    def compute_strain(self, u):
        ue = np.stack([u[self.tri_conn * 2], u[self.tri_conn * 2 + 1]], axis=-1)
        ux, uy = ue[:, :, 0], ue[:, :, 1]
        B = self.B
        exx = np.sum(B[:, 0, 0::2] * ux, axis=1)
        eyy = np.sum(B[:, 1, 1::2] * uy, axis=1)
        gxy = (np.sum(B[:, 2, 0::2] * ux, axis=1)
               + np.sum(B[:, 2, 1::2] * uy, axis=1))
        return np.stack([exx, eyy, gxy], axis=-1)


# ========================== 相场求解器 (AT2, 结构张量 A1) ==========================
class AnisoPhaseFieldSolver:
    """AT2 相场 FEM, theta 全场常数, 结构张量 A1 = I + beta*(a⊗a)."""

    def __init__(self, node_coords, tri_conn, Gc, l, beta, theta):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.Gc = Gc
        self.l = l
        self.beta = beta
        self.theta = float(theta)
        self.Nn = self.node_coords.shape[0]
        self.Ne = self.tri_conn.shape[0]
        self._precompute()

    def _precompute(self):
        coords = self.node_coords
        conn = self.tri_conn
        p0, p1, p2 = coords[conn[:, 0]], coords[conn[:, 1]], coords[conn[:, 2]]
        x0, y0 = p0[:, 0], p0[:, 1]
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        self.area = np.maximum(np.abs(area2) / 2.0, 1e-12)
        denom = np.where(np.abs(area2) < 1e-12, 1e-12, area2)
        self.dNdx = np.stack([(y1 - y2) / denom, (y2 - y0) / denom, (y0 - y1) / denom], axis=1)
        self.dNdy = np.stack([(x2 - x1) / denom, (x0 - x2) / denom, (x1 - x0) / denom], axis=1)
        self.dN_dot_a = (np.cos(self.theta) * self.dNdx
                         + np.sin(self.theta) * self.dNdy)
        self.grad = (self.dNdx[:, :, None] * self.dNdx[:, None, :]
                     + self.dNdy[:, :, None] * self.dNdy[:, None, :])
        self.aniso = self.dN_dot_a[:, :, None] * self.dN_dot_a[:, None, :]
        self.Ni = np.full((self.Ne, 3), 1.0 / 3.0)
        self.rows = np.repeat(self.tri_conn.ravel(), 3)
        self.cols = np.tile(self.tri_conn, (1, 3)).ravel()
        self.grad_aniso = (self.grad + self.beta * self.aniso) * self.area[:, None, None]

    def solve(self, H_gauss):
        Nn, Ne = self.Nn, self.Ne
        Gc, l = self.Gc, self.l
        K_aniso = (Gc * l) * self.grad_aniso
        M_e = (self.area[:, None, None] * self.Ni[:, :, None] * self.Ni[:, None, :]
               * (Gc / l + 2.0 * H_gauss)[:, None, None])
        K_e = K_aniso + M_e
        f_e = 2.0 * H_gauss[:, None] * self.area[:, None] * self.Ni
        from scipy.sparse import coo_matrix
        K = coo_matrix((K_e.reshape(-1), (self.rows, self.cols)),
                       shape=(Nn, Nn)).tocsr()
        rhs = np.bincount(self.tri_conn.ravel(), weights=f_e.ravel(), minlength=Nn)
        from scipy.sparse.linalg import spsolve, cg
        try:
            d = spsolve(K, rhs)
        except Exception:
            d, _ = cg(K, rhs, tol=1e-8, maxiter=5000)
        return np.clip(d, 0.0, 1.0)


# ========================== 输出工具 ==========================
def write_step_times(path, times):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'seconds'])
        for step in sorted(times):
            w.writerow([step, f'{times[step]:.2f}'])




# ========================== 单个角度算例 ==========================
def run_case(angle, args):
    mesh = mesh_key(args.mesh)
    case_dir = os.path.join(BASE, case_dir_name(angle, mesh))
    os.makedirs(case_dir, exist_ok=True)
    force_file = os.path.join(case_dir, f'force_fem_alpha{angle}_A1.csv')
    results_file = os.path.join(case_dir, f'results_fem_alpha{angle}_A1.csv')
    time_file = os.path.join(case_dir, f'time_fem_alpha{angle}_A1.csv')

    # Gc: 默认原始值; 可选有效 Gc 修正
    lm = args.lm if args.lm is not None else LM_DEFAULT
    if args.use_effective_gc:
        gc = GC * (1.0 - (lm / L) * (1.0 - BETA / 1000.0))
        print(f'  有效 Gc: {GC:.4f} -> {gc:.4f} N/mm (lm={lm:.3f}, beta={BETA:g})')
    else:
        gc = GC
        print(f'  使用原始 Gc: {GC:.4f} N/mm (不做能量修正)')

    theta_deg = 90.0 - angle          # alpha 与竖直方向的夹角
    theta = np.deg2rad(theta_deg)
    print(f'\n===== FEM alpha={angle}deg (纤维与 x 轴夹角 theta={theta_deg:g}deg), '
          f'l={L}, beta={BETA:g}, 网格尺寸 {mesh:g} mm, '
          f'结果目录 {os.path.basename(case_dir)} =====')

    node_coords, tri_conn = dsolve.read_gmsh_mesh(mesh_file(mesh))
    Nn, Ne = node_coords.shape[0], tri_conn.shape[0]
    print(f'网格: {Nn} 节点, {Ne} 单元')

    q11, q22, q12, q66 = build_local_Q()
    Q_elems = rotate_Q(q11, q22, q12, q66, np.full(Ne, theta))
    fem_solver = FEMDisplacementSolver(node_coords, tri_conn, Q_elems)
    pf_solver = AnisoPhaseFieldSolver(node_coords, tri_conn, gc, L, BETA, theta)

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
        print(f'\n========== alpha{angle} step {step}/{args.steps}, U={U_cur:.5f} mm ==========')
        d_prev_node = d_node.copy()
        u_prev = None
        failed = False
        converged = False
        for stag in range(1, MAX_STAG + 1):
            t0 = time.time()
            u_nodes, force_now = fem_solver.solve(d_gauss, U_cur)
            if not np.all(np.isfinite(u_nodes)):
                print(f'  [stag {stag}] 位移解发散, 停止计算')
                failed = True
                break
            eps = fem_solver.compute_strain(u_nodes)
            psi1, psi2, psi3 = local_driving_components(
                eps, q11, q22, q12, q66, np.full(Ne, theta))
            H1, H2, H3, H_total = dsolve.update_history(H1, H2, H3,
                                                       psi1, psi2, psi3)
            d_new = pf_solver.solve(H_total)
            if not np.all(np.isfinite(d_new)):
                print(f'  [stag {stag}] 相场解发散, 停止计算')
                failed = True
                break
            d_new = np.maximum(d_new, d_prev_node)      # 不可逆
            d_node = d_new
            d_gauss = d_new[tri_conn].mean(axis=1)

            if u_prev is None:
                du_ratio = 1.0
            else:
                du_ratio = (np.linalg.norm(u_nodes - u_prev)
                            / (np.linalg.norm(u_nodes) + STAG_EPS))
            dd_ratio = (np.linalg.norm(d_new - d_prev_node)
                        / (np.linalg.norm(d_new) + STAG_EPS))
            stag_metric = float(du_ratio + dd_ratio)
            u_prev = u_nodes.copy()
            d_prev_node = d_new.copy()
            print(f'  [stag {stag}] ||du||/||u||={du_ratio:.3e}, '
                  f'||dd||/||d||={dd_ratio:.3e}, 收敛指标={stag_metric:.3e}, '
                  f'd_max={d_new.max():.4f} ({time.time()-t0:.1f}s)')
            # 不做任何提前终止: 既不看"反力降到峰值 50% 以下",
            # 也不看"收敛停滞 N 次"。不满足收敛判据就一直交错到
            # MAX_STAG (默认 120) 次为止。
            if stag_metric < STAG_TOL:
                converged = True
                break
        if failed:
            break
        if not converged:
            print(f'  本步交错迭代跑满 {MAX_STAG} 次仍未满足收敛判据 '
                  f'(指标 {stag_metric:.3e}), 继续下一步')

        u_nodes, force = fem_solver.solve(d_gauss, U_cur)
        if not np.all(np.isfinite(u_nodes)) or not np.isfinite(force):
            print('  位移解发散, 停止算例')
            break
        with open(force_file, 'a', newline='') as f:
            csv.writer(f).writerow([step, U_cur, f'{force:.8f}'])
        force_max = max(force_max, force)
        print(f'  右端反力 F={force:.6f} N (峰值 {force_max:.6f} N, '
              f'破坏应力 {force_max / WIDTH:.4f} MPa)')

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

        # 整例停止判据: 默认反力降到峰值 50% 以下后再算 1 步即停;
        # 加 --no-fracture-stop 可关掉它, 一直算到 --steps 指定的步数
        if not args.no_fracture_stop:
            if (stop_extra_left is None and force_max > 0
                    and force < STOP_FRACTION * force_max):
                stop_extra_left = STOP_EXTRA_STEPS
                print(f'  反力降至峰值 {STOP_FRACTION:.0%} 以下, '
                      f'再算 {STOP_EXTRA_STEPS} 步后停止')
            elif stop_extra_left is not None:
                stop_extra_left -= 1
                if stop_extra_left < 0:
                    print(f'alpha{angle}: 破坏完成, 停止 (step {step})')
                    break
    print(f'alpha{angle}: 完成, 总耗时 {sum(step_times.values()):.1f}s '
          f'({sum(step_times.values()) / 60.0:.1f} min)')
    return force_file

def main():
    parser = argparse.ArgumentParser(
        description='圆孔板 - 位移有限元版 (默认原始 Gc, 无能量修正)')
    parser.add_argument('--angle', type=int, choices=ANGLES, default=None,
                        help='纤维角 alpha (缺省运行全部 5 个角度)')
    parser.add_argument('--mesh', type=float, default=MESH_DEFAULT,
                        choices=sorted(MESH_FILES),
                        help='网格尺寸 mm (0.1/0.15/0.2/0.3/0.5/0.8/1.2/2, 默认 0.15); '
                             '0.15 存到 fem_alpha{angle}/, 其他存到 '
                             'fem_alpha{angle}-m{mesh}/')
    parser.add_argument('--steps', type=int, default=N_STEPS,
                        help='最多运行的加载步数')
    parser.add_argument('--lm', type=float, default=None,
                        help=f'有效 Gc 修正的最小网格尺寸 (默认 {LM_DEFAULT} mm)')
    parser.add_argument('--use-effective-gc', action='store_true',
                        help='启用有效临界能量释放率修正 (默认关闭)')
    parser.add_argument('--no-fracture-stop', action='store_true',
                        help='反力降到峰值 50%% 以下也不停止, 一直算到 --steps 指定的步数')
    args = parser.parse_args()

    angles = [args.angle] if args.angle is not None else ANGLES
    print(f'待运行角度: {angles}, 网格尺寸 {mesh_key(args.mesh):g} mm')
    t_all = time.time()
    for a in angles:
        run_case(a, args)
    print(f'\n完成, 总耗时 {time.time()-t_all:.1f}s')


if __name__ == '__main__':
    main()
