"""DEM-FEM numerical driver for the circlehole1 benchmark.

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
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dsolve

# ========================== 材料与加载参数 ==========================
E11 = 26500.0     # MPa
E22 = 2600.0
G12 = 1300.0
NU12 = 0.35
GC_I = 0.622      # N/mm
GC_II = 0.472     # N/mm
L = 0.416         # mm
LM = 0.15         # mm (有效 Gc 修正的最小网格尺寸)
BETA = 300.0
U_MAX = 2.0       # mm
N_STEPS = 50

# 几何
XMIN, XMAX = -40.0, 40.0
YMIN, YMAX = -9.0, 9.0
LX = XMAX - XMIN
LY = YMAX - YMIN
FIX_X, FIX_Y = XMIN, YMIN     # 刚体位移固定点 (左下角)

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
    base = os.path.dirname(os.path.abspath(__file__))
    k = mesh_key(mesh)
    path = os.path.join(base, MESH_FILES[k])
    if not os.path.exists(path) and abs(k - 0.15) < 1e-9:
        path = os.path.join(base, 'circlehole.msh')      # 旧文件名兜底
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到网格文件: {path}')
    return path


def case_name(angle, mesh):
    """结果文件夹名: 0.15 沿用 alpha{angle}, 其他网格加 -m{尺寸} 后缀."""
    k = mesh_key(mesh)
    if abs(k - 0.15) < 1e-9:
        return f'alpha{angle}'
    return f'alpha{angle}-m{k:g}'

# ========================== 网络与训练参数 ==========================
HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 200
INIT_M = 2.0
MAX_EPOCHS = 5000
TOL = 1e-6
PATIENCE = 15
WEIGHT_DECAY = 1e-7

MAX_STAG = 120
STAG_TOL = 5e-3
STAG_EPS = 1e-8

STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 1


def stag_hparams(step, stag):
    """按 (加载步, 交错迭代) 返回 RPROP 的 (lr, min_step, max_step)。"""
    if step == 1 and stag ==1:
        return 5e-2, 5e-6, 0.2
    if (step == 1 and stag <= 5) or stag <=3:
        return 1e-4, 1e-6, 0.1
    return 1e-6, 1e-7, 0.05


def theta_func(angle_deg):
    theta = 90.0 - angle_deg   # alpha 是与竖直方向的夹角
    def f(x, y):
        return np.full_like(np.asarray(x, dtype=np.float64), np.deg2rad(theta))
    return f


def theta_torch(angle_deg):
    theta = 90.0 - angle_deg
    def f(x, y):
        return torch.full_like(x, np.deg2rad(theta), dtype=x.dtype)
    return f


def make_bc_functions(U_cur):
    """硬约束 ansatz:
    u_x: 左端 0, 右端 U_cur (线性插值), 两端 dist 为 0
    u_y: 仅在左下角固定点 u_y=0, 其余自由
    """
    def dist_func(x):
        xx = x[:, 0]
        yy = x[:, 1]
        dist = torch.ones_like(x)
        dist[:, 0] = (xx - XMIN) * (XMAX - xx) / ((LX / 2.0) ** 2)
        dist[:, 1] = ((xx - FIX_X) / (LX / 2.0)) ** 2 + ((yy - FIX_Y) / (LY / 2.0)) ** 2
        return dist

    def u_bc_func(x):
        u_bc = torch.zeros_like(x)
        u_bc[:, 0] = U_cur * (x[:, 0] - XMIN) / LX
        return u_bc

    return dist_func, u_bc_func


class FullEnergyMaterial(dsolve.CompositeMaterial):
    """论文 2.2 节假设 psi^- = 0: 位移子问题使用完整弹性能 (1-d)^2 * 1/2 eps^T Q eps。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_theta = None
        self._cache_Q = None

    def compute_psi(self, eps, d, theta):
        if self._cache_theta is not theta:
            self._cache_theta = theta
            self._cache_Q = self.get_global_stiffness(theta)
        Q = self._cache_Q
        psi = 0.5 * torch.einsum('...i,...ij,...j->...', eps, Q, eps)
        g = (1.0 - d) ** 2
        return g * psi, torch.zeros_like(psi)


def local_driving_components(material, eps, theta):
    """论文式 (20)-(22): 在局部(纤维)坐标系下计算三个驱动能量分量。"""
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    c = torch.cos(theta)
    s = torch.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s

    e11 = c2 * exx + cs * gxy + s2 * eyy
    e22 = s2 * exx - cs * gxy + c2 * eyy
    g12 = -2.0 * cs * exx + 2.0 * cs * eyy + (c2 - s2) * gxy

    e11p = torch.relu(e11)
    e22p = torch.relu(e22)

    s11p = material.Q11 * e11p + material.Q12 * e22p
    s22p = material.Q12 * e11p + material.Q22 * e22p
    tau12 = material.Q66 * g12

    psi_teedf = 0.5 * e11p * s11p
    psi_teedm = 0.5 * e22p * s22p
    psi_tsedm = 0.5 * g12 * tau12
    return psi_teedf, psi_teedm, psi_tsedm


class AnisoPhaseFieldSolver:
    """AT2 相场有限元求解器 (A1), 有效 Gc 按 Julia 参考文件公式。"""
    def __init__(self, node_coords, tri_conn, Gc, l, beta, theta_func, lm=None,
                 use_effective_gc=True):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.l = l
        self.beta = beta
        self.theta_func = theta_func

        if use_effective_gc:
            # 有效 Gc (Julia fig10.jl): Gc_eff = Gc*(1 - lm/l)*(1 - beta/1000)
            if lm is None:
                lm = self._min_edge_length()
            self.lm = lm
            self.Gc = Gc * (1.0 - lm / l) * (1.0 - beta / 1000.0)
            print(f'  有效 Gc: {Gc:.4f} -> {self.Gc:.4f} N/mm '
                  f'(lm={lm:.3f}, l={l:.3f}, beta={beta:g})')
        else:
            self.Gc = Gc
            print(f'  使用原始 Gc: {Gc:.4f} N/mm (未做有效 Gc 修正)')

        self.Nn = self.node_coords.shape[0]
        self.Ne = self.tri_conn.shape[0]
        self._precompute()

    def _min_edge_length(self):
        p0 = self.node_coords[self.tri_conn[:, 0]]
        p1 = self.node_coords[self.tri_conn[:, 1]]
        p2 = self.node_coords[self.tri_conn[:, 2]]
        e0 = np.linalg.norm(p1 - p0, axis=1)
        e1 = np.linalg.norm(p2 - p1, axis=1)
        e2 = np.linalg.norm(p0 - p2, axis=1)
        return float(np.minimum(np.minimum(e0, e1), e2).min())

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
        self.area = np.abs(area2) / 2.0
        self.area = np.maximum(self.area, 1e-12)
        denom = np.where(np.abs(area2) < 1e-12, 1e-12, area2)

        self.dNdx = np.stack([(y1 - y2) / denom, (y2 - y0) / denom, (y0 - y1) / denom], axis=1)
        self.dNdy = np.stack([(x2 - x1) / denom, (x0 - x2) / denom, (x1 - x0) / denom], axis=1)
        self.centroid = np.stack([(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0], axis=1)

    def solve(self, H_gauss, theta_gauss=None, fixed_nodes=None, fixed_vals=None):
        if theta_gauss is None:
            theta_gauss = self.theta_func(self.centroid[:, 0], self.centroid[:, 1])
        theta_gauss = np.asarray(theta_gauss)

        Nn = self.Nn
        Ne = self.Ne
        Gc = self.Gc
        l = self.l
        beta = self.beta

        c = np.cos(theta_gauss)
        s = np.sin(theta_gauss)
        a = np.stack([c, s], axis=-1)
        dN = np.stack([self.dNdx, self.dNdy], axis=-1)
        dN_dot_a = np.einsum('eij,ej->ei', dN, a)
        grad = np.einsum('eik,ejk->eij', dN, dN)
        aniso = dN_dot_a[:, :, None] * dN_dot_a[:, None, :]

        # A1 = I + beta*M
        K_aniso = Gc * l * self.area[:, None, None] * (grad + beta * aniso)

        Ni = np.full((Ne, 3), 1.0 / 3.0)
        M_e = (self.area[:, None, None]
               * Ni[:, :, None] * Ni[:, None, :]
               * (Gc / l + 2.0 * H_gauss)[:, None, None])
        K_e = K_aniso + M_e
        f_e = 2.0 * H_gauss[:, None] * self.area[:, None] * Ni

        conn = self.tri_conn
        rows = np.repeat(conn.ravel(), 3)
        cols = np.tile(conn, (1, 3)).ravel()
        vals = K_e.reshape(-1)
        rhs = np.bincount(conn.ravel(), weights=f_e.ravel(), minlength=Nn)

        from scipy.sparse import coo_matrix
        K = coo_matrix((vals, (rows, cols)), shape=(Nn, Nn)).tocsr()

        if fixed_nodes is not None:
            fixed_nodes = np.asarray(fixed_nodes, dtype=int)
            fixed_vals = np.asarray(fixed_vals, dtype=float)
            for i, node in enumerate(fixed_nodes):
                K[node, :] = 0.0
                K[node, node] = 1.0
                rhs[node] = fixed_vals[i]

        from scipy.sparse.linalg import spsolve, cg
        try:
            d = spsolve(K, rhs)
        except Exception:
            d, _ = cg(K, rhs, tol=1e-8, maxiter=5000)
        return np.clip(d, 0.0, 1.0)


def compute_reaction_force(u_nodes, d_gauss, loss_func, material, node_coords, tri_conn):
    """右端 (x=XMAX) 合力: 沿右边界积分 sigma_xx * dGamma, sigma = (1-d)^2 Q eps。"""
    with torch.no_grad():
        eps = loss_func.compute_strain(u_nodes)
        Q = material.get_global_stiffness(loss_func.flat_theta)
        sigma = torch.einsum('...ij,...j->...i', Q, eps).cpu().numpy()
    g = (1.0 - d_gauss.cpu().numpy()) ** 2
    sigma *= g[:, None]
    force = 0.0
    for e in range(tri_conn.shape[0]):
        n0, n1, n2 = tri_conn[e]
        for a, b in ((n0, n1), (n1, n2), (n2, n0)):
            if abs(node_coords[a, 0] - XMAX) < 1e-6 and abs(node_coords[b, 0] - XMAX) < 1e-6:
                edge_len = abs(node_coords[a, 1] - node_coords[b, 1])
                force += sigma[e, 0] * edge_len
    return force


def truncate_csv(path, keep_steps_below):
    if not os.path.exists(path):
        return
    with open(path, 'r', newline='') as f:
        lines = f.readlines()
    header = lines[0] if lines else ''
    rows = [ln for ln in lines[1:] if int(ln.split(',')[0].strip()) < keep_steps_below]
    with open(path, 'w', newline='') as f:
        f.write(header)
        f.writelines(rows)
    print(f'  已截断 {os.path.basename(path)}: 保留 step < {keep_steps_below} 的 {len(rows)} 行')


def read_step_times(path):
    """读取每步耗时 {step: seconds}; 文件不存在/为空返回 {}。"""
    times = {}
    if not os.path.exists(path):
        return times
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)   # 跳过表头
            for row in reader:
                if len(row) >= 2:
                    times[int(row[0])] = float(row[1])
    except (ValueError, OSError, csv.Error):
        pass
    return times


def write_step_times(path, times):
    """写入每步耗时 CSV (表头: step,seconds)。"""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'seconds'])
        for step in sorted(times):
            w.writerow([step, f'{times[step]:.2f}'])


def run_case(args, device):
    base = os.path.dirname(os.path.abspath(__file__))
    angle = args.angle
    mesh = mesh_key(args.mesh)
    case_dir = os.path.join(base, case_name(angle, mesh))
    tag = f'alpha{angle}_A1'
    msh_file = mesh_file(mesh)
    ckpt_file = os.path.join(case_dir, f'ckpt_{tag}.pt')
    results_file = os.path.join(case_dir, f'results_{tag}.csv')
    force_file = os.path.join(case_dir, f'force_{tag}.csv')
    time_file = os.path.join(case_dir, f'time_{tag}.csv')
    os.makedirs(case_dir, exist_ok=True)

    theta_np_func = theta_func(angle)
    print(f'\n===== alpha={angle}° (theta={90 - angle}°), 网格尺寸 '
          f'{mesh:g} mm, 结果目录 {os.path.basename(case_dir)} =====')

    # ---------------- 断点 / 回退重算 ----------------
    ck = None
    rollback = args.start_step is not None
    snap_file = None
    if not args.reset:
        if rollback:
            if args.start_step > 1:
                snap_file = os.path.join(case_dir, f'ckpt_{tag}_s{args.start_step - 1:02d}.pt')
                if os.path.exists(snap_file):
                    ck = torch.load(snap_file, map_location='cpu', weights_only=False)
                    print(f'{tag}: 使用快照 {os.path.basename(snap_file)} 回退, '
                          f'从 step {args.start_step} 重算')
                elif os.path.exists(ckpt_file):
                    ck = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                    print(f'{tag}: 警告: 没有 step {args.start_step - 1} 的快照, '
                          f'使用最新断点 (step {ck["step"]}) 热启动, '
                          f'从 step {args.start_step} 重算')
                else:
                    raise FileNotFoundError(
                        f'{tag}: 没有可用于从 step {args.start_step} 回退的断点/快照')
            else:
                rollback = False
        elif os.path.exists(ckpt_file):
            ck = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            if ck['step'] >= args.steps or (ck.get('stop_extra_left', None) is not None
                                            and ck['stop_extra_left'] < 0):
                print(f'{tag}: 已完成 {ck["step"]} 步, 跳过')
                return

    # ---------------- 网格 ----------------
    node_coords_np, tri_conn_np = dsolve.read_gmsh_mesh(msh_file)
    node_coords = torch.tensor(node_coords_np, dtype=torch.float64, device=device)
    tri_conn = torch.tensor(tri_conn_np, dtype=torch.long, device=device)
    fem_data = dsolve.compute_triangle_fem_data(node_coords, tri_conn)
    Nn, Ne = node_coords_np.shape[0], tri_conn_np.shape[0]
    print(f'网格: {Nn} 节点, {Ne} 单元')

    # ---------------- 材料与相场求解器 ----------------
    material = FullEnergyMaterial(E11, E22, G12, NU12, theta_torch(angle),
                                  is_plane_strain=False)
    pf_solver = AnisoPhaseFieldSolver(node_coords_np, tri_conn_np, GC_I, L, BETA,
                                      theta_np_func, lm=LM,
                                      use_effective_gc=not args.no_effective_gc)
    theta_np = theta_np_func(fem_data['flat_gauss_pts'][:, 0].cpu().numpy(),
                             fem_data['flat_gauss_pts'][:, 1].cpu().numpy())

    # ---------------- 状态恢复 ----------------
    step_start = 1
    force_rows = []
    if ck is not None:
        step_start = ck['step'] + 1
        if rollback:
            step_start = args.start_step
            force_rows = [r for r in ck['force_rows'] if r[0] < args.start_step]
        H1, H2, H3 = ck['H1'], ck['H2'], ck['H3']
        d_node = ck['d_node']
        d_gauss = ck['d_gauss'].to(device)
        net = dsolve.DisplacementNet(hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS,
                                     init_m=INIT_M, output_scale=U_MAX).double()
        net.load_state_dict(ck['net_state'])
        net = net.to(device)
        if not rollback:
            force_rows = ck['force_rows']
        force_max = ck.get('force_max', 0.0)
        stop_extra_left = ck.get('stop_extra_left', None)
        step_times = {s: t for s, t in ck.get('step_times', {}).items() if s < step_start}
        if rollback and snap_file is None:
            force_max = 0.0
            stop_extra_left = None
            print(f'{tag}: warm-start 回退, 已重置破坏停止状态')
        print(f'从断点恢复: 上次已完成 step {ck["step"]}, 本次从 step {step_start} 继续')
    else:
        torch.manual_seed(3)
        net = dsolve.DisplacementNet(hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS,
                                     init_m=INIT_M, output_scale=U_MAX).double()
        net = net.to(device)
        d_gauss = torch.zeros(Ne, dtype=torch.float64, device=device)
        d_node = np.zeros(Nn)
        H1 = np.zeros(Ne)
        H2 = np.zeros(Ne)
        H3 = np.zeros(Ne)
        force_max = 0.0
        stop_extra_left = None
        step_times = {}

    if ck is not None:
        truncate_csv(results_file, step_start)
        truncate_csv(force_file, step_start)

    if ck is None:
        write_step_times(time_file, {})
        with open(results_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'node', 'x', 'y', 'u_x', 'u_y', 'd'])
        with open(force_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'U', 'force', 'elastic_energy'])

    # ---------------- 加载步循环 ----------------
    for step in range(step_start, args.steps + 1):
        step_t0 = time.time()
        U_cur = U_MAX * step / N_STEPS
        print(f'\n========== 加载步 {step}/{N_STEPS}, U = {U_cur:.5f} ==========')
        dist_func, u_bc_func = make_bc_functions(U_cur)
        loss_func = dsolve.DEMLoss(material, fem_data, dist_func, u_bc_func)
        d_prev_node = d_node.copy()
        with torch.no_grad():
            u_prev = net(node_coords, dist_func, u_bc_func).detach().cpu().numpy()

        n_stag = 1 if args.no_stag else MAX_STAG
        for stag in range(1, n_stag + 1):
            lr, min_step, max_step = stag_hparams(step, stag)
            t0 = time.time()
            net, _ = dsolve.train_displacement_subproblem(
                net, loss_func, d_gauss,
                max_epochs=args.epochs, lr=lr, min_step=min_step, max_step=max_step,
                tol=TOL, patience=PATIENCE, weight_decay=WEIGHT_DECAY, verbose=False)
            print(f'  [stag {stag}/{n_stag}] 位移训练完成 (lr={lr:.0e}, '
                  f'min={min_step:.0e}, max={max_step:.0e}), {time.time()-t0:.1f}s')

            with torch.no_grad():
                u_nodes = net(node_coords, dist_func, u_bc_func)
                eps = loss_func.compute_strain(u_nodes)
                psi1, psi2, psi3 = local_driving_components(material, eps, loss_func.flat_theta)
            H1, H2, H3, H_total = dsolve.update_history(
                H1, H2, H3, psi1.cpu().numpy(), psi2.cpu().numpy(),
                psi3.cpu().numpy())
            print(f'  历史场 H 最大: {H_total.max():.4e}')

            d_new = pf_solver.solve(H_total, theta_gauss=theta_np)
            d_new = np.maximum(d_new, d_prev_node)
            d_node = d_new
            d_gauss = torch.tensor(d_new[tri_conn_np].mean(axis=1),
                                   dtype=torch.float64, device=device)

            u_np = u_nodes.detach().cpu().numpy()
            du_ratio = np.linalg.norm(u_np - u_prev) / (np.linalg.norm(u_np) + STAG_EPS)
            dd_ratio = np.linalg.norm(d_new - d_prev_node) / (np.linalg.norm(d_new) + STAG_EPS)
            stag_metric = float(du_ratio + dd_ratio)
            max_dd = float(np.abs(d_new - d_prev_node).max())

            d_prev_node = d_new.copy()
            u_prev = u_np.copy()
            print(f'  相场更新完成: ||Δu||/||u||={du_ratio:.3e}, ||Δd||/||d||={dd_ratio:.3e}, '
                  f'收敛指标={stag_metric:.3e} (max|Δd|={max_dd:.3e}), d最大={d_new.max():.3f}')
            if stag_metric < STAG_TOL:
                break

        with torch.no_grad():
            u_nodes = net(node_coords, dist_func, u_bc_func)
        force = compute_reaction_force(u_nodes, d_gauss, loss_func, material,
                                       node_coords_np, tri_conn_np)
        with torch.no_grad():
            energy = float(torch.exp(loss_func.compute_potential(net, d_gauss)))
        print(f'  右端反力 F = {force:.4e}, 弹性能 = {energy:.4e}')

        with open(results_file, 'a', newline='') as f:
            w = csv.writer(f)
            un = u_nodes.cpu().numpy()
            for i in range(Nn):
                w.writerow([step, i, node_coords_np[i, 0], node_coords_np[i, 1],
                            un[i, 0], un[i, 1], d_node[i]])
        with open(force_file, 'a', newline='') as f:
            csv.writer(f).writerow([step, U_cur, force, energy])
        force_rows.append([step, U_cur, force, energy])

        if stop_extra_left is None and force_max > 0.0 and force < STOP_FRACTION * force_max:
            stop_extra_left = STOP_EXTRA_STEPS
            print(f'  反力 {force:.4e} 降至最大反力 {force_max:.4e} 的 {STOP_FRACTION:.0%} 以下, '
                  f'再加载 {STOP_EXTRA_STEPS} 步后停止')
        force_max = max(force_max, force)
        stop_now = False
        if stop_extra_left is not None:
            stop_extra_left -= 1
            if stop_extra_left < 0:
                stop_now = True

        # 计时: 每个加载步只记录一次 (重跑覆盖), 总时间 = 各步之和
        step_time = time.time() - step_t0
        step_times[step] = step_time
        state = {
            'step': step,
            'net_state': net.state_dict(),
            'H1': H1, 'H2': H2, 'H3': H3,
            'd_node': d_node, 'd_gauss': d_gauss,
            'force_rows': force_rows,
            'force_max': force_max, 'stop_extra_left': stop_extra_left,
            'step_times': step_times,
        }
        torch.save(state, ckpt_file)
        torch.save(state, os.path.join(case_dir, f'ckpt_{tag}_s{step:02d}.pt'))
        write_step_times(time_file, step_times)
        total_time = sum(step_times.values())
        print(f'  本步耗时 {step_time:.1f}s, 累计训练总时间 {total_time:.1f}s '
              f'({total_time / 60.0:.1f} min)')
        if stop_now:
            print(f'  累计训练总时间: {total_time:.1f}s ({total_time / 60.0:.1f} min)')
            print(f'  破坏判定完成, 加载终止于 step {step}')
            print(f'{tag} 完成。')
            return
        print(f'  已保存断点 (step {step})')

    print(f'  累计训练总时间: {sum(step_times.values()):.1f}s '
          f'({sum(step_times.values()) / 60.0:.1f} min)')
    print(f'{tag} 完成。')


def main():
    parser = argparse.ArgumentParser(description='圆孔板 复现 Yu 2024 图 9/10 (DEM + 相场 staggered)')
    parser.add_argument('--angle', type=int, choices=[0, 15, 30, 45, 60], default=None,
                        help='纤维角 alpha (缺省运行全部 5 个角度)')
    parser.add_argument('--mesh', type=float, default=MESH_DEFAULT,
                        choices=sorted(MESH_FILES),
                        help='网格尺寸 mm (0.1/0.15/0.2/0.3/0.5/0.8/1.2/2, 默认 0.15); '
                             '0.15 存到 alpha{angle}/, 其他存到 alpha{angle}-m{mesh}/')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--steps', type=int, default=N_STEPS, help='最多运行的加载步数')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS, help='每次位移子问题最大 epoch 数')
    parser.add_argument('--no-stag', action='store_true', help='每步只做一次位移-相场交替')
    parser.add_argument('--reset', action='store_true', help='忽略已有断点重新开始')
    parser.add_argument('--start-step', type=int, default=None,
                        help='从第 N 个加载步开始重算 (回退重算)')
    parser.add_argument('--no-effective-gc', action='store_true',
                        help='关闭有效 Gc 修正, 使用原始 Gc_I')
    args = parser.parse_args()

    if args.reset and args.start_step is not None:
        parser.error('--reset 与 --start-step 不能同时使用')
    if args.start_step is not None and not (1 <= args.start_step <= N_STEPS):
        parser.error(f'--start-step 应在 1..{N_STEPS} 之间')

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定了 --device cuda, 但 torch.cuda.is_available()=False。')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'计算设备: {device}  (torch {torch.__version__}, '
          f'cuda available: {torch.cuda.is_available()})')
    if device.type == 'cpu':
        print('警告: 正在使用 CPU 训练。如需 GPU, 请安装 CUDA 版 PyTorch 或指定 --device cuda。')

    t_all = time.time()
    angles = [args.angle] if args.angle is not None else [0, 15, 30, 45, 60]
    print(f'待运行角度: {angles}')
    for a in angles:
        args.angle = a
        run_case(args, device)
    print(f'\n完成, 总耗时 {time.time()-t_all:.1f}s')


if __name__ == '__main__':
    main()
