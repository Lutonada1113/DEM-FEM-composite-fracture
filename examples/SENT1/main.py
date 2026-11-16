"""DEM-FEM numerical driver for the SENT1 benchmark.

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
E11 = 114.8e3      # MPa (114.8 GPa)
E22 = 11.7e3       # MPa
G12 = 9.66e3       # MPa
NU12 = 0.24
GC = 0.2774        # N/mm (表 2 的 Gc_II)
L = 0.01          # mm (长度尺度)
MIN_MESH_SIZE = 0.0015   # mm (最小网格尺寸, 用于有效 Gc 修正)
THETA_DEG = 30.0   # 纤维角
BETA = 300.0       # A1 惩罚参数
U_MAX = 0.02       # mm
N_STEPS = 40

# ========================== 网络与训练参数 ==========================
HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 200
INIT_M = 2.0
MAX_EPOCHS = 5000
TOL = 1e-6
PATIENCE = 10
WEIGHT_DECAY = 1e-7

MAX_STAG = 120
STAG_TOL = 5e-3
STAG_EPS = 1e-8    # 交错收敛判据中的防除零小量
STAG_CONV_STREAK = 3   # 交错迭代需连续满足收敛容限的次数

# 破坏停止判定: 反力降至最大反力的该比例以下时, 再加载若干步后停止
STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 0

# ========================== 边界条件选项 ==========================
# orig: 仅下端约束 u_x=0; tb: 上下两端约束 u_x=0; all: 四边约束 u_x=0
BC_MODES = ['orig', 'tb', 'all']
BC_LABEL = {'orig': 'u_x=0 at bottom edge only (original)',
            'tb': 'u_x=0 at top+bottom edges',
            'all': 'u_x=0 at all four edges'}
BC_TAG = {'orig': '', 'tb': '_tb', 'all': '_all'}


def stag_hparams(step, stag):
    """按 (加载步, 交错迭代) 返回 RPROP 的 (lr, min_step, max_step)。"""
    if step == 1 and stag == 1:
        return 1e-2, 1e-5, 0.1
    if (step==1 and stag <= 5) or stag<=1:
        return 1e-5, 1e-6, 0.05
    return 1e-6, 1e-7, 0.05


def theta_func(x, y):
    return np.full_like(np.asarray(x, dtype=np.float64), np.deg2rad(THETA_DEG))


def theta_torch(x, y):
    return torch.full_like(x, np.deg2rad(THETA_DEG), dtype=x.dtype)


def make_bc_functions(U_cur, bc='orig'):
    """硬约束 ansatz.

    u_y: 下端 y=-0.5 固定 0, 上端 y=0.5 施加 U_cur (三种边界条件相同).
    u_x: 按 bc 选择约束边:
        orig: 仅下端 (dist = y+0.5)
        tb  : 上下两端 (dist = (y+0.5)(0.5-y))
        all : 上下左右四边 (dist = (y+0.5)(0.5-y)(x+0.5)(0.5-x))
    """
    def dist_func(x):
        xx = x[:, 0]
        yy = x[:, 1]
        dist = torch.ones_like(x)
        dist[:, 1] = (yy + 0.5) * (0.5 - yy)
        if bc == 'orig':
            dist[:, 0] = yy + 0.5
        elif bc == 'tb':
            dist[:, 0] = (yy + 0.5) * (0.5 - yy)
        elif bc == 'all':
            dist[:, 0] = (yy + 0.5) * (0.5 - yy) * (xx + 0.5) * (0.5 - xx)
        else:
            raise ValueError(f'未知边界条件: {bc}')
        return dist

    def u_bc_func(x):
        u_bc = torch.zeros_like(x)
        u_bc[:, 1] = U_cur * (x[:, 1] + 0.5)
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
    """AT2 相场有限元求解器, 支持结构张量 A1/A2/A3 与可选有效 Gc 修正。"""
    def __init__(self, node_coords, tri_conn, Gc, l, beta, theta_func, tensor_type='A1',
                 lm=None, use_effective_gc=False):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.l = l
        self.beta = beta
        self.theta_func = theta_func
        self.tensor_type = tensor_type.upper()

        if use_effective_gc:
            # 有效临界能量释放率 (论文式 35)
            if lm is None:
                lm = self._min_edge_length()
            self.lm = lm
            f_beta = 1.0 - beta / 1000.0
            self.Gc = Gc * (1.0 - (lm / l) * f_beta)
            print(f'  有效 Gc: {Gc:.4f} -> {self.Gc:.4f} N/mm (lm={lm:.2e}, beta={beta:g})')
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
        ttype = self.tensor_type

        c = np.cos(theta_gauss)
        s = np.sin(theta_gauss)
        a = np.stack([c, s], axis=-1)
        dN = np.stack([self.dNdx, self.dNdy], axis=-1)
        dN_dot_a = np.einsum('eij,ej->ei', dN, a)
        grad = np.einsum('eik,ejk->eij', dN, dN)
        aniso = dN_dot_a[:, :, None] * dN_dot_a[:, None, :]

        if ttype in ('A1', 'A2'):
            K_aniso = Gc * l * self.area[:, None, None] * (grad + beta * aniso)
        elif ttype == 'A3':
            K_aniso = Gc * l * self.area[:, None, None] * (beta * grad + (1.0 - beta) * aniso)
        else:
            raise ValueError(f'未知结构张量类型: {ttype}')

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
    """上端 (y=0.5) 合力: 沿上边界积分 sigma_yy * dGamma, sigma = (1-d)^2 Q eps。"""
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
            if abs(node_coords[a, 1] - 0.5) < 1e-6 and abs(node_coords[b, 1] - 0.5) < 1e-6:
                force += sigma[e, 1] * abs(node_coords[a, 0] - node_coords[b, 0])
    return force


def truncate_csv(path, keep_steps_below):
    """只保留 step < keep_steps_below 的数据行 (保留表头); 文件不存在则跳过。"""
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


def run_case(args, device):
    base = os.path.dirname(os.path.abspath(__file__))
    angle = args.angle
    case_dir = os.path.join(base, f'SENT{angle}')
    bc_tag = BC_TAG.get(args.bc, '')
    tag = f'SENT{angle}_A1{bc_tag}'
    msh_file = os.path.join(case_dir, f'SENT{angle}.msh')
    ckpt_file = os.path.join(case_dir, f'ckpt_{tag}.pt')
    results_file = os.path.join(case_dir, f'results_{tag}.csv')
    force_file = os.path.join(case_dir, f'force_{tag}.csv')

    print(f'\n===== SENT{angle}: 纤维 {THETA_DEG}°, A1, beta={BETA} =====')
    print(f'边界条件 [{args.bc}]: {BC_LABEL.get(args.bc, "")}')

    # ---------------- 断点 / 回退重算 ----------------
    ck = None
    rollback = args.start_step is not None
    snap_file = None
    if not args.reset:
        if rollback:
            if args.start_step > 1:
                snap_file = os.path.join(base, f'ckpt_{tag}_s{args.start_step - 1:02d}.pt')
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
    material = FullEnergyMaterial(E11, E22, G12, NU12, theta_torch, is_plane_strain=False)
    pf_solver = AnisoPhaseFieldSolver(node_coords_np, tri_conn_np, GC, L, BETA,
                                      theta_func, tensor_type='A1',
                                      use_effective_gc=args.use_effective_gc,
                                      lm=args.lm if args.lm is not None else MIN_MESH_SIZE)
    theta_np = theta_func(fem_data['flat_gauss_pts'][:, 0].cpu().numpy(),
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
        if rollback and snap_file is None:
            force_max = 0.0
            stop_extra_left = None
            print(f'{tag}: warm-start 回退, 已重置破坏停止状态')
        print(f'从断点恢复: 上次已完成 step {ck["step"]}, 本次从 step {step_start} 继续')
    else:
        torch.manual_seed(12)
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

    # 结果文件与断点保持一致: 只保留 step < step_start 的行
    if ck is not None:
        truncate_csv(results_file, step_start)
        truncate_csv(force_file, step_start)

    if ck is None:
        with open(results_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'node', 'x', 'y', 'u_x', 'u_y', 'd'])
        with open(force_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'U', 'force', 'elastic_energy'])

    # ---------------- 加载步循环 ----------------
    for step in range(step_start, args.steps + 1):
        U_cur = U_MAX * step / N_STEPS
        print(f'\n========== 加载步 {step}/{N_STEPS}, U = {U_cur:.5f} ==========')
        dist_func, u_bc_func = make_bc_functions(U_cur, args.bc)
        loss_func = dsolve.DEMLoss(material, fem_data, dist_func, u_bc_func)
        d_prev_node = d_node.copy()
        with torch.no_grad():
            u_prev = net(node_coords, dist_func, u_bc_func).detach().cpu().numpy()

        n_stag = 1 if args.no_stag else MAX_STAG
        conv_streak = 0          # 连续满足收敛容限的次数
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
                H1, H2, H3, psi1.cpu().numpy(), psi2.cpu().numpy(), psi3.cpu().numpy())
            print(f'  历史场 H 最大: {H_total.max():.4e}')

            d_new = pf_solver.solve(H_total, theta_gauss=theta_np)
            d_new = np.maximum(d_new, d_prev_node)
            d_node = d_new
            d_gauss = torch.tensor(d_new[tri_conn_np].mean(axis=1),
                                   dtype=torch.float64, device=device)

            # 收敛判据: ||Δu||/||u|| + ||Δd||/||d|| < STAG_TOL
            # 需要连续 STAG_CONV_STREAK 次满足才判定该加载步收敛
            u_np = u_nodes.detach().cpu().numpy()
            du_ratio = np.linalg.norm(u_np - u_prev) / (np.linalg.norm(u_np) + STAG_EPS)
            dd_ratio = np.linalg.norm(d_new - d_prev_node) / (np.linalg.norm(d_new) + STAG_EPS)
            stag_metric = float(du_ratio + dd_ratio)
            max_dd = float(np.abs(d_new - d_prev_node).max())

            d_prev_node = d_new.copy()
            u_prev = u_np.copy()
            if stag_metric < STAG_TOL:
                conv_streak += 1
            else:
                conv_streak = 0
            print(f'  相场更新完成: ||Δu||/||u||={du_ratio:.3e}, ||Δd||/||d||={dd_ratio:.3e}, '
                  f'收敛指标={stag_metric:.3e} (max|Δd|={max_dd:.3e}), d最大={d_new.max():.3f}, '
                  f'连续满足 {conv_streak}/{STAG_CONV_STREAK}')
            if conv_streak >= STAG_CONV_STREAK:
                break

        with torch.no_grad():
            u_nodes = net(node_coords, dist_func, u_bc_func)
        force = compute_reaction_force(u_nodes, d_gauss, loss_func, material,
                                       node_coords_np, tri_conn_np)
        with torch.no_grad():
            energy = float(torch.exp(loss_func.compute_potential(net, d_gauss)))
        print(f'  上端反力 F = {force:.4e}, 弹性能 = {energy:.4e}')

        with open(results_file, 'a', newline='') as f:
            w = csv.writer(f)
            un = u_nodes.cpu().numpy()
            for i in range(Nn):
                w.writerow([step, i, node_coords_np[i, 0], node_coords_np[i, 1],
                            un[i, 0], un[i, 1], d_node[i]])
        with open(force_file, 'a', newline='') as f:
            csv.writer(f).writerow([step, U_cur, force, energy])
        force_rows.append([step, U_cur, force, energy])

        # 破坏判定: 反力降至最大反力 50% 以下时, 再加载 3 步后停止
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

        state = {
            'step': step,
            'net_state': net.state_dict(),
            'H1': H1, 'H2': H2, 'H3': H3,
            'd_node': d_node, 'd_gauss': d_gauss,
            'force_rows': force_rows,
            'force_max': force_max, 'stop_extra_left': stop_extra_left,
        }
        torch.save(state, ckpt_file)
        torch.save(state, os.path.join(case_dir, f'ckpt_{tag}_s{step:02d}.pt'))
        if stop_now:
            print(f'  破坏判定完成, 加载终止于 step {step}')
            print(f'{tag} 完成。')
            return
        print(f'  已保存断点 (step {step})')

    print(f'{tag} 完成。')


def main():
    parser = argparse.ArgumentParser(description='SENT30 复现 Yu 2024 图 7 (DEM + 相场 staggered)')
    parser.add_argument('--angle', type=int, choices=[30, 45, 60, 90], default=30,
                        help='纤维角度 (默认 30)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--steps', type=int, default=N_STEPS, help='最多运行的加载步数')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS, help='每次位移子问题最大 epoch 数')
    parser.add_argument('--no-stag', action='store_true', help='每步只做一次位移-相场交替')
    parser.add_argument('--reset', action='store_true', help='忽略已有断点重新开始')
    parser.add_argument('--start-step', type=int, default=None,
                        help='从第 N 个加载步开始重算 (回退重算; 优先使用 step N-1 的快照)')
    parser.add_argument('--use-effective-gc', action='store_true',
                        help='启用有效临界能量释放率修正 (论文式 35, 默认关闭)')
    parser.add_argument('--lm', type=float, default=None,
                        help='有效 Gc 修正中的最小单元尺寸 (默认 0.0015 mm)')
    parser.add_argument('--bc', type=str, choices=BC_MODES, default='orig',
                        help='边界条件: orig=仅下端约束u_x (原版), '
                             'tb=上下边约束u_x, all=四边约束u_x')
    args = parser.parse_args()

    if args.reset and args.start_step is not None:
        parser.error('--reset 与 --start-step 不能同时使用')
    if args.start_step is not None and not (1 <= args.start_step <= N_STEPS):
        parser.error(f'--start-step 应在 1..{N_STEPS} 之间')

    global THETA_DEG
    THETA_DEG = float(args.angle)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定了 --device cuda, 但 torch.cuda.is_available()=False, '
                           '请确认安装了 CUDA 版 PyTorch。')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'计算设备: {device}  (torch {torch.__version__}, '
          f'cuda available: {torch.cuda.is_available()})')
    if device.type == 'cpu':
        print('警告: 正在使用 CPU 训练。如需 GPU, 请安装 CUDA 版 PyTorch 或指定 --device cuda。')

    t_all = time.time()
    run_case(args, device)
    print(f'\n完成, 总耗时 {time.time()-t_all:.1f}s')


if __name__ == '__main__':
    main()
