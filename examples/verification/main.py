"""DEM-FEM numerical driver for the verification benchmark.

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
E11 = 171.0e3      # MPa
E22 = 9.08e3       # MPa
G12 = 5.29e3       # MPa
NU12 = 0.32
GC = 0.277         # N/mm (Gc22)
L = 0.01           # mm
U_MAX = 0.05       # mm
N_STEPS = 50
REFINE_FROM = 14      # 前 REFINE_FROM 个加载步保持原步长
REFINE_FACTOR = 10   # 第 REFINE_FROM -> REFINE_FROM+1 步之间的位移区间细化倍数

# 铺层 (左->右四个区域)
LAYUPS = {
    1: [0.0, -22.5, 0.0, 22.5],
    2: [0.0, 45.0, -45.0, 90.0],
}
REGION_BOUNDS = [-0.5, -0.25, 0.0, 0.25, 0.5]

BETA = {'A1': 300.0, 'A2': 300.0, 'A3': 0.2}
TENSORS = ['A1', 'A2', 'A3']


def u_at_step(step, refine_from=REFINE_FROM, factor=REFINE_FACTOR):
    """第 step 个加载步对应的加载位移 (mm)。

    前 refine_from 步保持原步长 (U_MAX/N_STEPS);
    第 refine_from 步 -> refine_from+1 步之间的位移区间细分为 factor 个子步;
    再之后的加载步恢复原步长。
    """
    base = U_MAX / N_STEPS
    if step <= refine_from:
        return base * step
    if step <= refine_from + factor:
        return base * (refine_from + (step - refine_from) / float(factor))
    return base * (step - factor + 1)


def stag_hparams(step, stag):
    """按 (加载步, 交错迭代) 返回 RPROP 的 (lr, min_step, max_step)。

    与其他算例 (SENT / 圆孔板) 的学习率变化方式一致:
    第 1 步第 1 次交错用大步长快速起步, 之后逐级降低;
    每一加载步内, 第 1 次交错用中等学习率, 后续交错用小学习率细调。
    """
    if step == 1 and stag == 1:
        return 1e-2, 1e-5, 0.1
    if (step == 1 and stag <= 5) or stag <= 1:
        return 1e-5, 1e-6, 0.05
    return 1e-6, 1e-8, 0.05



# ========================== 网络与训练参数 ==========================
HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 200
INIT_M = 2.0
MAX_EPOCHS = 5000
# 学习率不再全局固定: 按 (加载步, 交错迭代) 变化, 见 stag_hparams()
TOL = 1e-6
PATIENCE = 15
WEIGHT_DECAY = 1e-7

MAX_STAG = 30
STAG_TOL = 5e-3
STAG_EPS = 1e-6    # 交错收敛判据中的防除零小量 epsilon_0

# 破坏停止判定: 反力降至最大反力的该比例以下时, 再加载若干步后停止
STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 3


def theta_for_layup(layup_deg):
    """返回 theta(x, y) (弧度), 按 x 分四个区域, 逐区域常数。"""
    bounds = REGION_BOUNDS
    angles = [np.deg2rad(a) for a in layup_deg]

    def theta_func(x, y):
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x)
        for i in range(4):
            m = (x >= bounds[i] - 1e-12) & (x < bounds[i + 1] + 1e-12)
            out[m] = angles[i]
        return out

    def theta_torch(x, y):
        xn = x.detach().cpu().numpy()
        t = theta_func(xn, np.zeros_like(xn))
        return torch.tensor(t, dtype=x.dtype, device=x.device)

    return theta_func, theta_torch


def make_bc_functions(U_cur):
    """硬约束 ansatz: 下端 y=-0.5 完全固定, 上端 y=0.5 施加 u_y=U_cur (u_x 自由)。"""
    def dist_func(x):
        y = x[:, 1]
        dist = torch.ones_like(x)
        dist[:, 0] = y + 0.5
        dist[:, 1] = (y + 0.5) * (0.5 - y)
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
        # theta 每个加载步内是同一个张量对象, 全局刚度 Q 只需算一次
        if self._cache_theta is not theta:
            self._cache_theta = theta
            self._cache_Q = self.get_global_stiffness(theta)
        Q = self._cache_Q
        psi = 0.5 * torch.einsum('...i,...ij,...j->...', eps, Q, eps)
        g = (1.0 - d) ** 2
        return g * psi, torch.zeros_like(psi)


def local_driving_components(material, eps, theta):
    """论文式 (20)-(22): 在局部(纤维)坐标系下计算三个驱动能量分量。

    eps:   [N,3] 全局工程应变 (exx, eyy, gxy)
    theta: [N]   纤维角 (弧度)
    返回: psi_TEEDF, psi_TEEDM, psi_TSEDM
    """
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    c = torch.cos(theta)
    s = torch.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s

    # 全局工程应变 -> 局部正应变
    e11 = c2 * exx + cs * gxy + s2 * eyy
    e22 = s2 * exx - cs * gxy + c2 * eyy
    g12 = -2.0 * cs * exx + 2.0 * cs * eyy + (c2 - s2) * gxy

    e11p = torch.relu(e11)
    e22p = torch.relu(e22)

    # 局部应力 = Q_loc * eps_loc^+
    s11p = material.Q11 * e11p + material.Q12 * e22p
    s22p = material.Q12 * e11p + material.Q22 * e22p
    tau12 = material.Q66 * g12

    psi_teedf = 0.5 * e11p * s11p
    psi_teedm = 0.5 * e22p * s22p
    psi_tsedm = 0.5 * g12 * tau12
    return psi_teedf, psi_teedm, psi_tsedm


class AnisoPhaseFieldSolver:
    """AT2 相场有限元求解器, 支持结构张量 A1/A2/A3 与有效 Gc 修正。"""
    def __init__(self, node_coords, tri_conn, Gc, l, beta, theta_func, tensor_type='A1',
                 lm=None):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.l = l
        self.beta = beta
        self.theta_func = theta_func
        self.tensor_type = tensor_type.upper()

        # 有效临界能量释放率 (论文式 35)
        if lm is None:
            lm = self._min_edge_length()
        self.lm = lm
        f_beta = 1.0 - beta / 1000.0
        self.Gc = Gc * (1.0 - (lm / l) * f_beta)
        print(f'  有效 Gc: {Gc:.4f} -> {self.Gc:.4f} N/mm (lm={lm:.2e}, beta={beta:g})')

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
        H_gauss = np.asarray(H_gauss, dtype=np.float64)

        Ne = self.Ne
        Nn = self.Nn
        Gc = self.Gc
        l = self.l
        beta = self.beta
        ttype = self.tensor_type

        # ---------------- 向量化单元装配 ----------------
        c = np.cos(theta_gauss)
        s = np.sin(theta_gauss)
        a = np.stack([c, s], axis=-1)                        # [Ne, 2]
        dN = np.stack([self.dNdx, self.dNdy], axis=-1)       # [Ne, 3, 2]
        dN_dot_a = np.einsum('eij,ej->ei', dN, a)            # [Ne, 3]
        grad = np.einsum('eik,ejk->eij', dN, dN)             # [Ne, 3, 3]
        aniso = dN_dot_a[:, :, None] * dN_dot_a[:, None, :]  # [Ne, 3, 3]

        if ttype in ('A1', 'A2'):
            # A = I + beta*M  (2D 中 A2 = A1)
            K_aniso = Gc * l * self.area[:, None, None] * (grad + beta * aniso)
        elif ttype == 'A3':
            # A = M + beta*(I-M) -> beta*|grad d|^2 + (1-beta)*(grad d . a)^2
            K_aniso = Gc * l * self.area[:, None, None] * (beta * grad + (1.0 - beta) * aniso)
        else:
            raise ValueError(f'未知结构张量类型: {ttype}')

        Ni = np.full((Ne, 3), 1.0 / 3.0)
        M_e = (self.area[:, None, None]
               * Ni[:, :, None] * Ni[:, None, :]
               * (Gc / l + 2.0 * H_gauss)[:, None, None])
        K_e = K_aniso + M_e                                  # [Ne, 3, 3]
        f_e = 2.0 * H_gauss[:, None] * self.area[:, None] * Ni  # [Ne, 3]

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


def run_case(mesh_idx, tensor, args, device):
    """运行单个算例 (mesh_idx x tensor)。"""
    base = os.path.dirname(os.path.abspath(__file__))
    tag = f'm{mesh_idx}_{tensor}'
    msh_file = os.path.join(base, f'verification{mesh_idx}.msh')
    ckpt_file = os.path.join(base, f'ckpt_{tag}.pt')
    results_file = os.path.join(base, f'results_{tag}.csv')
    force_file = os.path.join(base, f'force_{tag}.csv')
    # 每个算例一个文件夹, 里面 fig/ 存放每个加载步的相场云图 (与有限元版一致)
    case_dir = os.path.join(base, f'dem_{tag}')
    os.makedirs(case_dir, exist_ok=True)

    layup = LAYUPS[mesh_idx]
    beta = BETA[tensor]
    print(f'\n===== 铺层 {layup} 度, 结构张量 {tensor}, beta={beta} =====')
    base_du = U_MAX / N_STEPS
    print(f'  加载方案: 前 {args.refine_from} 步步长 {base_du:.5f} mm; '
          f'第 {args.refine_from}→{args.refine_from + 1} 步区间细化 '
          f'{args.refine_factor} 倍 (步长 {base_du / args.refine_factor:.6f} mm); '
          f'之后恢复原步长 (断裂区细化为第 '
          f'{args.refine_from + 1}..{args.refine_from + args.refine_factor} 步)')

    # 已完成则跳过
    ck = None
    rollback = args.start_step is not None
    snap_file = None
    if not args.reset:
        if rollback:
            # 回退重算: 优先使用 step N-1 的快照, 否则用最新断点 warm start
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
                truncate_csv(results_file, args.start_step)
                truncate_csv(force_file, args.start_step)
            else:
                rollback = False   # --start-step 1 等价于从头开始
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
    theta_func, theta_torch = theta_for_layup(layup)
    material = FullEnergyMaterial(E11, E22, G12, NU12, theta_torch, is_plane_strain=False)
    pf_solver = AnisoPhaseFieldSolver(node_coords_np, tri_conn_np, GC, L, beta,
                                      theta_func, tensor_type=tensor, lm=args.lm)
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
            # warm-start 回退: 断点里的反力/停止状态超前, 重置破坏判定
            force_max = 0.0
            stop_extra_left = None
            print(f'{tag}: warm-start 回退, 已重置破坏停止状态')
        print(f'从断点恢复: 继续加载步 {step_start}')
    else:
        torch.manual_seed(0)
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

    # 若没有有效断点, 结果文件一律重写表头 (避免残留旧数据)
    if ck is None:
        with open(results_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'node', 'x', 'y', 'u_x', 'u_y', 'd'])
        with open(force_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'U', 'force', 'elastic_energy'])

    # ---------------- 加载步循环 ----------------
    for step in range(step_start, args.steps + 1):
        U_cur = u_at_step(step, args.refine_from, args.refine_factor)
        print(f'\n========== 加载步 {step}/{args.steps}, U = {U_cur:.5f} ==========')
        dist_func, u_bc_func = make_bc_functions(U_cur)
        loss_func = dsolve.DEMLoss(material, fem_data, dist_func, u_bc_func)
        d_prev_node = d_node.copy()
        # u^{(0)}: 本加载步开始时的热启动位移场 (第一个交错迭代的相对变化参考)
        with torch.no_grad():
            u_prev = net(node_coords, dist_func, u_bc_func).detach().cpu().numpy()

        n_stag = 1 if args.no_stag else MAX_STAG
        for stag in range(1, n_stag + 1):
            t0 = time.time()
            lr, min_step, max_step = stag_hparams(step, stag)
            net, _ = dsolve.train_displacement_subproblem(
                net, loss_func, d_gauss,
                max_epochs=args.epochs, lr=lr, min_step=min_step, max_step=max_step,
                tol=TOL, patience=PATIENCE, weight_decay=WEIGHT_DECAY, verbose=False)
            print(f'  [stag {stag}/{n_stag}] 位移训练完成 '
                  f'(lr={lr:.0e}, min={min_step:.0e}, max={max_step:.0e}), '
                  f'{time.time()-t0:.1f}s')

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
        print(f'  上端反力 F = {force:.4e}, 弹性能 = {energy:.4e}')

        # 每个加载步都保存节点位移与相场, 供任意步云图使用
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
        # 每步快照: 支持 --start-step 回退到任意步
        torch.save(state, os.path.join(base, f'ckpt_{tag}_s{step:02d}.pt'))
        if stop_now:
            print(f'  破坏判定完成, 加载终止于 step {step}')
            print(f'{tag} 完成。')
            return
        print(f'  已保存断点 (step {step})')

    print(f'{tag} 完成。')


def main():
    parser = argparse.ArgumentParser(description='复现 Yu 2024 图 5 (DEM + 相场 staggered)')
    parser.add_argument('--mesh', type=int, choices=[1, 2], default=None,
                        help='铺层 1 或 2; 缺省运行全部')
    parser.add_argument('--tensor', type=str, choices=TENSORS, default=None,
                        help='结构张量 A1/A2/A3; 缺省运行全部')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='计算设备: auto(默认)/cuda/cpu')
    parser.add_argument('--steps', type=int, default=N_STEPS, help='最多运行的加载步数')
    parser.add_argument('--refine-from', type=int, default=REFINE_FROM,
                        help=f'从第 N 步之后开始细化加载区间 (默认 {REFINE_FROM})')
    parser.add_argument('--refine-factor', type=int, default=REFINE_FACTOR,
                        help=f'细化倍数 (默认 {REFINE_FACTOR})')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS, help='每次位移子问题最大 epoch 数')
    parser.add_argument('--no-stag', action='store_true', help='每个加载步只做一次位移-相场交替')
    parser.add_argument('--reset', action='store_true', help='忽略已有断点重新开始')
    parser.add_argument('--start-step', type=int, default=None,
                        help='从第 N 个加载步开始重算 (回退重算; 优先使用 step N-1 的快照)')
    parser.add_argument('--lm', type=float, default=0.003,
                        help='有效 Gc 修正中的最小单元尺寸 (默认 0.003 mm, 对应布种尺寸)')
    args = parser.parse_args()
    if args.reset and args.start_step is not None:
        parser.error('--reset 与 --start-step 不能同时使用')
    max_step = N_STEPS + max(0, args.refine_factor - 1)
    if args.start_step is not None and not (1 <= args.start_step <= max_step):
        parser.error(f'--start-step 应在 1..{max_step} 之间')

    # 设备选择
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            '指定了 --device cuda, 但 torch.cuda.is_available()=False。'
            '请确认已安装 CUDA 版 PyTorch (例如: pip install torch --index-url '
            'https://download.pytorch.org/whl/cu121)。')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'计算设备: {device}  (torch {torch.__version__}, '
          f'cuda available: {torch.cuda.is_available()})')
    if device.type == 'cpu':
        print('警告: 正在使用 CPU 训练。如需 GPU, 请安装 CUDA 版 PyTorch '
              '或显式指定 --device cuda。')

    # 配置组合: 缺省 2 铺层 x 3 张量 = 6 个算例
    meshes = [args.mesh] if args.mesh is not None else [1, 2]
    tensors = [args.tensor] if args.tensor is not None else TENSORS
    configs = [(m, t) for m in meshes for t in tensors]
    print(f'待运行算例 ({len(configs)} 个): {[f"m{m}_{t}" for m, t in configs]}')

    t_all = time.time()
    for mesh_idx, tensor in configs:
        t0 = time.time()
        run_case(mesh_idx, tensor, args, device)
        print(f'--- {f"m{mesh_idx}_{tensor}"} 耗时 {time.time()-t0:.1f}s ---')
    print(f'\n全部算例完成, 总耗时 {time.time()-t_all:.1f}s')


if __name__ == '__main__':
    main()
