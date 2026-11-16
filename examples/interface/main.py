# -*- coding: utf-8 -*-
"""DEM-FEM numerical driver for the interface benchmark.

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
# 层内 IM7/8552 (表 5), MPa
E11 = 171.42e3
E22 = 9.08e3
G12 = 5.29e3
NU12 = 0.32
# 界面 (各向同性), MPa
E_IF = 9.08e3
NU_IF = 0.32
G_IF = E_IF / (2.0 * (1.0 + NU_IF))
# Gc (N/mm)
GC_INTRA = 0.277          # 层内 = Gc22
GC_IF = {'weak': 0.0277, 'strong1': 0.277, 'strong2': 2.77}
# 相场
L = 0.0065                # mm
BETA_INTRA = 300.0        # A1 惩罚参数
BETA_IF = 0.0             # 界面各向同性
# 加载
U_MAX = 0.075             # mm
N_STEPS = 50

# ========================== 网格 ==========================
# 用户上传的 inp 已含几何开缝 (楔形窄缝), 直接转换得到 interface_new.msh
MSH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'interface_new.msh')

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
STAG_EPS = 1e-8    # 交错收敛判据中的防除零小量

CASES = ['weak', 'strong1', 'strong2']


def stag_hparams(step, stag):
    """按 (加载步, 交错迭代) 返回 RPROP 的 (lr, min_step, max_step)."""
    if step == 1 and stag == 1:
        return 1e-2, 1e-6, 0.1
    if (step == 1 and stag <= 2) or stag ==1:
        return 1e-4, 1e-7, 0.05
    return 1e-5, 1e-8, 0.05


def make_bc_functions(U_cur):
    """硬约束 ansatz: 下边 y=-0.5 固定 u_y=0 (u_x 自由); 上边 y=0.5 施加 u_y=U."""
    def dist_func(x):
        y = x[:, 1]
        dist = torch.ones_like(x)
        dist[:, 0] = 1.0                     # u_x 自由
        dist[:, 1] = (y + 0.5) * (0.5 - y)   # 上/下边 u_y=0
        return dist

    def u_bc_func(x):
        u_bc = torch.zeros_like(x)
        u_bc[:, 1] = U_cur * (x[:, 1] + 0.5)
        return u_bc

    return dist_func, u_bc_func


# ========================== 材料数据 (按区域) ==========================
def build_material_data(node_coords, tri_conn):
    """按单元形心划分区域, 返回每单元 Q 矩阵 / Gc / beta.
    region: 0 = 层内 (|x|>0.01), 1 = 界面 (|x|<=0.01).
    theta 全为 0 (层内 theta=0, 界面各向同性), 故局部系=全局系.
    """
    p0 = node_coords[tri_conn[:, 0]]
    p1 = node_coords[tri_conn[:, 1]]
    p2 = node_coords[tri_conn[:, 2]]
    centroid = (p0 + p1 + p2) / 3.0
    region = np.where(np.abs(centroid[:, 0]) <= 0.01, 1, 0)

    # 层内 Q (平面应力)
    nu21 = NU12 * E22 / E11
    den = 1.0 - NU12 * nu21
    q11, q22, q12, q66 = E11 / den, E22 / den, NU12 * E22 / den, G12
    # 界面 Q (各向同性, 平面应力)
    den_if = 1.0 - NU_IF ** 2
    q11i = q22i = E_IF / den_if
    q12i = NU_IF * E_IF / den_if
    q66i = G_IF

    Q = np.zeros((tri_conn.shape[0], 3, 3))
    m = region == 0
    i = region == 1
    Q[m] = [[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, q66]]
    Q[i] = [[q11i, q12i, 0.0], [q12i, q22i, 0.0], [0.0, 0.0, q66i]]

    Gc = np.where(region == 0, GC_INTRA, GC_IF['weak'])  # 默认 weak, 外部按工况覆盖
    beta = np.where(region == 0, BETA_INTRA, BETA_IF)
    return dict(Q=Q, Gc=Gc, beta=beta, region=region, centroid=centroid)


def local_driving_components(eps, Q_elems):
    """论文式(20)-(22), theta=0 (局部系=全局系).
    对界面元素, Q 为各向同性, 公式自动退化为各向同性拉伸分解.
    """
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    e11p = torch.relu(exx)
    e22p = torch.relu(eyy)
    s11p = Q_elems[:, 0, 0] * e11p + Q_elems[:, 0, 1] * e22p
    s22p = Q_elems[:, 1, 0] * e11p + Q_elems[:, 1, 1] * e22p
    tau12 = Q_elems[:, 2, 2] * gxy
    psi_teedf = 0.5 * e11p * s11p
    psi_teedm = 0.5 * e22p * s22p
    psi_tsedm = 0.5 * gxy * tau12
    return psi_teedf, psi_teedm, psi_tsedm


# ========================== 相场求解器 (Gc/beta 随空间变化) ==========================
class InterfacePhaseFieldSolver:
    """AT2 相场 FEM, 支持每单元 Gc 与 beta (界面区 beta=0 即各向同性)."""
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
        self.centroid = np.stack([(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0], axis=1)

    def solve(self, H_gauss):
        Nn, Ne = self.Nn, self.Ne
        Gc = self.Gc
        l = self.l
        beta = self.beta
        # theta=0 -> a=(1,0), dN_dot_a = dNdx
        dN = np.stack([self.dNdx, self.dNdy], axis=-1)
        a = np.array([1.0, 0.0])
        dN_dot_a = self.dNdx
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
        vals = K_e.reshape(-1)
        rhs = np.bincount(conn.ravel(), weights=f_e.ravel(), minlength=Nn)

        from scipy.sparse import coo_matrix
        K = coo_matrix((vals, (rows, cols)), shape=(Nn, Nn)).tocsr()
        from scipy.sparse.linalg import spsolve, cg
        try:
            d = spsolve(K, rhs)
        except Exception:
            d, _ = cg(K, rhs, tol=1e-8, maxiter=5000)
        return np.clip(d, 0.0, 1.0)


# ========================== DEM 能量损失 (每单元材料) ==========================
class InterfaceDEMLoss:
    def __init__(self, Q_elems, fem_data, dist_func, u_bc_func):
        self.node_coords = fem_data['node_coords']
        self.flat_conn = fem_data['flat_conn']
        self.flat_gauss_pts = fem_data['flat_gauss_pts']
        self.flat_gauss_w = fem_data['flat_gauss_w']
        self.flat_dN_dx = fem_data['flat_dN_dx']
        self.flat_dN_dy = fem_data['flat_dN_dy']
        self.dist_func = dist_func
        self.u_bc_func = u_bc_func
        self.Q = torch.tensor(Q_elems, dtype=torch.float64, device=fem_data['node_coords'].device)
        self.node_coords_t = self.node_coords

    def compute_strain(self, u_nodes):
        u_e = u_nodes[self.flat_conn]
        ux, uy = u_e[:, :, 0], u_e[:, :, 1]
        exx = torch.sum(self.flat_dN_dx * ux, dim=1)
        eyy = torch.sum(self.flat_dN_dy * uy, dim=1)
        gxy = (torch.sum(self.flat_dN_dy * ux, dim=1) +
               torch.sum(self.flat_dN_dx * uy, dim=1))
        return torch.stack([exx, eyy, gxy], dim=-1)

    def compute_potential(self, net, d_gauss):
        u_nodes = net(self.node_coords_t, self.dist_func, self.u_bc_func)
        eps = self.compute_strain(u_nodes)
        psi = 0.5 * torch.einsum('ei,eij,ej->e', eps, self.Q, eps)
        energy = torch.sum(self.flat_gauss_w * ((1.0 - d_gauss) ** 2) * psi)
        return torch.log(energy + 1e-12)


# ========================== 反力 ==========================
def compute_reaction_force(u_nodes, d_gauss, Q_elems, node_coords, tri_conn):
    """上边 (y=0.5) 反力: 沿上边界积分 sigma_yy, sigma=(1-d)^2 Q eps."""
    with torch.no_grad():
        u_e = u_nodes[tri_conn]
        ux, uy = u_e[:, :, 0], u_e[:, :, 1]
        p0 = node_coords[tri_conn[:, 0]]
        p1 = node_coords[tri_conn[:, 1]]
        p2 = node_coords[tri_conn[:, 2]]
        x0, y0 = p0[:, 0], p0[:, 1]
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        denom = torch.where(torch.abs(area2) < 1e-12,
                            torch.full_like(area2, 1e-12), area2)
        dNdx = torch.stack([(y1 - y2) / denom, (y2 - y0) / denom, (y0 - y1) / denom], dim=1)
        dNdy = torch.stack([(x2 - x1) / denom, (x0 - x2) / denom, (x1 - x0) / denom], dim=1)
        exx = torch.sum(dNdx * ux, dim=1)
        eyy = torch.sum(dNdy * uy, dim=1)
        gxy = (torch.sum(dNdy * ux, dim=1) + torch.sum(dNdx * uy, dim=1))
        eps = torch.stack([exx, eyy, gxy], dim=-1)
        Q = torch.tensor(Q_elems, dtype=torch.float64, device=u_nodes.device)
        sigma = torch.einsum('eij,ej->ei', Q, eps)
        g = ((1.0 - d_gauss) ** 2)
        sigma *= g[:, None]
    force = 0.0
    node_coords_np = node_coords.cpu().numpy() if torch.is_tensor(node_coords) else node_coords
    tri_conn_np = tri_conn.cpu().numpy() if torch.is_tensor(tri_conn) else tri_conn
    sigma_np = sigma.cpu().numpy()
    for e in range(tri_conn_np.shape[0]):
        n0, n1, n2 = tri_conn_np[e]
        for a, b in ((n0, n1), (n1, n2), (n2, n0)):
            if (abs(node_coords_np[a, 1] - 0.5) < 1e-6 and
                    abs(node_coords_np[b, 1] - 0.5) < 1e-6):
                force += sigma_np[e, 1] * abs(node_coords_np[a, 0] - node_coords_np[b, 0])
    return force


# ========================== 绘图: 每步相场云图 ==========================


# ========================== 主流程 ==========================
def run_case(case, args, device):
    base = os.path.dirname(os.path.abspath(__file__))
    case_dir = os.path.join(base, case)
    os.makedirs(case_dir, exist_ok=True)
    tag = f'interface_{case}'
    msh_file = MSH_FILE
    force_file = os.path.join(case_dir, f'force_{case}.csv')
    results_file = os.path.join(case_dir, f'results_{case}.csv')
    ckpt_file = os.path.join(case_dir, f'ckpt_{tag}.pt')

    print(f'\n===== {case}: 界面 Gc={GC_IF[case]} N/mm =====')

    # ---------------- 网格 ----------------
    node_coords_np, tri_conn_np = dsolve.read_gmsh_mesh(msh_file)
    node_coords = torch.tensor(node_coords_np, dtype=torch.float64, device=device)
    tri_conn = torch.tensor(tri_conn_np, dtype=torch.long, device=device)
    fem_data = dsolve.compute_triangle_fem_data(node_coords, tri_conn)
    Nn, Ne = node_coords_np.shape[0], tri_conn_np.shape[0]
    print(f'网格: {Nn} 节点, {Ne} 单元')

    # ---------------- 材料数据 ----------------
    mdata = build_material_data(node_coords_np, tri_conn_np)
    mdata['Gc'] = np.where(mdata['region'] == 0, GC_INTRA, GC_IF[case])
    reg_if = mdata['region'] == 1
    print(f'单元: 层内 {Ne - reg_if.sum()}, 界面 {reg_if.sum()}')
    print(f'界面区 Gc = {GC_IF[case]} N/mm, beta=0; 层内 Gc=0.277, beta=300')

    pf_solver = InterfacePhaseFieldSolver(node_coords_np, tri_conn_np,
                                          mdata['Gc'], L, mdata['beta'])

    # ---------------- 位移网络 ----------------
    # ---------------- 断点 / 恢复 ----------------
    ck = None
    if os.path.exists(ckpt_file) and not args.reset:
        ck = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        if ck['step'] >= args.steps:
            print(f'{tag}: 已完成 {ck["step"]} 步, 跳过')
            return ck.get('force_rows', [])

    if ck is not None:
        H1, H2, H3 = ck['H1'], ck['H2'], ck['H3']
        d_node = ck['d_node']
        d_gauss = ck['d_gauss'].to(device)
        net = dsolve.DisplacementNet(hidden_layers=HIDDEN_LAYERS,
                                     hidden_neurons=HIDDEN_NEURONS,
                                     init_m=INIT_M, output_scale=U_MAX).double()
        net.load_state_dict(ck['net_state'])
        net = net.to(device)
        force_rows = ck['force_rows']
        force_max = ck.get('force_max', 0.0)
        stop_extra_left = ck.get('stop_extra_left', None)
        step_start = ck['step'] + 1
        print(f'{tag}: 从断点恢复, 上次完成 step {ck["step"]}, 从 {step_start} 继续')
    else:
        torch.manual_seed(12)
        net = dsolve.DisplacementNet(hidden_layers=HIDDEN_LAYERS,
                                     hidden_neurons=HIDDEN_NEURONS,
                                     init_m=INIT_M, output_scale=U_MAX).double()
        net = net.to(device)
        d_gauss = torch.zeros(Ne, dtype=torch.float64, device=device)
        d_node = np.zeros(Nn)
        H1 = np.zeros(Ne)
        H2 = np.zeros(Ne)
        H3 = np.zeros(Ne)
        force_max = 0.0
        stop_extra_left = None
        step_start = 1
        force_rows = []
        with open(force_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'U', 'force'])
        with open(results_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'node', 'x', 'y', 'u_x', 'u_y', 'd'])

    # 恢复时截断结果文件 (只保留 step < step_start)
    if ck is not None:
        truncate_csv(force_file, step_start)
        truncate_csv(results_file, step_start)

    for step in range(step_start, args.steps + 1):
        U_cur = U_MAX * step / N_STEPS
        print(f'\n========== {case} 加载步 {step}/{N_STEPS}, U = {U_cur:.5f} ==========')
        dist_func, u_bc_func = make_bc_functions(U_cur)
        loss_func = InterfaceDEMLoss(mdata['Q'], fem_data, dist_func, u_bc_func)

        # 收敛判据 (与 SENT/circlehole 一致): ||Δu||/||u|| + ||Δd||/||d|| < STAG_TOL
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
            print(f'  [stag {stag}] 位移训练完成 ({time.time()-t0:.1f}s)')

            with torch.no_grad():
                u_nodes = net(node_coords, dist_func, u_bc_func)
                eps = loss_func.compute_strain(u_nodes)
                psi1, psi2, psi3 = local_driving_components(eps, loss_func.Q)
            H1, H2, H3, H_total = dsolve.update_history(
                H1, H2, H3, psi1.cpu().numpy(), psi2.cpu().numpy(), psi3.cpu().numpy())
            print(f'  历史场 H max: {H_total.max():.4e}')

            d_new = pf_solver.solve(H_total)
            d_new = np.maximum(d_new, d_prev_node)   # 相场不可逆 (单调不减)
            d_node = d_new
            d_gauss_new = d_new[tri_conn_np].mean(axis=1)
            d_gauss = torch.tensor(d_gauss_new, dtype=torch.float64, device=device)

            u_np = u_nodes.detach().cpu().numpy()
            du_ratio = np.linalg.norm(u_np - u_prev) / (np.linalg.norm(u_np) + STAG_EPS)
            dd_ratio = np.linalg.norm(d_new - d_prev_node) / (np.linalg.norm(d_new) + STAG_EPS)
            stag_metric = float(du_ratio + dd_ratio)
            max_dd = float(np.abs(d_new - d_prev_node).max())

            d_prev_node = d_new.copy()
            u_prev = u_np.copy()
            print(f'  相场更新: ||Δu||/||u||={du_ratio:.3e}, ||Δd||/||d||={dd_ratio:.3e}, '
                  f'收敛指标={stag_metric:.3e} (max|Δd|={max_dd:.3e}), d最大={d_new.max():.3f}')
            if stag_metric < STAG_TOL:
                print(f'  交错收敛于 stag {stag}')
                break

        # ---------------- 反力与输出 ----------------
        with torch.no_grad():
            u_nodes = net(node_coords, dist_func, u_bc_func)
            force = compute_reaction_force(u_nodes, d_gauss, mdata['Q'],
                                           node_coords, tri_conn)
        force_rows.append([step, U_cur, force])
        with open(force_file, 'a', newline='') as f:
            csv.writer(f).writerow([step, U_cur, f'{force:.8f}'])
        force_max = max(force_max, force)
        print(f'  反力 F = {force:.6f} N (max so far {force_max:.6f})')

        # results CSV (节点 u, d)
        with torch.no_grad():
            u_np = u_nodes.cpu().numpy()
        with open(results_file, 'a', newline='') as f:
            w = csv.writer(f)
            for nid in range(Nn):
                w.writerow([step, nid, node_coords_np[nid, 0], node_coords_np[nid, 1],
                            f'{u_np[nid, 0]:.8f}', f'{u_np[nid, 1]:.8f}',
                            f'{d_node[nid]:.8f}'])

        # 破坏停止判定: 反力降至峰值的 50% 后再算 2 步
        if force_max > 0 and force < STOP_FRACTION * force_max:
            if stop_extra_left is None:
                stop_extra_left = STOP_EXTRA_STEPS
            else:
                stop_extra_left -= 1
            if stop_extra_left < 0:
                print(f'{case}: 反力降至峰值 50% 以下, 停止 (step {step})')
                break
        else:
            stop_extra_left = None

        # 断点
        torch.save({
            'step': step, 'net_state': net.state_dict(),
            'H1': H1, 'H2': H2, 'H3': H3,
            'd_node': d_node, 'd_gauss': d_gauss.cpu(),
            'force_rows': force_rows, 'force_max': force_max,
            'stop_extra_left': stop_extra_left,
        }, ckpt_file)

    print(f'{case} 完成, 共 {len(force_rows)} 步')
    return force_rows


STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 2


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


def main():
    parser = argparse.ArgumentParser(description='interface 算例 (图 12/13)')
    parser.add_argument('--case', choices=CASES, default=None)
    parser.add_argument('--steps', type=int, default=N_STEPS)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-stag', action='store_true')
    parser.add_argument('--reset', action='store_true', help='忽略已有断点, 从头重算')
    args = parser.parse_args()

    device = torch.device(args.device)
    cases = [args.case] if args.case else CASES

    base = os.path.dirname(os.path.abspath(__file__))
    for case in cases:
        run_case(case, args, device)

    # 三合一载荷-位移曲线 (图 13 样式), 从已有 force CSV 生成


if __name__ == '__main__':
    main()
