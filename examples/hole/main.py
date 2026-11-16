# -*- coding: utf-8 -*-
"""DEM-FEM numerical driver for the hole benchmark.

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

# ========================== 材料与加载参数 (论文表 6) ==========================
E11 = 47.4e3       # MPa
E22 = 16.2e3
G12 = 7.0e3
NU12 = 0.26
GC = 0.622         # N/mm (单一 Gc, 不用有效 Gc 修正)
L = 0.4            # mm
BETA = 300.0
U_MAX = 1.0        # mm
N_STEPS = 100

# 几何与边界
XMIN, XMAX = -40.0, 40.0
YMIN, YMAX = -9.0, 9.0
LX = XMAX - XMIN
LY = YMAX - YMIN
FIX_X, FIX_Y = XMIN, YMIN     # 刚体位移固定点 (左下角)

CASES = ['C', 'S', 'T']
HOLE_NAME = {'C': 'Circle R=2.5', 'S': 'Square a=4.432', 'T': 'Triangle'}

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
STAG_EPS = 1e-8

STOP_FRACTION = 0.5
STOP_EXTRA_STEPS = 1


def stag_hparams(step, stag):
    """按 (加载步, 交错迭代) 返回 RPROP 的 (lr, min_step, max_step)。"""
    if step == 1 and stag ==1:
        return 5e-2, 5e-6, 0.2
    if (step == 1 and stag <= 5) or stag <=2:
        return 1e-4, 1e-6, 0.1
    return 1e-5, 1e-7, 0.05


THETA = np.deg2rad(90.0)   # 纤维方向与 x 轴夹角 = 90° (竖直)


def theta_np_func(x, y):
    return np.full_like(np.asarray(x, dtype=np.float64), THETA)


def theta_torch_func(x, y):
    return torch.full_like(x, THETA, dtype=x.dtype)


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
    """论文 2.2 节假设 psi^- = 0: 位移子问题使用完整弹性能 (1-d)^2 * 1/2 eps^T Q eps."""
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
    """论文式 (20)-(22): 在局部(纤维)坐标系下计算三个驱动能量分量."""
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
    """AT2 相场有限元求解器 (A1), 固定 theta=90°, 不用有效 Gc 修正."""
    def __init__(self, node_coords, tri_conn, Gc, l, beta):
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.l = l
        self.beta = beta
        self.Gc = Gc
        print(f'  相场: Gc={Gc:.4f} N/mm (原始值, 未做有效 Gc 修正), '
              f'l={l:.3f}, beta={beta:g}')
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
        self.area = np.abs(area2) / 2.0
        self.area = np.maximum(self.area, 1e-12)
        denom = np.where(np.abs(area2) < 1e-12, 1e-12, area2)
        self.dNdx = np.stack([(y1 - y2) / denom, (y2 - y0) / denom, (y0 - y1) / denom], axis=1)
        self.dNdy = np.stack([(x2 - x1) / denom, (x0 - x2) / denom, (x1 - x0) / denom], axis=1)
        self.centroid = np.stack([(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0], axis=1)

    def solve(self, H_gauss):
        Nn, Ne = self.Nn, self.Ne
        Gc, l, beta = self.Gc, self.l, self.beta
        dN = np.stack([self.dNdx, self.dNdy], axis=-1)
        dN_dot_a = self.dNdy      # theta=90°, a=(0,1)
        grad = np.einsum('eik,ejk->eij', dN, dN)
        aniso = dN_dot_a[:, :, None] * dN_dot_a[:, None, :]
        K_aniso = Gc * l * self.area[:, None, None] * (grad + beta * aniso)
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


def compute_reaction_force(u_nodes, d_gauss, loss_func, material, node_coords, tri_conn):
    """右端 (x=XMAX) 合力: 沿右边界积分 sigma_xx * dGamma."""
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
    rows = []
    for ln in lines[1:]:
        try:
            if int(ln.split(',')[0].strip()) < keep_steps_below:
                rows.append(ln)
        except ValueError:
            pass
    with open(path, 'w', newline='') as f:
        f.write(header)
        f.writelines(rows)
    print(f'  已截断 {os.path.basename(path)}: 保留 step < {keep_steps_below} 的 {len(rows)} 行')


def csv_is_valid(path, expected_header, n_cols):
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r', newline='') as f:
            first = f.readline().strip()
            if first != expected_header:
                return False
            for line in f:
                line = line.strip()
                if line:
                    return len(line.split(',')) == n_cols
    except OSError:
        return False
    return True


def repair_csv(path, expected_header, n_cols, tag):
    if not csv_is_valid(path, expected_header, n_cols):
        backup = path + '.bak'
        try:
            os.replace(path, backup)
        except OSError:
            return
        print(f'警告: {os.path.basename(path)} 格式与当前版本不一致, '
              f'已备份为 {os.path.basename(backup)} 并重建')


def read_step_times(path):
    times = {}
    if not os.path.exists(path):
        return times
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    times[int(row[0])] = float(row[1])
    except (ValueError, OSError, csv.Error):
        pass
    return times


def write_step_times(path, times):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'seconds'])
        for step in sorted(times):
            w.writerow([step, f'{times[step]:.2f}'])




def run_case(args, device):
    base = os.path.dirname(os.path.abspath(__file__))
    case = args.case
    case_dir = os.path.join(base, case)
    tag = f'hole_{case}'
    msh_file = os.path.join(base, f'{case}.msh')
    ckpt_file = os.path.join(case_dir, f'ckpt_{tag}.pt')
    results_file = os.path.join(case_dir, f'results_{tag}.csv')
    force_file = os.path.join(case_dir, f'force_{tag}.csv')
    time_file = os.path.join(case_dir, f'time_{tag}.csv')
    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(case_dir, exist_ok=True)

    repair_csv(results_file, 'step,node,x,y,u_x,u_y,d', 7, tag)
    repair_csv(force_file, 'step,U,force,elastic_energy', 4, tag)

    print(f'\n===== {HOLE_NAME[case]} 孔, Gc={GC} N/mm, l={L}, beta={BETA}, theta=90° =====')

    ck = None
    if not args.reset and os.path.exists(ckpt_file):
        ck = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        if ck['step'] >= args.steps or (ck.get('stop_extra_left', None) is not None
                                        and ck['stop_extra_left'] < 0):
            print(f'{tag}: 已完成 {ck["step"]} 步, 跳过')
            return

    node_coords_np, tri_conn_np = dsolve.read_gmsh_mesh(msh_file)
    node_coords = torch.tensor(node_coords_np, dtype=torch.float64, device=device)
    tri_conn = torch.tensor(tri_conn_np, dtype=torch.long, device=device)
    fem_data = dsolve.compute_triangle_fem_data(node_coords, tri_conn)
    Nn, Ne = node_coords_np.shape[0], tri_conn_np.shape[0]
    print(f'网格: {Nn} 节点, {Ne} 单元')

    material = FullEnergyMaterial(E11, E22, G12, NU12, theta_torch_func,
                                  is_plane_strain=False)
    pf_solver = AnisoPhaseFieldSolver(node_coords_np, tri_conn_np, GC, L, BETA)
    theta_np = theta_np_func(fem_data['flat_gauss_pts'][:, 0].cpu().numpy(),
                             fem_data['flat_gauss_pts'][:, 1].cpu().numpy())

    step_start = 1
    force_rows = []
    if ck is not None:
        step_start = ck['step'] + 1
        H1, H2, H3 = ck['H1'], ck['H2'], ck['H3']
        d_node = ck['d_node']
        d_gauss = ck['d_gauss'].to(device)
        net = dsolve.DisplacementNet(hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS,
                                     init_m=INIT_M, output_scale=U_MAX).double()
        net.load_state_dict(ck['net_state'])
        net = net.to(device)
        force_rows = ck['force_rows']
        force_max = ck.get('force_max', 0.0)
        stop_extra_left = ck.get('stop_extra_left', None)
        step_times = {s: t for s, t in ck.get('step_times', {}).items() if s < step_start}
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
    if not csv_is_valid(results_file, 'step,node,x,y,u_x,u_y,d', 7):
        with open(results_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'node', 'x', 'y', 'u_x', 'u_y', 'd'])
    if not csv_is_valid(force_file, 'step,U,force,elastic_energy', 4):
        with open(force_file, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'U', 'force', 'elastic_energy'])

    for step in range(step_start, args.steps + 1):
        step_t0 = time.time()
        U_cur = U_MAX * step / N_STEPS
        print(f'\n========== {case} 加载步 {step}/{N_STEPS}, U = {U_cur:.5f} ==========')
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
            print(f'  [stag {stag}/{n_stag}] 位移训练完成 (lr={lr:.0e}), {time.time()-t0:.1f}s')

            with torch.no_grad():
                u_nodes = net(node_coords, dist_func, u_bc_func)
                eps = loss_func.compute_strain(u_nodes)
                psi1, psi2, psi3 = local_driving_components(material, eps, loss_func.flat_theta)
            H1, H2, H3, H_total = dsolve.update_history(
                H1, H2, H3, psi1.cpu().numpy(), psi2.cpu().numpy(),
                psi3.cpu().numpy())
            print(f'  历史场 H 最大: {H_total.max():.4e}')

            d_new = pf_solver.solve(H_total)
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
            print(f'  相场更新: ||Δu||/||u||={du_ratio:.3e}, ||Δd||/||d||={dd_ratio:.3e}, '
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
        write_step_times(time_file, step_times)
        total_time = sum(step_times.values())
        print(f'  本步耗时 {step_time:.1f}s, 累计总时间 {total_time:.1f}s '
              f'({total_time / 60.0:.1f} min)')
        if stop_now:
            print(f'{tag} 完成, 加载终止于 step {step}, '
                  f'累计总时间 {total_time:.1f}s ({total_time / 60.0:.1f} min)')
            return

    print(f'{tag} 完成, 累计总时间 {sum(step_times.values()):.1f}s')




def main():
    parser = argparse.ArgumentParser(description='三种孔形板 复现 Yu 2024 第 4.4 节 '
                                                 '(DEM + 相场 staggered)')
    parser.add_argument('--case', type=str, choices=CASES, default=None,
                        help='孔形 C/S/T (缺省运行全部)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--steps', type=int, default=N_STEPS, help='最多运行的加载步数')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help='每次位移子问题最大 epoch 数')
    parser.add_argument('--no-stag', action='store_true', help='每步只做一次位移-相场交替')
    parser.add_argument('--reset', action='store_true', help='忽略已有断点重新开始')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定了 --device cuda, 但 torch.cuda.is_available()=False.')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'计算设备: {device}  (torch {torch.__version__}, '
          f'cuda available: {torch.cuda.is_available()})')

    t_all = time.time()
    cases = [args.case] if args.case else CASES
    print(f'待运行孔形: {cases}')
    for c in cases:
        args.case = c
        run_case(args, device)
    print(f'\n完成, 总耗时 {time.time()-t_all:.1f}s')


if __name__ == '__main__':
    main()
