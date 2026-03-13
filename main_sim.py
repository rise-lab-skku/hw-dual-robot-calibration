# main_sim.py (체인 검증 기능 추가)
import numpy as np
from pathlib import Path
import yaml
from scipy.optimize import least_squares
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 사용자 정의 모듈 임포트 ---
try:
    from solver.initialization import solve_init_two_step_abcd, solve_axyb_dq
    from solver.uncertainty import run_optimization_with_vce_dual, run_optimization_with_vce_dual_bicamera, run_optimization_with_vce_unified
    # [수정] 시뮬레이션 코드에서 프로젝션 함수와 Intrinsics 클래스를 가져옵니다.
    from sim.single_datagen import DivisionIntrinsics, project_division_model, se3
except ImportError as e:
    print(f"오류: 필요한 모듈을 찾을 수 없습니다. ({e})")
    print("      'main_sim.py'와 'sim' 폴더, 'solver' 폴더가 같은 위치에 있는지 확인하세요.")
    exit()

# --- 유틸리티 함수 ---
def to_div_intrinsics(intr):
    """
    intr가 dict이든 DivisionIntrinsics이든 project_division_model에 맞는
    DivisionIntrinsics 인스턴스로 변환해 반환합니다.
    """
    if isinstance(intr, DivisionIntrinsics):
        return intr

    if isinstance(intr, dict):
        # width/height 없을 수 있으니 합리적인 기본값 사용
        width  = intr.get('width', 1280)
        height = intr.get('height', 1024)
        # sx/sy 하나만 들어오는 케이스 대비
        sx = intr.get('sx', intr.get('sy', 1.0))
        sy = intr.get('sy', intr.get('sx', sx))
        return DivisionIntrinsics(
            width=width, height=height,
            c=intr.get('c', 8.0),
            kappa=intr.get('kappa', 0.0),
            sx=sx, sy=sy,
            cx=intr.get('cx', width/2.0),
            cy=intr.get('cy', height/2.0),
        )

    raise TypeError(f"Unsupported intrinsics type: {type(intr)}")

def inv4(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]; Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti
def log_so3(R: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0); theta = np.arccos(cos_theta)
    if abs(theta) < 1e-12: return np.zeros(3)
    return (theta / (2.0 * np.sin(theta))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

# --- 데이터 로딩 함수 (이전과 동일) ---
def load_full_sim_data(data_root: Path) -> dict:
    # ... (이전과 동일한 코드) ...
    cam1_dir = data_root / "cam1" / "poses"; cam2_dir = data_root / "cam2" / "poses"
    if not cam1_dir.exists() or not cam2_dir.exists(): return None
    cam1_files = sorted(cam1_dir.glob("frame_*.yaml")); cam2_files = sorted(cam2_dir.glob("frame_*.yaml"))
    output = {"A_init": [], "B_init": [], "C_init": [], "D_init": [], "T_B1E1_obs": [], "T_C1B_obs": [], "T_B2E2_obs": [], "T_C2B_obs": [], "obj_pts1": [], "img_pts1": [], "obj_pts2": [], "img_pts2": [], "T_B1E1_gt": [], "gt": None}
    for f1, f2 in zip(cam1_files, cam2_files):
        with open(f1, 'r') as f: d1 = yaml.safe_load(f)
        with open(f2, 'r') as f: d2 = yaml.safe_load(f)
        T_B1E1 = np.array(d1["T_Base_to_EE"]); T_C1B = np.array(d1["T_Cam_to_Board"]); T_B2E2 = np.array(d2["T_Base_to_EE"]); T_C2B = np.array(d2["T_Cam_to_Board"])
        output["T_B1E1_obs"].append(inv4(T_B1E1)); output["T_C1B_obs"].append(inv4(T_C1B)); output["T_B2E2_obs"].append(T_B2E2); output["T_C2B_obs"].append(T_C2B)
        output["A_init"].append(T_B1E1); output["B_init"].append(T_C1B); output["C_init"].append(T_B2E2); output["D_init"].append(T_C2B)
        output["obj_pts1"].append(np.array(d1["object_points"])); output["img_pts1"].append(np.array(d1["image_points"])); output["obj_pts2"].append(np.array(d2["object_points"])); output["img_pts2"].append(np.array(d2["image_points"]))
        T_B1E1_gt = np.array(d1["GT"]["T_Base_to_EE"])
        output["T_B1E1_gt"].append(T_B1E1_gt)
        if output["gt"] is None: output["gt"] = d1["GT"]
    return output

# --- 평가 함수 ---
def evaluate_and_print(title: str, X_gt, Y_gt, Z_gt, X_est, Y_est, Z_est):
    # ... (이전과 동일한 코드) ...
    print(f"\n--- {title} ---")
    err_mat_x = inv4(X_gt) @ X_est; t_err_x = np.linalg.norm(err_mat_x[:3, 3]) * 1000.0; r_err_x = np.rad2deg(np.linalg.norm(log_so3(err_mat_x[:3, :3])))
    print(f"[ X (C1->E1) ] Δt={t_err_x:7.4f} mm, ΔR={r_err_x:7.4f} deg")
    err_mat_y = inv4(Y_gt) @ Y_est; t_err_y = np.linalg.norm(err_mat_y[:3, 3]) * 1000.0; r_err_y = np.rad2deg(np.linalg.norm(log_so3(err_mat_y[:3, :3])))
    print(f"[ Y (B2->B1) ] Δt={t_err_y:7.4f} mm, ΔR={r_err_y:7.4f} deg")
    err_mat_z = inv4(Z_gt) @ Z_est; t_err_z = np.linalg.norm(err_mat_z[:3, 3]) * 1000.0; r_err_z = np.rad2deg(np.linalg.norm(log_so3(err_mat_z[:3, :3])))
    print(f"[ Z (C2->E2) ] Δt={t_err_z:7.4f} mm, ΔR={r_err_z:7.4f} deg")
    print("--------------------------------" + "-"*len(title))

# ==========================================================
# ============ [신규] 체인 검증 및 RMSE 계산 함수 ============
# ==========================================================
def rmse_reproject(T_Board_Cam, obj_pts, img_pts, intr):
    """주어진 자세로 3D 포인트를 투영하고, 관측된 2D 포인트와의 RMSE를 계산합니다."""
    intr_obj = to_div_intrinsics(intr)
    proj_pts, visible_mask = project_division_model(obj_pts, T_Board_Cam, intr_obj)
    
    # 유효한 포인트만 필터링
    valid_indices = np.where(visible_mask)[0]
    if len(valid_indices) == 0:
        return None
        
    err = proj_pts[valid_indices] - img_pts[valid_indices]
    return np.sqrt(np.mean(np.sum(err**2, axis=1)))

def verify_kinematic_chain(data, X, Y, Z, intr1):
    """
    초기값(X,Y,Z)과 관측값(A,C,D)으로 운동학적 체인을 구성하여 B를 예측하고,
    실제 관측된 B와의 차이 및 재투영 오차를 비교하여 체인의 무결성을 검증합니다.
    """
    print("\n[단계 1.5] 운동학적 체인 무결성 검증 시작...")
    
    n = len(data['A_init'])
    sum_dt, sum_dr, sum_rmse_chain, sum_rmse_obs = 0.0, 0.0, 0.0, 0.0
    
    for i in range(n):
        # 관측 데이터 (솔버 입력과 동일한 방향)
        A_i, C_i, D_i = data['A_init'][i], data['C_init'][i], data['D_init'][i]
        
        # 체인으로 B 예측: B_pred = inv(X) @ inv(A) @ Y @ C @ Z @ D
        B_pred = inv4(X) @ inv4(A_i) @ Y @ C_i @ Z @ D_i
        
        # 비교 대상인 실제 관측값 B
        B_obs = data['B_init'][i]
        
        # [1] 변환 행렬 간의 차이 계산
        Terr = inv4(B_obs) @ B_pred
        twist = log_so3(Terr[:3,:3]) # 회전 오차
        trans = Terr[:3,3]          # 위치 오차
        dt_mm = np.linalg.norm(trans) * 1000.0
        dr_deg = np.rad2deg(np.linalg.norm(twist))
        sum_dt += dt_mm
        sum_dr += dr_deg

        # [2] 재투영 오차(RMSE) 계산
        obj_pts = data['obj_pts1'][i]
        img_pts = data['img_pts1'][i]
        rmse_chain = rmse_reproject(B_pred, obj_pts, img_pts, intr1)
        rmse_obs = rmse_reproject(B_obs, obj_pts, img_pts, intr1)
        sum_rmse_chain += rmse_chain
        sum_rmse_obs += rmse_obs

        if i < 5: # 처음 5개 프레임만 상세 출력
            print(f"  - Frame {i:02d}: Δt={dt_mm:7.2f} mm, ΔR={dr_deg:6.2f} deg | RMSE(Chain)={rmse_chain:.3f}px, RMSE(Obs)={rmse_obs:.3f}px")

    print("---------------------------------------------------------------------------------")
    print(f"  [평균] 변환 차이: Δt={sum_dt/n:.2f} mm, ΔR={sum_dr/n:.2f} deg")
    print(f"  [평균] RMSE(Chain)={sum_rmse_chain/n:.3f}px, RMSE(Observed)={sum_rmse_obs/n:.3f}px")
    print("---------------------------------------------------------------------------------")
    print("✅ 체인 무결성 검증 완료.")

def reprojection_error_func(params, object_points, image_points_list, num_images):
    """
    [수정됨] SciPy least_squares를 위한 재투영 오차 계산 함수.
    project_division_model을 사용하여 벡터화된 계산을 수행합니다.
    """
    # 1. 파라미터 벡터를 변수로 분해
    c, kappa, sx, sy, cx, cy = params[0:6]
    # DivisionIntrinsics 객체 생성 (width/height는 기본값 사용)
    intr = DivisionIntrinsics(c=c, kappa=kappa, sx=sx, sy=sy, cx=cx, cy=cy)
    
    poses_params = params[6:].reshape((num_images, 6))
    all_errors = []

    # 2. 모든 이미지에 대해 재투영 오차 계산
    for i in range(num_images):
        rvec, tvec = poses_params[i, :3], poses_params[i, 3:]
        R, _ = cv2.Rodrigues(rvec)
        # PnP 결과는 Cam->Board 이므로, Board->Cam 변환을 위해 역행렬을 취합니다.
        T_Board_Cam = se3(R, tvec)
        
        # 3. 모든 3D 포인트를 한 번에 2D 이미지 평면에 투영
        proj_pts, visible_mask = project_division_model(object_points, T_Board_Cam, intr)
        
        # 4. 모든 포인트에 대한 오차 계산
        observed_pts = image_points_list[i].reshape(-1, 2)
        error = proj_pts - observed_pts
        # ★★★★★ 핵심 수정 사항 ★★★★★
        # 보이지 않는 점들로 인해 발생한 NaN 오차를 0으로 대체합니다.
        # 이렇게 하면 오차 배열의 크기가 항상 일정하게 유지됩니다.
        error[np.isnan(error)] = 0.0
        # ★★★★★★★★★★★★★★★★★★★
        all_errors.append(error.flatten())
            
    # 모든 오차를 하나의 벡터로 결합하여 반환
    return np.concatenate(all_errors) if all_errors else np.array([])


########################################################################################################
########################################################################################################

def solve_ax_zb_kronecker_robust(A_list, B_list):
    """
    Kronecker Product를 이용하여 AX=ZB 방정식을 푸는 강건한(Robust) 함수.

    이전 버전의 sign ambiguity 문제를 해결하기 위해, SVD로 구한 해 벡터의
    부호를 검사하고, trace가 더 큰(물리적으로 더 타당한) 쪽을 선택합니다.

    Args:
        A_list (list): 4x4 동차 변환 행렬 A의 리스트.
        B_list (list): 4x4 동차 변환 행렬 B의 리스트.

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]:
            - 성공 시 (X, Z) 4x4 동차 변환 행렬 튜플.
            - 실패 시 (None, None).
            
    [중요] 만약 실제 시스템이 A * X_real^-1 = Z_real * B 라면,
    이 함수가 반환하는 X는 X_real^-1 이고, Z는 Z_real 입니다.
    따라서 X_real을 얻으려면 반환된 X에 역행렬을 취해야 합니다.
    X_real = np.linalg.inv(X)
    """
    if len(A_list) != len(B_list) or len(A_list) < 3:
        print("Error: 최소 3쌍 이상의 A, B 행렬 데이터가 필요합니다.")
        return None, None

    n = len(A_list)
    M_rot = np.zeros((9 * n, 18))

    # --- 1단계: 회전 (Rotation) 계산 ---
    for i in range(n):
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        M_rot[9 * i : 9 * (i + 1), :] = np.hstack([
            np.kron(np.eye(3), Ra),
            -np.kron(Rb.T, np.eye(3))
        ])

    _, _, Vt = np.linalg.svd(M_rot)
    r_vec = Vt[-1, :]

    vec_Rx_raw = r_vec[:9]
    vec_Rz_raw = r_vec[9:]
    
    Rx_est_raw = vec_Rx_raw.reshape(3, 3)
    Rz_est_raw = vec_Rz_raw.reshape(3, 3)

    # --- [개선점] Sign Ambiguity 해결 ---
    # 참 회전 행렬은 trace가 양수일 가능성이 높음 (특히 작은 회전의 경우 +3에 가까움)
    # -R 행렬은 trace가 음수일 가능성이 높음
    # 두 행렬의 trace 합을 비교하여 해 벡터의 전체 부호를 결정
    if np.trace(Rx_est_raw) + np.trace(Rz_est_raw) < 0:
        vec_Rx = -vec_Rx_raw
        vec_Rz = -vec_Rz_raw
    else:
        vec_Rx = vec_Rx_raw
        vec_Rz = vec_Rz_raw

    Rx_est = vec_Rx.reshape(3, 3)
    Rz_est = vec_Rz.reshape(3, 3)

    # 가장 가까운 SO(3) 행렬(회전 행렬)을 찾음
    U, _, Vt = np.linalg.svd(Rx_est)
    Rx = U @ Vt
    
    U, _, Vt = np.linalg.svd(Rz_est)
    Rz = U @ Vt

    # --- 2단계: 이동 (Translation) 계산 ---
    M_trans = np.zeros((3 * n, 6))
    d_trans = np.zeros((3 * n, 1))

    for i in range(n):
        Ra = A_list[i][:3, :3]
        ta = A_list[i][:3, 3]
        tb = B_list[i][:3, 3]
        
        M_trans[3 * i : 3 * (i + 1), :] = np.hstack([Ra, -np.eye(3)])
        d_trans[3 * i : 3 * (i + 1)] = (Rz @ tb - ta).reshape(3, 1)

    t_vec, _, _, _ = np.linalg.lstsq(M_trans, d_trans, rcond=None)
    tx = t_vec[:3].flatten()
    tz = t_vec[3:].flatten()

    # --- 최종 결과 조립 ---
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx

    Z = np.eye(4)
    Z[:3, :3] = Rz
    Z[:3, 3] = tz

    return X, Z

########################################################################################################
########################################################################################################


# ==========================================================
# =================== [신규] 3D 시각화 함수 ===================
# ==========================================================
def plot_frame(ax, T, length=0.1, text=""):
    """하나의 4x4 변환 행렬을 3D 축에 좌표계로 그립니다."""
    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2], R[0, i], R[1, i], R[2, i],
                  length=length, color=colors[i], arrow_length_ratio=0.2)
    if text:
        ax.text(origin[0], origin[1], origin[2], f'  {text}', color='k')

def visualize_kinematics_3d(data):
    """데이터 로딩 직후 로봇 자세와 월드 객체들을 3D로 시각화합니다."""
    print("\n[단계 0] 생성된 데이터 3D 시각화 시작...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 로봇 베이스 좌표계 (월드 원점)
    T_world_base1 = np.eye(4)
    plot_frame(ax, T_world_base1, length=0.2, text="Base1 (World)")

    # 2. 캘리브레이션 보드 위치
    T_base1_board = np.array(data['gt']['B']) # 원본 코드의 'B'가 보드 위치
    plot_frame(ax, T_base1_board, length=0.15, text="Board")

    # 3. 로봇 자세(End-Effector) 그리기
    gt_poses = data['T_B1E1_gt']
    obs_poses = data['T_B1E1_obs'] # 데이터 로딩 시 역행렬을 취했으므로, 다시 역행렬을 취해 원래의 T_Base_to_EE로 변환
    obs_poses_original = [inv4(T) for T in obs_poses]

    for i, T_gt in enumerate(gt_poses):
        plot_frame(ax, T_gt, length=0.05)
    
    for i, T_obs in enumerate(obs_poses_original):
        # 관측된 자세는 다른 색상/스타일로 구분 (여기서는 GT와 동일하게 그림)
        # 좀 더 명확한 구분을 위해 선 스타일 등을 바꿀 수 있습니다.
        plot_frame(ax, T_obs, length=0.05)

    # GT와 OBS의 위치를 점으로 찍어 차이를 명확하게 보여줌
    gt_positions = np.array([T[:3, 3] for T in gt_poses])
    obs_positions = np.array([T[:3, 3] for T in obs_poses_original])
    ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], color='g', marker='o', s=10, label='GT Poses')
    ax.scatter(obs_positions[:, 0], obs_positions[:, 1], obs_positions[:, 2], color='b', marker='^', s=10, label='Observed Poses')

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('3D Visualization of Robot Poses')
    ax.legend()
    # 축의 스케일을 동일하게 맞춰 왜곡을 방지
    max_range = np.array([gt_positions.max(axis=0), obs_positions.max(axis=0)]).max(axis=0)
    min_range = np.array([gt_positions.min(axis=0), obs_positions.min(axis=0)]).min(axis=0)
    ax.set_box_aspect(max_range - min_range)

    print("✅ 3D 시각화 그래프를 표시합니다. 창을 닫으면 다음 단계가 진행됩니다.")
    plt.show()

############
def hat(v: np.ndarray) -> np.ndarray:
    """3x1 벡터를 3x3 skew-symmetric 행렬로 변환합니다."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=float)

def so3_exp(w: np.ndarray) -> np.ndarray:
    """
    로드리게스 공식을 사용하여 so(3) 벡터(축-각도)를
    SO(3) 회전 행렬로 변환합니다.
    """
    w = np.asarray(w, dtype=float).flatten()
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    axis = w / theta
    K = hat(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """3x3 회전 행렬과 3x1 이동 벡터로 4x4 변환 행렬을 만듭니다."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def perturb_transform(
    T_gt: np.ndarray,
    sigma_angle_deg: float,
    sigma_trans_m: float,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    주어진 4x4 변환 행렬(T_gt)에 지정된 수준의 회전 및 이동 노이즈를 추가하여
    초기 추정값으로 사용할 새로운 변환 행렬을 생성합니다.

    Args:
        T_gt (np.ndarray): 노이즈를 추가할 4x4 Ground Truth 변환 행렬.
        sigma_angle_deg (float): 회전 노이즈의 표준편차 (단위: degrees).
        sigma_trans_m (float): 이동 노이즈의 표준편차 (단위: meters).
        rng (np.random.Generator, optional): 재현성을 위한 NumPy 난수 생성기.

    Returns:
        np.ndarray: 노이즈가 추가된 4x4 변환 행렬.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. 회전(Rotation) 노이즈 생성
    #    - 각 축에 대해 표준편차만큼의 랜덤 회전 각도(라디안) 생성
    sigma_angle_rad = np.deg2rad(sigma_angle_deg)
    w_noise = rng.normal(0, sigma_angle_rad, 3)  # so(3) 공간에서의 랜덤 벡터
    R_noise = so3_exp(w_noise)                   # SO(3) 랜덤 회전 행렬

    # 2. 이동(Translation) 노이즈 생성
    t_noise = rng.normal(0, sigma_trans_m, 3)

    # 3. 노이즈 변환 행렬 생성
    T_noise = se3(R_noise, t_noise)

    # 4. GT 행렬에 노이즈 변환 행렬을 곱하여 최종 결과 생성
    #    (T_noise @ T_gt는 T_gt를 기준으로 하는 글로벌 좌표계에서 노이즈를 가하는 것과 같음)
    T_perturbed = T_noise @ T_gt
    
    return T_perturbed
###########


# ==================================
# ============ 메인 실행부 ============
# ==================================
def main():
    # --- 1. 데이터 로딩 ---
    sim_root = Path("./data_dual_cam_realistic")
    data = load_full_sim_data(sim_root)
    if not data: print(f"오류: '{sim_root}'에서 데이터를 불러오지 못했습니다."); return
    print(f"✅ 성공: '{sim_root}' 에서 {len(data['A_init'])}개의 데이터 쌍을 불러왔습니다.")

    visualize_kinematics_3d(data)

    X_gt, Y_gt, Z_gt = np.array(data['gt']['X']), np.array(data['gt']['Y']), np.array(data['gt']['Z'])

    Board_gt = np.array(data['gt']['B'])

    def rot_deg_from_R(R):
        return np.rad2deg(np.linalg.norm(log_so3(R)))

    def se3_delta_stats(B_obs, B_pred):
        """inv(B_obs)@B_pred 의 Δt(mm), ΔR(deg), Frobenius residual 반환"""
        Terr = inv4(B_obs) @ B_pred
        dt_mm = float(np.linalg.norm(Terr[:3, 3]) * 1000.0)
        dr_deg = float(rot_deg_from_R(Terr[:3, :3]))
        fro_res = float(np.linalg.norm(B_obs - B_pred, ord='fro'))
        return dt_mm, dr_deg, fro_res

    print("\n[체인 무결성/잔차 검사]  (GT vs DQ 추정)")
    # --- 프레임별 잔차 출력 (처음 5개) & 평균 집계 ---
    sum_gt_t, sum_gt_r, sum_gt_f = 0.0, 0.0, 0.0
    sum_est_t, sum_est_r, sum_est_f = 0.0, 0.0, 0.0
    for i, (A, B) in enumerate(zip(data['T_B1E1_obs'], data['B_init'])):
        # GT로 예측
        LHS = A @ Board_gt
        RHS = inv4(X_gt) @ B
        # B_pred_gt = X_gt @ A @ Board_gt
        dt_gt, dr_gt, fro_gt = se3_delta_stats(LHS, RHS)
        sum_gt_t += dt_gt; sum_gt_r += dr_gt; sum_gt_f += fro_gt

        if i < 5:
            print(f"  - Frame {i:02d} | "
                f"GT: Δt={dt_gt:6.3f} mm, ΔR={dr_gt:6.3f} deg, ‖Δ‖F={fro_gt:7.4f}")
            
    # --- 2. 초기값 계산 ---
    print("\n[단계 1] 외부 파라미터(Extrinsics) 초기값 계산 시작...")
    try:
        X_init_C1E1, Y_init_B2B1, Z_init_C2E2 = solve_init_two_step_abcd(data['A_init'], data['B_init'], data['C_init'], data['D_init'])
        print("✅ 성공: 외부 파라미터 초기값 계산 완료.")
    except Exception as e:
        print(f"❌ 오류: 초기값 계산 실패: {e}"); return
    
    X_est, Z_est = solve_ax_zb_kronecker_robust(data['T_B1E1_obs'], data['B_init'])

    if X_est is not None:
        print("--- AX=ZB Solver 결과 ---")
        np.set_printoptions(precision=3, suppress=True)
        print("추정된 X:\n", inv4(Z_est))
        print("\n실제 X (GT):\n", X_gt)
        print("\n추정된 Z:\n", X_est)
        print("\n실제 Z (GT):\n", Board_gt)
        evaluate_and_print("Initial Guess (초기값)", Board_gt, X_gt, Z_gt, X_est, inv4(Z_est), np.eye(4))
       
    # --- 3. 최적화 준비 (내부 파라미터 초기값 설정) ---
    # Ground Truth Intrinsic
    intrinsics_gt = DivisionIntrinsics()
    
    # Init Intrinsic
    intrinsics_datasheet = DivisionIntrinsics(width=1280, height=1024, c=8.4303, kappa=0.99992e-3, sx=5.20997e-3, sy=5.20e-3, cx=659.99, cy=481.96)

    intrinsics_init_dict = {
        'c': intrinsics_datasheet.c, 'kappa': intrinsics_datasheet.kappa, 'sx': intrinsics_datasheet.sx, 'sy': intrinsics_datasheet.sy,
        'cx': intrinsics_datasheet.cx, 'cy': intrinsics_datasheet.cy,
        'width': intrinsics_datasheet.width, 'height': intrinsics_datasheet.height,
    }

    intrinsics_gt_dict = {
        'c': intrinsics_gt.c, 'kappa': intrinsics_gt.kappa, 'sx': intrinsics_gt.sx, 'sy': intrinsics_gt.sy,
        'cx': intrinsics_gt.cx, 'cy': intrinsics_gt.cy,
        'width': intrinsics_gt.width, 'height': intrinsics_gt.height,
    }
    
    print("\n[단계 2] 내부 파라미터(Intrinsics) 초기값 설정 완료.")

    print(intrinsics_gt_dict)
    print(intrinsics_init_dict)

    # X_EC 초기값
    angle_error_level = 0.2  # degrees
    trans_error_level = 0.005 # meters

    # 3. 노이즈를 추가하여 초기 추정값 생성
    X_init = perturb_transform(
        T_gt=X_gt,
        sigma_angle_deg=angle_error_level,
        sigma_trans_m=trans_error_level
    )

    Board_init = perturb_transform(
        T_gt=Board_gt,
        sigma_angle_deg=angle_error_level,
        sigma_trans_m=trans_error_level
    )

    # --- 3. VCE 최적화 실행 ---
    print("\n[단계 3] AX=ZB VCE 최적화 실행...")
    X_EC_final, X_WB_final, T_BE_list_final, intrinsics_final = run_optimization_with_vce_unified(
        model_type='division',
        T_ee_cam_init=X_init,
        T_base_board_init=Board_init,
        T_be_list_init=data["T_B1E1_obs"],
        img_pts_list=data['img_pts1'],
        obj_pts_list=data['obj_pts1'],
        T_be_list_obs=data["T_B1E1_obs"],
        intrinsics_init=intrinsics_init_dict,
        sigma_image_px=0.1,
        sigma_angle_deg=0.1,
        sigma_trans_mm=1.0,
        max_vce_iter=15,
        max_param_iter=15,
        term_thresh=1e-6,
        is_target_based=True,
        estimate_ec=True,
        estimate_wb=True,
        estimate_be=True,
        estimate_intrinsics=False,
        is_scara=False
    )

    print("Final intrinsics")
    print(intrinsics_final)

    evaluate_and_print("Initial Guess (초기값)", X_gt, Board_gt, np.eye(4), X_init, Board_init, np.eye(4))
    evaluate_and_print("Initial Guess (최적화값)", X_gt, Board_gt, np.eye(4), X_EC_final, X_WB_final, np.eye(4))

    num_poses = len(data["T_B1E1_gt"])
    sum_dt_obs, sum_dr_obs = 0.0, 0.0
    sum_dt_final, sum_dr_final = 0.0, 0.0

    print(f"총 {num_poses}개의 자세에 대해 오차를 계산합니다.")
    print("=" * 50)
    print("  - Frame | Obs Err (t, R)   | Final Err (t, R)")
    print("=" * 50)

    for i in range(num_poses):
        # 각 리스트에서 해당 프레임의 변환 행렬을 가져옴
        T_gt = data["T_B1E1_gt"][i]
        T_obs = data['T_B1E1_obs'][i]
        T_final = T_BE_list_final[i]

        # 1. 관측값(obs) vs Ground Truth(gt) 오차 계산
        dt_obs, dr_obs, _ = se3_delta_stats(T_obs, T_gt)
        sum_dt_obs += dt_obs
        sum_dr_obs += dr_obs

        # 2. 최종 결과(final) vs Ground Truth(gt) 오차 계산
        dt_final, dr_final, _ = se3_delta_stats(T_final, T_gt)
        sum_dt_final += dt_final
        sum_dr_final += dr_final
        
        # 처음 5개 프레임에 대한 상세 결과 출력
        if i < 5:
            print(f"  - {i:05d} | "
                f"{dt_obs:6.3f} mm, {dr_obs:6.3f} deg | "
                f"{dt_final:6.3f} mm, {dr_final:6.3f} deg")

    # 평균 오차 계산
    avg_dt_obs = sum_dt_obs / num_poses
    avg_dr_obs = sum_dr_obs / num_poses
    avg_dt_final = sum_dt_final / num_poses
    avg_dr_final = sum_dr_final / num_poses

    print("...")
    print("=" * 50)
    print("\n--- 최종 평균 오차 요약 ---")
    print(f"  [최적화 이전] 관측된 자세 (Observed vs. GT):")
    print(f"    - 평균 이동 오차 (Δt): {avg_dt_obs:.4f} mm")
    print(f"    - 평균 회전 오차 (ΔR): {avg_dr_obs:.4f} deg\n")

    print(f"  [최적화 이후] 보정된 자세 (Final vs. GT):")
    print(f"    - 평균 이동 오차 (Δt): {avg_dt_final:.4f} mm")
    print(f"    - 평균 회전 오차 (ΔR): {avg_dr_final:.4f} deg\n")

    t_improvement = avg_dt_obs - avg_dt_final
    r_improvement = avg_dr_obs - avg_dr_final

    print(f"--- 개선량 ---")
    print(f"  - 이동 정확도 개선: {t_improvement:+.4f} mm")
    print(f"  - 회전 정확도 개선: {r_improvement:+.4f} deg")
    print("-" * 25)

    return

    # --- 4. VCE 최적화 실행 ---
    print("\n[단계 3] VCE 최적화 시작...")
    # try:
    #     (X_final_E1C1, Y_final_B1B2, Z_final_E2C2, 
    #      _, _, _, intrinsics_final_list) = run_optimization_with_vce_dual(
    #         model_type='division',
    #         # 초기값: (E1->C1), (B1->B2), (E2->C2) 방향으로 전달
    #         X1_EC_init=inv4(X_init_C1E1),
    #         T_B1B2_init=Y_init_B2B1,
    #         E2_C2_init=Z_init_C2E2,
    #         # 로봇 자세 리스트: (E1->B1), (B2->E2), (C2->B) 방향으로 전달
    #         T_E1B1_list_init=data['T_B1E1_obs'], 
    #         T_E2B2_list_init=data['C_init'],
    #         T_C2B_list_init=data['D_init'],
    #         # 포인트 리스트
    #         img_pts_list=data['img_pts1'],
    #         obj_pts_list=data['obj_pts1'],
    #         # 관측값 (초기값과 동일한 값을 사용)
    #         T_E1B1_list_obs=data['T_B1E1_obs'],
    #         T_E2B2_list_obs=data['C_init'],
    #         T_C2B_list_obs=None,
    #         # 내부 파라미터 및 노이즈 설정
    #         intrinsics_init=intrinsics_init_dict,
    #         sigma_image_px=0.1,
    #         sigma_angle_deg=0.1,
    #         sigma_trans_mm=1.0,
    #         # 최적화 옵션
    #         max_vce_iter=10,
    #         max_param_iter=15,
    #         term_thresh=1e-6,
    #         estimate_x1ec=True,
    #         estimate_b1b2=True,
    #         estimate_e2c2=True,
    #         estimate_e1b1=True,
    #         estimate_b2e2=True,
    #         estimate_c2b=False,
    #         estimate_intrinsics=False,
    #         include_sy=True,
    #         is_scara_x1=False,
    #     )
    #     print("✅ 성공: VCE 최적화 완료.")
    # except Exception as e:
    #     print(f"❌ 오류: 최적화 실패: {e}"); return
    
    try:
        (
            X1_EC_bi, T_B1B2_bi, E2_C2_bi,
            T_E1B1_list_bi, T_B2E2_list_bi, T_C2B_list_bi, T_C1B_list_bi,
            intr1_final, intr2_final
        ) = run_optimization_with_vce_dual_bicamera(
            model_type='division',
            # 전역 초기변수
            X1_EC_init=inv4(X_init_C1E1),
            T_B1B2_init=Y_init_B2B1,
            E2_C2_init=Z_init_C2E2,
            # per-pose 초기값
            T_E1B1_list_init=data['T_B1E1_obs'], 
            T_E2B2_list_init=data['C_init'],
            T_C2B_list_init=data['D_init'],
            T_C1B_list_init=data['B_init'],     # ^C1 T_B (cam2 블록)
            # 관측(이미지/3D)
            obj_pts_list=data['obj_pts1'],       
            img1_pts_list=data['img_pts1'],             # cam1 2D
            img2_pts_list=data['img_pts2'],             # cam2 2D
            # 포즈 관측(Fictitious obs; 보통 보드는 고정)
            T_E1B1_list_obs=data['T_B1E1_obs'],
            T_E2B2_list_obs=data['C_init'],
            T_C2B_list_obs=data['D_init'],
            T_C1B_list_obs=data['B_init'],
            # 카메라별 division intrinsics
            intr1_init=intrinsics_init_dict,
            intr2_init=intrinsics_init_dict,
            # 노이즈 (초기 분산)
            sigma_image_px=1.0,
            sigma_angle_deg=0.1,
            sigma_trans_mm=1.0,
            # 반복/LM
            max_vce_iter=15,
            max_param_iter=15,
            term_thresh=1e-6,
            # 추정 플래그
            estimate_x1ec=True,
            estimate_b1b2=True,
            estimate_e2c2=True,
            estimate_e1b1=True,
            estimate_b2e2=True,
            estimate_c2b=False,   # ^C2 T_B (보통 고정)
            estimate_c1b=False,   # ^C1 T_B (보통 고정)
            estimate_intr1=False,  # cam1 내부파라미터 추정 여부
            estimate_intr2=False,  # cam2 내부파라미터 추정 여부
            include_sy1=False,
            include_sy2=False,
            is_scara_x1=False,
        )
        print("✅ 성공: VCE 최적화 완료.")
    except Exception as e:
        print(f"❌ 오류: 최적화 실패: {e}"); return
    
    print(intr1_final)
    print(intr2_final)

    # --- 5. 최종 결과 비교 및 평가 ---
    print("\n\n" + "="*52); print("========= 🔬 최종 캘리브레이션 결과 비교 🔬 ========="); print("="*52)
    evaluate_and_print("Initial Guess (초기값)", X_gt, Y_gt, Z_gt, inv4(X_init_C1E1), inv4(Y_init_B2B1), inv4(Z_init_C2E2))
    evaluate_and_print("Final Optimized (최적화 결과)", X_gt, Y_gt, Z_gt, X1_EC_bi, inv4(T_B1B2_bi), inv4(E2_C2_bi))
    # evaluate_and_print("Final Optimized (최적화 결과)", X_gt, Y_gt, Z_gt, inv4(X_final_E1C1), Y_final_B1B2, Z_final_E2C2)
    # verify_kinematic_chain(
    #     data,
    #     X_init_C1E1,      # ^C1 T_E1 로 맞춰서 전달
    #     Y_init_B2B1,           # ^B1 T_B2
    #     Z_init_C2E2,            # ^E2 T_C2
    #     intrinsics_gt  # Cam1 intrinsics
    # )
    
    verify_kinematic_chain(
        data,
        inv4(X1_EC_bi),      # ^C1 T_E1 로 맞춰서 전달
        T_B1B2_bi,           # ^B1 T_B2
        E2_C2_bi,            # ^E2 T_C2
        to_div_intrinsics(intrinsics_datasheet)  # Cam1 intrinsics
    )

    # verify_kinematic_chain(data, X_init_C1E1, Y_init_B2B1, Z_init_C2E2, intrinsics_init_dict)
    # verify_kinematic_chain(data, inv4(X_final_E1C1), Y_final_B1B2, Z_final_E2C2, intrinsics_final_list)
    
    intrinsics_final1 = intrinsics_final_list[0] if isinstance(intrinsics_final_list, list) else intrinsics_final_list
    print("\n--- Intrinsics (Cam1) ---")
    print(f"  - GT      : c={intrinsics_datasheet.c:.4f}, kappa={intrinsics_datasheet.kappa:<.8f}, cx={intrinsics_datasheet.cx:.4f}, cy={intrinsics_datasheet.cy:.4f}")
    print(f"  - Initial : c={intrinsics_init_dict['c']:.4f}, kappa={intrinsics_init_dict['kappa']:<.8f}, cx={intrinsics_init_dict['cx']:.4f}, cy={intrinsics_init_dict['cy']:.4f}")
    print(f"  - Final   : c={intrinsics_final1['c']:.4f}, kappa={intrinsics_final1['kappa']:<.8f}, cx={intrinsics_final1['cx']:.4f}, cy={intrinsics_final1['cy']:.4f}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()