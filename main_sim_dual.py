# main_sim.py (체인 검증 기능 추가)
import numpy as np
from pathlib import Path
import yaml
from scipy.optimize import least_squares
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
from dataclasses import asdict

# --- 사용자 정의 모듈 임포트 ---
try:
    from solver.initialization import solve_init_two_step_abcd
    from solver.uncertainty import (run_optimization_with_vce_dual, 
                                    run_optimization_with_vce_dual_bicamera, 
                                    run_optimization_with_vce_unified,
                                    run_optimization_with_vce_shared_target_v2)
    from solver.lie import LieOptimizationSolverAXBYCZD as LieOptimizationSolver
    from sim.single_datagen import DivisionIntrinsics, project_division_model, se3
except ImportError as e:
    print(f"오류: 필요한 모듈을 찾을 수 없습니다. ({e})")
    print("      'main_sim.py'와 'sim' 폴더, 'solver' 폴더가 같은 위치에 있는지 확인하세요.")
    exit()

# --- 유틸리티 함수 ---
def _print_T(name, T):
    np.set_printoptions(precision=4, suppress=True)
    print(f"{name} =\n{T}\n")

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
    output = {"A_init": [], "B_init": [], "C_init": [], "D_init": [], "T_B1E1_obs": [], "T_C1B_obs": [], "T_B2E2_obs": [], "T_C2B_obs": [], "obj_pts1": [], "img_pts1": [], "obj_pts2": [], "img_pts2": [], "T_B1E1_gt": [], "T_B2E2_gt": [], "gt": None}
    for f1, f2 in zip(cam1_files, cam2_files):
        with open(f1, 'r') as f: d1 = yaml.safe_load(f)
        with open(f2, 'r') as f: d2 = yaml.safe_load(f)
        T_B1E1 = np.array(d1["T_Base_to_EE"]); T_C1B = np.array(d1["T_Cam_to_Board"]); T_B2E2 = np.array(d2["T_Base_to_EE"]); T_C2B = np.array(d2["T_Cam_to_Board"])
        output["T_B1E1_obs"].append(inv4(T_B1E1)); output["T_C1B_obs"].append(inv4(T_C1B)); output["T_B2E2_obs"].append(inv4(T_B2E2)); output["T_C2B_obs"].append(T_C2B)
        output["A_init"].append(T_B1E1); output["B_init"].append(T_C1B); output["C_init"].append(T_B2E2); output["D_init"].append(T_C2B)
        output["obj_pts1"].append(np.array(d1["object_points"])); output["img_pts1"].append(np.array(d1["image_points"])); output["obj_pts2"].append(np.array(d2["object_points"])); output["img_pts2"].append(np.array(d2["image_points"]))
        T_B1E1_gt = np.array(d1["GT"]["T_Base1_to_EE1"])
        T_B2E2_gt = np.array(d2["GT"]["T_Base2_to_EE2"])
        output["T_B1E1_gt"].append(T_B1E1_gt)
        output["T_B2E2_gt"].append(T_B2E2_gt)
        T_C1B_gt = np.array(d1["GT"]["T_Cam1_to_Board"])
        T_C2B_gt = np.array(d2["GT"]["T_Cam2_to_Board"])
        output["T_C1B_gt"] = T_C1B_gt
        output["T_C2B_gt"] = T_C2B_gt
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
    rng: Optional[np.random.Generator] = None
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

    # visualize_kinematics_3d(data)

    X_gt, Y_gt, Z_gt = np.array(data['gt']['X']), np.array(data['gt']['Y']), np.array(data['gt']['Z'])

    # Board_gt = np.array(data['gt']['B'])

    def rot_deg_from_R(R):
        return np.rad2deg(np.linalg.norm(log_so3(R)))

    def se3_delta_stats(B_obs, B_pred):
        """inv(B_obs)@B_pred 의 Δt(mm), ΔR(deg), Frobenius residual 반환"""
        Terr = inv4(B_obs) @ B_pred
        dt_mm = float(np.linalg.norm(Terr[:3, 3]) * 1000.0)
        dr_deg = float(rot_deg_from_R(Terr[:3, :3]))
        fro_res = float(np.linalg.norm(B_obs - B_pred, ord='fro'))
        return dt_mm, dr_deg, fro_res

    # print("\n[체인 무결성/잔차 검사]  (GT vs DQ 추정)")
    # # --- 프레임별 잔차 출력 (처음 5개) & 평균 집계 ---
    # sum_gt_t, sum_gt_r, sum_gt_f = 0.0, 0.0, 0.0
    # sum_est_t, sum_est_r, sum_est_f = 0.0, 0.0, 0.0
    # for i, (A, B) in enumerate(zip(data['T_B1E1_obs'], data['B_init'])):
    #     # GT로 예측
    #     LHS = A @ Board_gt
    #     RHS = inv4(X_gt) @ B
    #     # B_pred_gt = X_gt @ A @ Board_gt
    #     dt_gt, dr_gt, fro_gt = se3_delta_stats(LHS, RHS)
    #     sum_gt_t += dt_gt; sum_gt_r += dr_gt; sum_gt_f += fro_gt

    #     if i < 5:
    #         print(f"  - Frame {i:02d} | "
    #             f"GT: Δt={dt_gt:6.3f} mm, ΔR={dr_gt:6.3f} deg, ‖Δ‖F={fro_gt:7.4f}")
            
    # (선택) 3D 시각화
    try:
        visualize_kinematics_3d(data)
    except Exception as e:
        print(f"[경고] 3D 시각화 건너뜀: {e}")

    # --- 2) GT 변환 로드 ---
    X_gt = np.array(data['gt']['X'])  # C1->E1
    Y_gt = np.array(data['gt']['Y'])  # B2->B1
    Z_gt = np.array(data['gt']['Z'])  # C2->E2

    # --- 3) 초기값(폐형식) 계산: Two-step DQ/Closed-form ---
    print("\n[단계 1] 외부 파라미터(Extrinsics) 초기값 계산 시작...")
    try:
        # solve_init_two_step_abcd 입력은 4x4 행렬 리스트들
        X_est, Y_est, Z_est = solve_init_two_step_abcd(
            data['A_init'],  # A_i
            data['B_init'],  # B_i
            data['C_init'],  # C_i
            data['D_init']   # D_i
        )
        print("✅ 성공: 외부 파라미터 초기값 계산 완료.")
    except Exception as e:
        print(f"❌ 오류: 초기값 계산 실패: {e}")
        return

    # --- 4) 결과 요약 출력 ---
    _print_T("X_est (C1->E1)", X_est)
    _print_T("Y_est (B2->B1)", Y_est)
    _print_T("Z_est (C2->E2)", Z_est)

    # --- 5) GT 대비 오차 출력 ---
    evaluate_and_print("Initial Guess (Two-step initialization)",
                       X_gt, Y_gt, Z_gt,
                       inv4(X_est), inv4(Y_est), inv4(Z_est))

    # --- 6) (선택) 체인 무결성/재투영 RMSE 검증 ---
    # intrinsics 가 있으면 재투영까지 검증. 없으면 변환 오차만 본다.
    intr1 = data['gt'].get('intrinsics_cam1', None)
    try:
        if intr1 is not None:
            verify_kinematic_chain(data, X_est, Y_est, Z_est, intr1)
        else:
            print("\n[알림] intrinsics_cam1 미제공 → 재투영 RMSE 검증은 건너뜀(변환 오차만 확인).")
    except Exception as e:
        print(f"[경고] 체인 무결성 검증 중 예외: {e}")

    print("\n[단계 2] Lie 그룹 최적화 시작...")
    
    # LieOptimizationSolver는 A, B, C, D 행렬들과
    # X, Y, Z의 초기 추정값을 직접 입력으로 받습니다.
    # solve_init_two_step_abcd에서 계산된 X_est, Y_est, Z_est가
    # 솔버가 기대하는 초기값 형식(inv(X_phys) 등)과 일치합니다.
    lie_solver = LieOptimizationSolver(
        A=data['A_init'],
        B=data['B_init'],
        C=data['C_init'],
        D=data['D_init'],
        X0=X_est,
        Y0=Y_est,
        Z0=Z_est
        # scipy.optimize.least_squares에 전달할 추가 인자를 여기에 넣을 수 있습니다.
        # 예: ftol=1e-6, xtol=1e-6
    )

    try:
        # 최적화 실행
        lie_results = lie_solver.solve()
        X_lie, Y_lie, Z_lie = lie_results['X'], lie_results['Y'], lie_results['Z']
        print("✅ 성공: Lie 그룹 최적화 완료.")

        # --- 7) 최적화 결과 요약 출력 ---
        print("\n--- 최적화 결과 (Lie Group Optimization) ---")
        _print_T("X_lie_opt (C1->E1)", inv4(X_lie))
        _print_T("Y_lie_opt (B2->B1)", inv4(Y_lie))
        _print_T("Z_lie_opt (C2->E2)", inv4(Z_lie))

        # --- 8) 최적화 결과 GT 대비 오차 출력 ---
        evaluate_and_print("Optimized Result (Lie Group)",
                           X_gt, Y_gt, Z_gt,
                           inv4(X_lie), inv4(Y_lie), inv4(Z_lie))

        # --- 9) 최적화된 결과로 체인 검증 ---
        # 최적화된 결과에 대한 체인 무결성을 검증하여 초기값과 비교합니다.
        # verify_kinematic_chain(data, inv4(X_lie), inv4(Y_lie), inv4(Z_lie), data['gt']['intr1'])

    except Exception as e:
        print(f"❌ 오류: Lie 그룹 최적화 실패: {e}")
        import traceback
        traceback.print_exc() # 디버깅을 위해 에러 traceback 출력
       
    # --- 3. 최적화 준비 (내부 파라미터 초기값 설정) ---
    # Ground Truth Intrinsic
    intrinsics_gt = DivisionIntrinsics()
    
    # Init Intrinsic
    intrinsics_datasheet = DivisionIntrinsics(width=1280, height=1024, c=8.4303, kappa=0.99992e-3, sx=5.20997e-3, sy=5.20e-3, cx=659.99, cy=481.96)
    # intrinsics_datasheet = DivisionIntrinsics(width=1280, height=1024, c=8.0, kappa=1.0e-8, sx=5.20e-3, sy=5.20e-3, cx=640.0, cy=521.0)

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
    
    # --- 5) run_optimization_with_vce_dual_bicamera 호출 ---
    print("\n[단계 2] Dual-Arm Bi-Camera VCE 최적화 실행...")

    X1_EC_init = inv4(X_est)              # ^C1T_E1
    T_B1B2_init = Y_est             # ^B1T_B2
    E2_C2_init = Z_est              # ^E2T_C2

    # per-pose transforms
    # T_E1B1_list_init = data['A_init']  # ^E1T_B1
    T_E1B1_list_init = data['T_B1E1_obs']
    # T_B2E2_list_init = data['C_init']  # ^B2T_E2
    T_E2B2_list_init = data['T_B2E2_obs']
    # T_C1B_list_init = data['T_C1B_gt']
    # T_C2B_list_init = data['T_C2B_gt']

    T_C2B_list_init  = data['D_init']  # ^C2T_B
    T_C1B_list_init  = data['B_init']  # ^C1T_B

    obj_pts_list = data['obj_pts1']   # 체커보드 3D 좌표
    img1_pts_list = data['img_pts1']  # cam1 이미지 포인트
    img2_pts_list = data['img_pts2']  # cam2 이미지 포인트

    #################################################
    ## 실험 1. 양방향 잔차와, 단방향 잔차의 최적화 성능 비교 ##
    #################################################

    print("="*70)
    print("🚀 [실험 1.1] Bi-Camera (카메라 2대) 최적화 시작")
    print("="*70)

    # --- Bi-Camera (카메라 2대) 최적화 호출 ---
    result_bicam = run_optimization_with_vce_dual_bicamera(
        model_type='division',
        X1_EC_init=X1_EC_init,
        T_B1B2_init=T_B1B2_init,
        E2_C2_init=E2_C2_init,
        T_E1B1_list_init=T_E1B1_list_init,
        T_E2B2_list_init=T_E2B2_list_init,
        T_C2B_list_init=T_C2B_list_init,
        T_C1B_list_init=T_C1B_list_init,
        obj_pts_list=obj_pts_list,
        img1_pts_list=img1_pts_list,
        img2_pts_list=img2_pts_list,
        T_E1B1_list_obs=T_E1B1_list_init,
        T_E2B2_list_obs=T_E2B2_list_init,
        intr1_init=intrinsics_init_dict,
        intr2_init=intrinsics_init_dict,
        sigma_image_px=0.1,
        sigma_angle_deg=0.1,
        sigma_trans_mm=1.0,
        max_vce_iter=10,
        max_param_iter=15,
        term_thresh=1e-6,
        estimate_x1ec=True,
        estimate_b1b2=True,
        estimate_e2c2=True,
        estimate_e1b1=True,
        estimate_b2e2=True,
        estimate_c2b=False,
        estimate_c1b=False,
        estimate_intr1=False,
        estimate_intr2=False,
        include_sy1=False,
        include_sy2=False,
        is_scara_x1=False
    )

    (X1_EC_bi, T_B1B2_bi, E2_C2_bi,
    T_E1B1_list_bi, T_E2B2_list_bi, _, _,
    intr1_final_bi, intr2_final_bi) = result_bicam


    print("\n" + "="*70)
    print("🚀 [실험 1.2] Single-Camera (카메라 1대) 듀얼-암 최적화 시작")
    print("="*70)

    # --- Dual-Arm (카메라 1대) 최적화 호출 ---
    # T_C2B_list_init 인자는 카메라1의 보드 관측(T_C1B_list)을 사용해야 함
    result_dual = run_optimization_with_vce_dual(
        model_type='division',
        X1_EC_init=X1_EC_init,
        T_B1B2_init=T_B1B2_init,
        E2_C2_init=E2_C2_init,
        T_E1B1_list_init=T_E1B1_list_init,
        T_E2B2_list_init=T_E2B2_list_init,
        T_C2B_list_init=T_C2B_list_init,
        obj_pts_list=obj_pts_list,
        img_pts_list=img1_pts_list,
        T_E1B1_list_obs=T_E1B1_list_init,
        T_E2B2_list_obs=T_E2B2_list_init,
        intrinsics_init=intrinsics_init_dict,
        sigma_image_px=0.1,
        sigma_angle_deg=0.1,
        sigma_trans_mm=1.0,
        max_vce_iter=10,
        max_param_iter=15,
        term_thresh=1e-6,
        estimate_x1ec=True,
        estimate_b1b2=True,
        estimate_e2c2=True,
        estimate_e1b1=True,
        estimate_b2e2=True,
        estimate_c2b=False,
        estimate_intrinsics=False,
        include_sy=False,
        is_scara_x1=False
    )

    (X1_EC_dual, T_B1B2_dual, E2_C2_dual,
    T_E1B1_list_dual, T_E2B2_list_dual, _,
    intr_dual_final) = result_dual


    # ## 3. 최적화 결과 비교 ##

    print("\n\n" + "="*70)
    print("📊 최종 결과 비교 분석")
    print("="*70)

    # --- 전역 파라미터 GT 대비 오차 비교 ---
    print("\n--- [전역 파라미터 오차 비교 (vs GT)] ---")
    evaluate_and_print("Bi-Camera (2대)",
                    X_gt, Y_gt, Z_gt,
                    X1_EC_bi, inv4(T_B1B2_bi), inv4(E2_C2_bi))

    evaluate_and_print("Single-Camera (1대)",
                    X_gt, Y_gt, Z_gt,
                    X1_EC_dual, inv4(T_B1B2_dual), inv4(E2_C2_dual))


    # --- 카메라 내부 파라미터 비교 ---
    print("\n--- [카메라 내부 파라미터 비교 (vs GT)] ---")
    print(f"GT Intrinsics  : {intrinsics_gt_dict}")
    print(f"Bi-Cam (Cam1)  : {intr1_final_bi}")
    print(f"Bi-Cam (Cam2)  : {intr2_final_bi}")
    print(f"Single-Cam (Cam1): {intr_dual_final}")


    # --- Per-pose 오차 비교 ---
    def _calculate_pose_errors(T_est_list, T_gt_list, inv_T_est=False):
        """포즈 리스트 간의 평균 t, R 오차를 계산하는 헬퍼 함수"""
        num_poses = len(T_gt_list)
        sum_dt, sum_dr = 0.0, 0.0
        for i in range(num_poses):
            T_est = inv4(T_est_list[i]) if inv_T_est else T_est_list[i]
            T_gt = T_gt_list[i]
            dt, dr, _ = se3_delta_stats(T_est, T_gt)
            sum_dt += dt
            sum_dr += dr
        return (sum_dt / num_poses, sum_dr / num_poses)

    # Compare robot pose
    avg_dt_b1e1_bi, avg_dr_b1e1_bi = _calculate_pose_errors(T_E1B1_list_bi, data['T_B1E1_gt'])
    avg_dt_b2e2_bi, avg_dr_b2e2_bi = _calculate_pose_errors(T_E2B2_list_bi, data['T_B2E2_gt'])

    avg_dt_b1e1_dual, avg_dr_b1e1_dual = _calculate_pose_errors(T_E1B1_list_dual, data['T_B1E1_gt'])
    avg_dt_b2e2_dual, avg_dr_b2e2_dual = _calculate_pose_errors(T_E2B2_list_dual, data['T_B2E2_gt'])

    print("\n--- [Per-Pose 평균 오차 비교 (vs GT)] ---")
    print("                     |  Bi-Camera (2대)  | Single-Camera (1대)")
    print("---------------------|-------------------|-------------------")
    print(f" B1->E1 | Δt (mm)    | {avg_dt_b1e1_bi:17.4f} | {avg_dt_b1e1_dual:17.4f}")
    print(f"        | ΔR (deg)   | {avg_dr_b1e1_bi:17.4f} | {avg_dr_b1e1_dual:17.4f}")
    print("---------------------|-------------------|-------------------")
    print(f" B2->E2 | Δt (mm)    | {avg_dt_b2e2_bi:17.4f} | {avg_dt_b2e2_dual:17.4f}")
    print(f"        | ΔR (deg)   | {avg_dr_b2e2_bi:17.4f} | {avg_dr_b2e2_dual:17.4f}")
    print("="*70)

    # #######################################################
    # ## 실험 2. 다른 종류의 카메라를 가정했을 경우의 최적화 성능 비교 ##
    # #######################################################

    # # Ground-Truth Intrinsic
    # try:
    #     from sim.dual_datagen import TABLE_GT_INTR, TABLE_GT_INTR_CAM2
    # except ImportError:
    #     print("\n[오류] dual_datagen.py 파일을 찾을 수 없거나, 파일 내에")
    #     print("      TABLE_GT_INTR 또는 TABLE_GT_INTR_CAM2 변수가 없습니다.")

    # # Init Intrinsic
    # cam1_intrinsics_datasheet = DivisionIntrinsics(width=1280, height=1024, c=8.4303, kappa=0.99992e-3, sx=5.20997e-3, sy=5.20e-3, cx=659.99, cy=481.96)
    # cam2_intrinsics_datasheet = DivisionIntrinsics(width=1280, height=1024, c=8.0, kappa=1.0e-8, sx=5.20e-3, sy=5.20e-3, cx=640.0, cy=521.0)

    # cam1_intrinsics_init_dict = {
    #     'c': cam1_intrinsics_datasheet.c, 'kappa': cam1_intrinsics_datasheet.kappa, 'sx': cam1_intrinsics_datasheet.sx, 'sy': cam1_intrinsics_datasheet.sy,
    #     'cx': cam1_intrinsics_datasheet.cx, 'cy': cam1_intrinsics_datasheet.cy,
    #     'width': cam1_intrinsics_datasheet.width, 'height': cam1_intrinsics_datasheet.height,
    # }

    # cam2_intrinsics_init_dict = {
    #     'c': cam2_intrinsics_datasheet.c, 'kappa': cam2_intrinsics_datasheet.kappa, 'sx': cam2_intrinsics_datasheet.sx, 'sy': cam2_intrinsics_datasheet.sy,
    #     'cx': cam2_intrinsics_datasheet.cx, 'cy': cam2_intrinsics_datasheet.cy,
    #     'width': cam2_intrinsics_datasheet.width, 'height': cam2_intrinsics_datasheet.height,
    # }

    # print("="*70)
    # print("🚀 [실험 2] Different Bi-Camera (Ground truth intrinsic이 다른 카메라 2대) 최적화 시작")
    # print("="*70)

    # # --- Bi-Camera (카메라 2대) 최적화 호출 ---
    # result = run_optimization_with_vce_dual_bicamera(
    #     model_type='division',
    #     X1_EC_init=X1_EC_init,
    #     T_B1B2_init=T_B1B2_init,
    #     E2_C2_init=E2_C2_init,
    #     T_E1B1_list_init=T_E1B1_list_init,
    #     T_E2B2_list_init=T_B2E2_list_init,
    #     T_C2B_list_init=T_C2B_list_init,
    #     T_C1B_list_init=T_C1B_list_init,
    #     obj_pts_list=obj_pts_list,
    #     img1_pts_list=img1_pts_list,
    #     img2_pts_list=img2_pts_list,
    #     T_E1B1_list_obs=T_E1B1_list_init,
    #     T_E2B2_list_obs=T_B2E2_list_init,
    #     intr1_init=cam1_intrinsics_init_dict,
    #     intr2_init=cam2_intrinsics_init_dict,
    #     sigma_image_px=0.1,
    #     sigma_angle_deg=0.1,
    #     sigma_trans_mm=1.0,
    #     max_vce_iter=10,
    #     max_param_iter=15,
    #     term_thresh=1e-6,
    #     estimate_x1ec=True,
    #     estimate_b1b2=True,
    #     estimate_e2c2=True,
    #     estimate_e1b1=True,
    #     estimate_b2e2=True,
    #     estimate_c2b=False,
    #     estimate_c1b=False,
    #     estimate_intr1=True,
    #     estimate_intr2=True,
    #     include_sy1=False,
    #     include_sy2=False,
    #     is_scara_x1=False
    # )

    # (X1_EC_bi, T_B1B2_bi, E2_C2_bi,
    # T_E1B1_list_bi, T_B2E2_list_bi, _, _,
    # intr1_final_bi, intr2_final_bi) = result

    # def _compare_intrinsics(title: str, gt_intr_dataclass, final_intr_dict: dict):
    #     """Intrinsic dataclass와 dict를 비교하여 표 형식으로 출력합니다."""
    #     # dataclass를 비교하기 쉬운 dict 형태로 변환
    #     gt_intr_dict = asdict(gt_intr_dataclass)
        
    #     print("\n" + "="*50)
    #     print(f"📊 {title}")
    #     print("="*50)
    #     print(f"{'Parameter':<10} | {'Ground Truth':>15} | {'Final Result':>15} | {'Difference':>15}")
    #     print("-"*68)
        
    #     params_to_check = ['c', 'kappa', 'sx', 'sy', 'cx', 'cy']
    #     for key in params_to_check:
    #         gt_val = gt_intr_dict.get(key, float('nan'))
    #         final_val = final_intr_dict.get(key, float('nan'))
    #         diff = final_val - gt_val
    #         print(f"{key:<10} | {gt_val:15.6e} | {final_val:15.6e} | {diff:15.6e}")
    #     print("="*50)

    # # --- 카메라 1 Intrinsic 비교 ---
    # _compare_intrinsics(
    #     "Camera 1 Intrinsic 비교",
    #     TABLE_GT_INTR,
    #     intr1_final_bi
    # )

    # # --- 카메라 2 Intrinsic 비교 ---
    # _compare_intrinsics(
    #     "Camera 2 Intrinsic 비교",
    #     TABLE_GT_INTR_CAM2,
    #     intr2_final_bi
    # )

    #######################################################
    ########### 실험 3. two AX=ZB vs AXB=YCZD ##############
    #######################################################

    result = run_optimization_with_vce_dual_bicamera(
        model_type='division',
        X1_EC_init=X1_EC_init,
        T_B1B2_init=T_B1B2_init,
        E2_C2_init=E2_C2_init,
        T_E1B1_list_init=T_E1B1_list_init,
        T_E2B2_list_init=T_E2B2_list_init,
        T_C2B_list_init=T_C2B_list_init,
        T_C1B_list_init=T_C1B_list_init,
        obj_pts_list=obj_pts_list,
        img1_pts_list=img1_pts_list,
        img2_pts_list=img2_pts_list,
        T_E1B1_list_obs=T_E1B1_list_init,
        T_E2B2_list_obs=T_E2B2_list_init,
        intr1_init=intrinsics_init_dict,
        intr2_init=intrinsics_init_dict,
        sigma_image_px=0.1,
        sigma_angle_deg=0.1,
        sigma_trans_mm=1.0,
        max_vce_iter=15,
        max_param_iter=15,
        term_thresh=1e-6,
        estimate_x1ec=True,
        estimate_b1b2=True,
        estimate_e2c2=True,
        estimate_e1b1=True,
        estimate_b2e2=True,
        estimate_c2b=False,
        estimate_c1b=False,
        estimate_intr1=False,
        estimate_intr2=False,
        include_sy1=False,
        include_sy2=False,
        is_scara_x1=False
    )

    (X1_EC_bi, T_B1B2_bi, E2_C2_bi,
    T_E1B1_list_bi, T_B2E2_list_bi, _, _,
    intr1_final_bi, intr2_final_bi) = result

    # --- 최종 결과 비교 ---
    print("\n\n" + "="*70)
    print("📊 최종 결과 비교: Two-Step vs. Lie vs. UAHC")
    print("="*70)

    print("\n--- [접근법 1] ---")
    evaluate_and_print("Initial Guess (Two-step initialization)",
                       X_gt, Y_gt, Z_gt,
                       inv4(X_est), inv4(Y_est), inv4(Z_est))

    print("\n--- [접근법 2] ---")
    evaluate_and_print("Optimized Result (Lie Based)",
                           X_gt, Y_gt, Z_gt,
                           inv4(X_lie), inv4(Y_lie), inv4(Z_lie))

    print("\n--- [접근법 3] ---")
    evaluate_and_print("Optimized Result (Bi-camera VCE)",
                    X_gt, Y_gt, Z_gt,
                    X1_EC_bi, inv4(T_B1B2_bi), inv4(E2_C2_bi))

    return





    # --- [실험 2] 새로운 '공유 T' 결합형 모델 최적화 (Shared Target V2) ---
    print("\n" + "="*70)
    print("🚀 [실험 2] '공유 T' 결합형 모델 최적화 (Shared Target V2) 시작")
    print("="*70)

    # 'v2' 함수에 맞는 입력 파라미터 준비
    # T_B1_Board 초기값 유추: T_B1_Board = T_B1_E1 @ T_E1_C1 @ T_C1_Board
    T_B1_Board_init = inv4(data['T_B1E1_obs'][0]) @ inv4(X1_EC_init) @ T_C1B_list_init[0]
    
    result_v2 = run_optimization_with_vce_shared_target_v2(
        model_type='division',
        # 새로운 체인 방향에 맞게 입력 파라미터 변환
        T_B1_Board_init=T_B1_Board_init,         # Board -> Base1
        T_C1E1_init=X1_EC_init,                  # Cam1  -> EE1
        T_B2B1_init=inv4(T_B1B2_init),           # Base1 -> Base2
        T_C2E2_init=inv4(E2_C2_init),            # EE2   -> Cam2
        T_E1B1_list_init=data['T_B1E1_obs'],    # EE1   -> Base1
        T_B2E2_list_init=data['T_B2E2_obs'],    # EE2   -> Base2
        # ---
        obj_pts_list=obj_pts_list,
        img1_pts_list=img1_pts_list,
        img2_pts_list=img2_pts_list,
        T_E1B1_list_obs=data['T_B1E1_obs'],
        T_B2E2_list_obs=data["T_B2E2_obs"],
        intr1_init=intrinsics_init_dict,
        intr2_init=intrinsics_init_dict,
        max_vce_iter=15,
        max_param_iter=15,
        # 추정 플래그 이름 변경에 주의
        estimate_b1board=True,
        estimate_c1e1=True,
        estimate_b2b1=True,
        estimate_c2e2=True,
        estimate_e1b1=True,
        estimate_b2e2=True,
        estimate_intr1=False,
        estimate_intr2=False,
    )

    (T_B1_Board_v2, T_C1E1_v2, T_B2B1_v2, T_C2E2_v2, _, _, intr1_final_v2, intr2_final_v2) = result_v2


    # --- [결과 비교] ---
    print("\n\n" + "="*70)
    print("📊 최종 결과 비교: 양방향 체인 vs. 공유 T 결합형")
    print("="*70)

    # --- 전역 파라미터 GT 대비 오차 비교 ---
    print("\n--- [전역 파라미터 오차 비교 (vs GT)] ---")
    print("X: C1->E1,  Y: B2->B1,  Z: C2->E2")

    # Bi-camera 결과 정리
    X_res_bi = X1_EC_bi
    Y_res_bi = inv4(T_B1B2_bi)
    Z_res_bi = inv4(E2_C2_bi)
    evaluate_and_print("Bi-camera (양방향 체인)", X_gt, Y_gt, Z_gt, X_res_bi, Y_res_bi, Z_res_bi)

    # Shared Target V2 결과 정리 (결과 방향이 GT와 다르므로 역변환 필요)
    X_res_v2 = T_C1E1_v2
    Y_res_v2 = T_B2B1_v2
    Z_res_v2 = T_C2E2_v2
    evaluate_and_print("Shared Target V2 (공유 T)", X_gt, Y_gt, Z_gt, X_res_v2, Y_res_v2, Z_res_v2)

    # --- 카메라 내부 파라미터 비교 ---
    print("\n--- [카메라 내부 파라미터 비교] ---")
    # print(f"GT Intrinsics (Cam1)   : {ground_truth['intrinsics1']}") # GT가 있다면
    # print(f"GT Intrinsics (Cam2)   : {ground_truth['intrinsics2']}")
    print("\n[Bi-camera (양방향 체인)]")
    print(f"  - Cam1 Final: {intr1_final_bi}")
    print(f"  - Cam2 Final: {intr2_final_bi}")
    print("\n[Shared Target V2 (공유 T)]")
    print(f"  - Cam1 Final: {intr1_final_v2}")
    print(f"  - Cam2 Final: {intr2_final_v2}")
    print("="*70)

    # --- [3-A] 로봇 1 / 카메라 1에 대한 AX=Z'B 최적화 ---
    print("\n--- [3-A] Optimizing Robot 1 system (AX=Z'B)... ---")

    # T_base_board 초기값 유추 (첫번째 관측 포즈 기준)
    # T_B1_Board = T_B1_E1 * T_E1_C1 * T_C1_Board
    T_B1_Board_init_R1 = inv4(data['T_B1E1_obs'][0]) @ inv4(X1_EC_init) @ T_C1B_list_init[0]
    
    # run_optimization_with_vce_unified 함수 호출
    # 이 함수는 X = T_EndEffector_Camera, Z' = T_Base_Board를 반환
    (X_EC_final_R1, T_B1_Board_final, _, _) = run_optimization_with_vce_unified(
        model_type='division',
        T_ee_cam_init=X1_EC_init,
        T_base_board_init=T_B1_Board_init_R1,
        T_be_list_init=data['T_B1E1_obs'],
        img_pts_list=img1_pts_list,
        obj_pts_list=obj_pts_list,
        T_be_list_obs=data['T_B1E1_obs'],
        intrinsics_init=intrinsics_init_dict,
        sigma_image_px=0.1,
        sigma_angle_deg=0.1,
        sigma_trans_mm=1.0,
        max_vce_iter=5,
        max_param_iter=15,
        term_thresh=1e-6,
        is_target_based=True,
        estimate_ec=True,
        estimate_wb=True,
        estimate_be=True,
        estimate_intrinsics=False,
        is_scara=False
    )

    # 최종 X (C1->E1)
    X_decoupled = inv4(X_EC_final_R1)
    print("\n[로봇 1 최적화 결과]")
    _print_T("X_decoupled (C1->E1)", X_decoupled)
    _print_T("T_B1_Board", T_B1_Board_final)


    # --- [3-B] 로봇 2 / 카메라 2에 대한 A'X'=Z''B' 최적화 ---
    print("\n--- [3-B] Optimizing Robot 2 system (A'X'=Z''B')... ---")

    # T_base_board 초기값 유추 (첫번째 관측 포즈 기준)
    # T_B2_Board = T_B2_E2 * T_E2_C2 * T_C2_Board
    T_B2_Board_init_R2 = data['C_init'][0] @ E2_C2_init @ T_C2B_list_init[0]

    # run_optimization_with_vce_unified 함수 호출
    (X_EC_final_R2, T_B2_Board_final, _, _) = run_optimization_with_vce_unified(
        model_type='division',
        T_ee_cam_init=inv4(E2_C2_init),
        T_base_board_init=T_B2_Board_init_R2,
        T_be_list_init=data['T_B2E2_obs'],
        img_pts_list=img2_pts_list,
        obj_pts_list=obj_pts_list,
        T_be_list_obs=data['T_B2E2_obs'],
        intrinsics_init=intrinsics_init_dict,
        sigma_image_px=0.1,
        sigma_angle_deg=0.1,
        sigma_trans_mm=1.0,
        max_vce_iter=5,
        max_param_iter=15,
        term_thresh=1e-6,
        is_target_based=True,
        estimate_ec=True,
        estimate_wb=True,
        estimate_be=True,
        estimate_intrinsics=False,
        is_scara=False
    )

    # 최종 Z (C2->E2)
    # 함수 출력 X_EC_final_R2는 T_E2_C2에 해당. Z는 C2->E2 이므로 역변환.
    Z_decoupled = inv4(X_EC_final_R2)
    print("\n[로봇 2 최적화 결과]")
    _print_T("Z_decoupled (C2->E2)", Z_decoupled)
    _print_T("T_B2_Board", T_B2_Board_final)


    # --- [3-C] Y 파라미터 유도 ---
    print("\n--- [3-C] Deriving Y parameter (B1->B2)... ---")
    # T_B1_B2 = T_B1_Board * inv(T_B2_Board)
    T_B1_B2_decoupled = T_B1_Board_final @ inv4(T_B2_Board_final)

    # Y는 B2->B1 이므로 위 결과의 역변환
    Y_decoupled = inv4(T_B1_B2_decoupled)
    print("\n[Y 유도 결과]")
    _print_T("Y_decoupled (B2->B1)", Y_decoupled)


    # --- [3-D] 최종 결과 비교 ---
    print("\n\n" + "="*70)
    print("📊 최종 결과 비교: 통합 최적화 vs. 분리 후 조합")
    print("="*70)

    print("\n--- [접근법 1: 통합 최적화 (Simultaneous)] ---")
    evaluate_and_print("Simultaneous (Bi-camera VCE)",
                    X_gt, Y_gt, Z_gt,
                    X1_EC_bi, inv4(T_B1B2_bi), inv4(E2_C2_bi))

    print("\n--- [접근법 2: 분리 후 조합 (Decoupled)] ---")
    evaluate_and_print("Decoupled (AX=Z'B x2)",
                    X_gt, Y_gt, Z_gt,
                    inv4(X_decoupled), Y_decoupled, inv4(Z_decoupled))
    
    print("\n--- [접근법 3: 공동 체인을 통한 최적화 (Semi-coupled)]")
    evaluate_and_print("Shared Target", 
                    X_gt, Y_gt, Z_gt, 
                    X_res_v2, Y_res_v2, Z_res_v2)

    # #######################################################
    # #######################################################
    # #######################################################
    # #######################################################

    # result = run_optimization_with_vce_dual_bicamera(
    #     model_type='division',
    #     X1_EC_init=X1_EC_init,
    #     T_B1B2_init=T_B1B2_init,
    #     E2_C2_init=E2_C2_init,
    #     T_E1B1_list_init=T_E1B1_list_init,
    #     T_E2B2_list_init=T_B2E2_list_init,
    #     T_C2B_list_init=T_C2B_list_init,
    #     T_C1B_list_init=T_C1B_list_init,
    #     obj_pts_list=obj_pts_list,
    #     img1_pts_list=img1_pts_list,
    #     img2_pts_list=img2_pts_list,
    #     T_E1B1_list_obs=T_E1B1_list_init,
    #     T_E2B2_list_obs=T_B2E2_list_init,
    #     intr1_init=intrinsics_init_dict,
    #     intr2_init=intrinsics_init_dict,
    #     sigma_image_px=0.1,
    #     sigma_angle_deg=0.1,
    #     sigma_trans_mm=1.0,
    #     max_vce_iter=10,
    #     max_param_iter=15,
    #     term_thresh=1e-6,
    #     estimate_x1ec=True,
    #     estimate_b1b2=True,
    #     estimate_e2c2=True,
    #     estimate_e1b1=True,
    #     estimate_b2e2=True,
    #     estimate_c2b=False,
    #     estimate_c1b=False,
    #     estimate_intr1=True,
    #     estimate_intr2=True,
    #     include_sy1=False,
    #     include_sy2=False,
    #     is_scara_x1=False
    # )

    # (X1_EC, T_B1B2, E2_C2,
    # T_E1B1_list, T_B2E2_list, T_C2B_list, T_C1B_list,
    # intr1_final, intr2_final) = result

    # print("\n[최적화 결과 요약]")
    # _print_T("X1_EC (C1->E1)", X1_EC)
    # _print_T("T_B1B2 (B1->B2)", T_B1B2)
    # _print_T("E2_C2 (E2->C2)", E2_C2)
    # print("intr_gt:", intrinsics_gt_dict)
    # print("intr1_final:", intr1_final)
    # print("intr2_final:", intr2_final)

    # # -----------------------------
    # # 전역 파라미터: GT 대비 오차 (너의 evaluate_and_print 사용)
    # #   - GT:   X_gt(C1->E1), Y_gt(B2->B1), Z_gt(C2->E2)
    # #   - 추정: X1_EC(C1->E1), inv(T_B1B2)(B2->B1), inv(E2_C2)(C2->E2)
    # # -----------------------------
    # print("\n[전역 파라미터: GT 대비]")
    # evaluate_and_print("Final (Bi-camera VCE)",
    #                 X_gt, Y_gt, Z_gt,
    #                 X1_EC, inv4(T_B1B2), inv4(E2_C2))

    # # -----------------------------
    # # per-pose: 관측 vs GT, 최종 vs GT
    # #   - 관측: data['T_B1E1_obs'][i]  (네 파이프라인 기준: B1->E1로 맞춰져 있다고 가정)
    # #   - 최종: inv(T_E1B1_list[i])    (E1->B1의 역 = B1->E1)
    # #   - 비교는 네가 만든 se3_delta_stats 사용 (인자 순서 유지)
    # # -----------------------------
    # num_poses = len(data["T_B1E1_gt"])

    # sum_dt_obs_b1e1, sum_dr_obs_b1e1 = 0.0, 0.0
    # sum_dt_fin_b1e1, sum_dr_fin_b1e1 = 0.0, 0.0

    # sum_dt_obs_b2e2, sum_dr_obs_b2e2 = 0.0, 0.0
    # sum_dt_fin_b2e2, sum_dr_fin_b2e2 = 0.0, 0.0

    # print(f"\n총 {num_poses}개의 자세에 대해 오차를 계산합니다.")
    # print("=" * 50)
    # print("  - Frame |  B1->E1  Obs (t,R) |  B1->E1  Final (t,R)  ||  B2->E2  Obs (t,R) |  B2->E2  Final (t,R)")
    # print("=" * 50)

    # for i in range(num_poses):
    #     # --- B1->E1 ---
    #     T_gt_B1E1  = data["T_B1E1_gt"][i]     # GT  (B1->E1)
    #     T_obs_B1E1 = data["T_B1E1_obs"][i]    # Obs (B1->E1)
    #     # 최적화 결과는 E1->B1 이므로, 비교를 위해 B1->E1로 역변환
    #     T_fin_B1E1 = T_E1B1_list[i]     # Final (B1->E1)

    #     dt_obs_b1e1,   dr_obs_b1e1,   _ = se3_delta_stats(T_obs_B1E1,  T_gt_B1E1)
    #     dt_final_b1e1, dr_final_b1e1, _ = se3_delta_stats(T_fin_B1E1,  T_gt_B1E1)

    #     sum_dt_obs_b1e1   += dt_obs_b1e1
    #     sum_dr_obs_b1e1   += dr_obs_b1e1
    #     sum_dt_fin_b1e1   += dt_final_b1e1
    #     sum_dr_fin_b1e1   += dr_final_b1e1

    #     # --- B2->E2 ---
    #     T_gt_B2E2  = data["T_B2E2_gt"][i]     # GT  (B2->E2)
    #     T_obs_B2E2 = inv4(T_B2E2_list_init[i])      # Obs (B2->E2)  ← 너가 main에서 data['C_init']로 넣은 값
    #     T_fin_B2E2 = inv4(T_B2E2_list[i])           # Final (B2->E2)

    #     dt_obs_b2e2,   dr_obs_b2e2,   _ = se3_delta_stats(T_obs_B2E2,  T_gt_B2E2)
    #     dt_final_b2e2, dr_final_b2e2, _ = se3_delta_stats(T_fin_B2E2,  T_gt_B2E2)

    #     sum_dt_obs_b2e2   += dt_obs_b2e2
    #     sum_dr_obs_b2e2   += dr_obs_b2e2
    #     sum_dt_fin_b2e2   += dt_final_b2e2
    #     sum_dr_fin_b2e2   += dr_final_b2e2

    #     if i < 5:
    #         print(f"  - {i:05d} | "
    #             f"{dt_obs_b1e1:7.3f}mm,{dr_obs_b1e1:6.3f}° | {dt_final_b1e1:7.3f}mm,{dr_final_b1e1:6.3f}°  || "
    #             f"{dt_obs_b2e2:7.3f}mm,{dr_obs_b2e2:6.3f}° | {dt_final_b2e2:7.3f}mm,{dr_final_b2e2:6.3f}°")

    # # --- 평균 ---
    # avg_dt_obs_b1e1 = sum_dt_obs_b1e1 / num_poses
    # avg_dr_obs_b1e1 = sum_dr_obs_b1e1 / num_poses
    # avg_dt_fin_b1e1 = sum_dt_fin_b1e1 / num_poses
    # avg_dr_fin_b1e1 = sum_dr_fin_b1e1 / num_poses

    # avg_dt_obs_b2e2 = sum_dt_obs_b2e2 / num_poses
    # avg_dr_obs_b2e2 = sum_dr_obs_b2e2 / num_poses
    # avg_dt_fin_b2e2 = sum_dt_fin_b2e2 / num_poses
    # avg_dr_fin_b2e2 = sum_dr_fin_b2e2 / num_poses

    # print("=" * 50)
    # print("\n--- 최종 평균 오차 요약 ---")
    # print("[B1->E1] 관측 vs GT:")
    # print(f"  - Δt: {avg_dt_obs_b1e1:.4f} mm, ΔR: {avg_dr_obs_b1e1:.4f} deg")
    # print("[B1->E1] 최종  vs GT:")
    # print(f"  - Δt: {avg_dt_fin_b1e1:.4f} mm, ΔR: {avg_dr_fin_b1e1:.4f} deg")
    # print(f"  개선량: Δt {avg_dt_obs_b1e1 - avg_dt_fin_b1e1:+.4f} mm | ΔR {avg_dr_obs_b1e1 - avg_dr_fin_b1e1:+.4f} deg")

    # print("\n[B2->E2] 관측 vs GT:")
    # print(f"  - Δt: {avg_dt_obs_b2e2:.4f} mm, ΔR: {avg_dr_obs_b2e2:.4f} deg")
    # print("[B2->E2] 최종  vs GT:")
    # print(f"  - Δt: {avg_dt_fin_b2e2:.4f} mm, ΔR: {avg_dr_fin_b2e2:.4f} deg")
    # print(f"  개선량: Δt {avg_dt_obs_b2e2 - avg_dt_fin_b2e2:+.4f} mm | ΔR {avg_dr_obs_b2e2 - avg_dr_fin_b2e2:+.4f} deg")
    # print("-" * 25)

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
