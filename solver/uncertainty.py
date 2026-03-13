# solver/vector_builder.py
# -*- coding: utf-8 -*-
# "Uncertainty-Aware Hand-Eye Calibration" 논문의 Photogrammetric 모델을 따름.
# 관측 벡터(l)는 2D 이미지 픽셀 좌표와 6D 로봇 자세로 구성.

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from utils.jacobian import (calculate_analytical_jacobian_division_model, 
                            calculate_analytical_jacobian_polynomial_model,
                            calculate_analytical_jacobian_division_model_dual,
                            calculate_analytical_jacobian_division_model_dual_bicamera,
                            calculate_analytical_jacobian_shared_target_v2,
                            calculate_analytical_jacobian_division_model_axbycz)
from scipy.linalg import cho_factor, cho_solve
from utils.projection import DivisionIntrinsics, PolyIntrinsics, DivisionProjector, PolynomialProjector

def safe_mat_to_vec6d(T: np.ndarray, tol_orth=1e-3) -> np.ndarray:
    Rm = np.asarray(T[:3, :3], dtype=float)
    t  = np.asarray(T[:3, 3], dtype=float)

    # 유한성 먼저
    if not (np.isfinite(Rm).all() and np.isfinite(t).all()):
        raise ValueError("non-finite transform in mat_to_vec6d")

    # 필요할 때만 정규직교화
    orth_err = np.linalg.norm(Rm.T @ Rm - np.eye(3), ord='fro')
    det = np.linalg.det(Rm)
    if orth_err > tol_orth or det < 0.0:
        U, S, Vt = np.linalg.svd(Rm)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
    else:
        R_ortho = Rm

    # (중요) 자코비안과 동일한 규약 유지: intrinsic 'xyz'
    euler = Rotation.from_matrix(R_ortho).as_euler('XYZ', degrees=False)
    return np.array([euler[0], euler[1], euler[2], t[0], t[1], t[2]], dtype=float)

def mat_to_vec6d(T):
    """
    4x4 변환 행렬을 6D 벡터(Euler XYZ extrinsic, 이동)로 변환합니다.
    """
    try:
        euler_angles = Rotation.from_matrix(T[:3, :3]).as_euler('XYZ', degrees=False)
    except ValueError:
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        euler_angles = Rotation.from_quat(q).as_euler('XYZ', degrees=False)
    
    t = T[:3, 3]
    return np.concatenate((euler_angles, t))

def vec6d_to_mat(vec):
    """6D 벡터(Euler XYZ extrinsic, 이동)를 4x4 변환 행렬로 변환합니다."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('XYZ', vec[:3], degrees=False).as_matrix()
    T[:3, 3] = vec[3:]
    return T

def verify_jacobian(
    model_type: str,
    X_EC: np.ndarray,
    X_WB: np.ndarray,
    T_BE_list: list,
    obj_pts_list: list,
    intrinsics: dict,
    layout: dict,
    build_x_current_func,
    analytical_jacobian_func,
    project_func,
    epsilon: float = 1e-7,
    **kwargs
):
    """
    해석적 자코비안을 수치적 자코비안(중앙 차분법)과 비교하여 검증합니다.

    Args:
        model_type (str): 'division' 또는 'polynomial'
        X_EC, X_WB, T_BE_list, obj_pts_list, intrinsics: 현재 파라미터 값
        layout (dict): 파라미터 벡터의 레이아웃
        build_x_current_func: 파라미터 구조체로부터 1D 벡터 x_k를 생성하는 함수
        analytical_jacobian_func: 검증할 해석적 자코비안 계산 함수
        project_func: 3D 포인트를 2D 이미지로 투영하는 함수
        epsilon (float): 미소 변화량
        **kwargs: 자코비안 및 프로젝션 함수에 필요한 추가 인자들
    """
    print("\n" + "="*50)
    print(" Jacobian Verification Start")
    print("="*50)

    # 1. 현재 파라미터에서 해석적 자코비안 계산
    A_analytical = analytical_jacobian_func(
        T_ee_cam=X_EC, T_base_board=X_WB, T_be_list=T_BE_list, obj_pts_list=obj_pts_list,
        **intrinsics, **kwargs
    )

    # 2. 수치적 자코비안 계산 (Finite Difference)
    x_k = build_x_current_func(layout)
    num_params = len(x_k)
    num_residuals = A_analytical.shape[0]
    A_numerical = np.zeros_like(A_analytical)

    print(f"Verifying Jacobian with {num_params} parameters and {num_residuals} residuals...")

    for i in range(num_params):
        # 파라미터 i에 대해 양/음 방향으로 미소 변화(perturbation)를 줌
        x_plus = x_k.copy()
        x_plus[i] += epsilon
        x_minus = x_k.copy()
        x_minus[i] -= epsilon

        # 미소 변화에 따른 잔차(residual) 계산
        res_plus = project_func(x_plus, layout, obj_pts_list)
        res_minus = project_func(x_minus, layout, obj_pts_list)
        
        # 중앙 차분법(central difference)으로 i번째 열(column) 계산
        jacobian_col = (res_plus - res_minus) / (2 * epsilon)
        A_numerical[:, i] = jacobian_col

    # 3. 두 자코비안 비교
    diff = np.abs(A_analytical - A_numerical)
    relative_diff = np.abs(diff / (A_numerical + 1e-9)) # 0으로 나누는 것 방지

    max_abs_diff = np.max(diff)
    mean_abs_diff = np.mean(diff)
    max_rel_diff = np.max(relative_diff)
    
    # 가장 큰 차이가 발생한 위치 찾기
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)

    print("\n--- Jacobian Verification Results ---")
    print(f"Max Absolute Difference : {max_abs_diff:.6e} at (row={max_diff_idx[0]}, col={max_diff_idx[1]})")
    print(f"Mean Absolute Difference: {mean_abs_diff:.6e}")
    print(f"Max Relative Difference : {max_rel_diff:.6e}")
    
    # 허용 오차 (보통 1e-5 ~ 1e-7 사이면 양호)
    tolerance = 1e-5
    if max_abs_diff < tolerance:
        print(f"SUCCESS: Jacobian seems correct (Max difference < {tolerance}).")
    else:
        print(f"WARNING: Jacobian might be incorrect (Max difference >= {tolerance}).")
        print("         Please check the implementation for the parameter corresponding to the column with max difference.")

    print("="*50 + "\n")
    return A_analytical, A_numerical, diff

# ---------- VCE solver ------------

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# --- 기존 유틸리티 및 클래스 정의가 여기에 있다고 가정 ---
# from utils.projection import DivisionIntrinsics, PolyIntrinsics, DivisionProjector, PolynomialProjector
# from utils.se3 import mat_to_vec6d, vec6d_to_mat, safe_mat_to_vec6d
from utils.jacobian import calculate_analytical_jacobian_division_model, calculate_analytical_jacobian_polynomial_model

# def run_optimization_with_vce_unified(
#     model_type: str,
#     T_ee_cam_init: np.ndarray,
#     T_base_board_init: np.ndarray,
#     T_be_list_init: list,
#     img_pts_list: list,
#     obj_pts_list: list,
#     T_be_list_obs: list,
#     intrinsics_init: dict,
#     sigma_image_px: float,
#     sigma_angle_deg: float,
#     sigma_trans_mm: float,
#     max_vce_iter: int = 5,
#     max_param_iter: int = 15,
#     term_thresh: float = 1e-6,
#     *,
#     is_target_based: bool = True,
#     estimate_ec: bool = True,
#     estimate_wb: bool = True,
#     estimate_be: bool = True,
#     estimate_intrinsics: bool = False,
#     is_scara: bool = False
# ):
#     """
#     VCE 프레임워크 내에서 scipy.optimize.least_squares와 분석적 자코비안을 사용하여 파라미터를 최적화합니다.
#     """
#     if model_type not in ['division', 'polynomial']:
#         raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

#     # --- 상태 변수 및 초기 설정 (기존과 동일) ---
#     nr = len(T_be_list_init)
#     X_EC, X_WB = T_ee_cam_init.copy(), T_base_board_init.copy()
#     T_BE_list = [T.copy() for T in T_be_list_init]
#     tz0 = float(T_ee_cam_init[2, 3])

#     if model_type == 'division':
#         include_sy = intrinsics_init.get('include_sy', False)
#         proj = DivisionProjector(DivisionIntrinsics(
#             c=intrinsics_init['c'], kappa=intrinsics_init['kappa'], sx=intrinsics_init['sx'],
#             sy=intrinsics_init['sy'], cx=intrinsics_init['cx'], cy=intrinsics_init['cy'],
#             include_sy=include_sy))
#     else:
#         proj = PolynomialProjector(PolyIntrinsics(
#             fx=intrinsics_init['fx'], fy=intrinsics_init['fy'], cx=intrinsics_init['cx'],
#             cy=intrinsics_init['cy'], dist_coeffs=np.array(intrinsics_init['dist_coeffs']).reshape(-1)))

#     l_obs_img = np.concatenate([pts.flatten() for pts in img_pts_list])
#     if estimate_be:
#         l_obs_pose_m = np.concatenate([safe_mat_to_vec6d(T) for T in T_be_list_obs])
#         l_obs_pose_m.reshape(-1, 6)[:, 3:] *= 1000.0 # m -> mm
#         l_obs_pose = l_obs_pose_m.flatten()
#         l_obs = np.concatenate([l_obs_img, l_obs_pose])
#     else:
#         l_obs = l_obs_img
#     ni_total = len(l_obs_img)

#     VAR_I_FLOOR, VAR_A_FLOOR, VAR_T_FLOOR = 1e-12, (np.deg2rad(1e-9))**2, (1e-9)**2
#     var_i = max(sigma_image_px**2, VAR_I_FLOOR)
#     var_a = max(sigma_angle_deg**2, VAR_A_FLOOR)
#     var_t = max(sigma_trans_mm**2, VAR_T_FLOOR)

#     # --- 파라미터 벡터 <-> 변수 변환 헬퍼 함수 (기존과 동일) ---
#     def compute_layout():
#         layout, current_col = {}, 0
#         dim_ec = 5 if is_scara else 6
#         if estimate_ec: layout['ec'] = slice(current_col, current_col + dim_ec); current_col += dim_ec
#         if is_target_based and estimate_wb: layout['wb'] = slice(current_col, current_col + 6); current_col += 6
#         if estimate_be: layout['be'] = slice(current_col, current_col + 6 * nr); current_col += 6 * nr
#         if estimate_intrinsics:
#             num_intr = (6 if proj.intr.include_sy else 5) if model_type == 'division' else (4 + len(proj.intr.dist_coeffs.flatten()))
#             layout['intr'] = slice(current_col, current_col + num_intr); current_col += num_intr
#         layout['total'] = current_col
#         return layout

#     def build_param_vector(layout, ec, wb, be_list, projector):
#         vecs = []
#         if estimate_ec: vecs.append(mat_to_vec6d(ec)[:5] if is_scara else mat_to_vec6d(ec))
#         if is_target_based and estimate_wb: vecs.append(mat_to_vec6d(wb))
#         if estimate_be: vecs.append(np.concatenate([mat_to_vec6d(T) for T in be_list]))
#         if estimate_intrinsics:
#             intr = projector.intr
#             if model_type == 'division': iv = [intr.c, intr.kappa, intr.sx] + ([intr.sy, intr.cx, intr.cy] if intr.include_sy else [intr.cx, intr.cy])
#             else: iv = [intr.fx, intr.fy, intr.cx, intr.cy, *intr.dist_coeffs.flatten()]
#             vecs.append(np.array(iv, dtype=float))
#         return np.concatenate(vecs) if vecs else np.array([])

#     def unpack_param_vector(x, layout):
#         ec = X_EC; wb = X_WB; be = T_BE_list; pr = proj
#         if estimate_ec:
#             e6 = np.array([*x[layout['ec']], tz0]) if is_scara else x[layout['ec']]
#             ec = vec6d_to_mat(e6)
#         if is_target_based and estimate_wb: wb = vec6d_to_mat(x[layout['wb']])
#         if estimate_be: be = [vec6d_to_mat(v) for v in x[layout['be']].reshape(nr, 6)]
#         if estimate_intrinsics:
#             iv = x[layout['intr']]
#             if model_type == 'division':
#                 sy_n = iv[3] if proj.intr.include_sy else iv[2]
#                 pr = DivisionProjector(DivisionIntrinsics(c=iv[0], kappa=iv[1], sx=iv[2], sy=sy_n, cx=iv[-2], cy=iv[-1], include_sy=proj.intr.include_sy))
#             else:
#                 pr = PolynomialProjector(PolyIntrinsics(fx=iv[0], fy=iv[1], cx=iv[2], cy=iv[3], dist_coeffs=iv[4:]))
#         return ec, wb, be, pr
    
#     print("="*50); print("Optimization Start (using scipy.optimize.least_squares with Analytic Jacobian)"); print("="*50)

#     for vce_iter in range(max_vce_iter):
#         print(f"\n--- VCE Iteration {vce_iter + 1}/{max_vce_iter} ---")
#         print(f"Variances: σ_img²={var_i:.3e}, σ_ang²={var_a:.3e}, σ_trans²={var_t:.3e}")
        
#         wi, wa, wt = 1.0/var_i, 1.0/var_a, 1.0/var_t
#         Pll_diag = np.full(ni_total, wi)
#         if estimate_be:
#             Pll_diag = np.concatenate([Pll_diag, np.tile([wa]*3 + [wt]*3, nr)])
#         sqrt_P = np.sqrt(Pll_diag)
        
#         layout = compute_layout()
#         x_initial = build_param_vector(layout, X_EC, X_WB, T_BE_list, proj)

#         # --- SciPy를 위한 목적 함수와 자코비안 함수 정의 ---
#         # 이 함수들은 루프 내에서 현재의 가중치(sqrt_P)와 상태(layout)를 참조해야 함
#         def objective_func(x_k):
#             try: ec, wb, be, pr = unpack_param_vector(x_k, layout)
#             except Exception: return np.full_like(l_obs, 1e10)
            
#             f_pix = pr.project_dataset_flat(ec, wb, be, obj_pts_list)
#             f_k = np.concatenate((f_pix, x_k[layout['be']])) if estimate_be else f_pix
#             w = l_obs - f_k
#             return w * sqrt_P if not np.isnan(w).any() else np.full_like(w, 1e10)

#         def jacobian_func(x_k):
#             # 1. 현재 파라미터로 상태 변수들을 다시 계산
#             ec, wb, be, pr = unpack_param_vector(x_k, layout)
            
#             # 2. 분석적 자코비안 계산 (기존 함수 재사용)
#             if model_type == 'division':
#                 intr = pr.intr
#                 A_img = calculate_analytical_jacobian_division_model(
#                     ec, wb, be, obj_pts_list, intr.c, intr.kappa, intr.sx, 
#                     intr.sy, intr.cx, intr.cy, is_target_based=is_target_based, is_scara=is_scara,
#                     estimate_ec=estimate_ec, estimate_wb=estimate_wb,
#                     estimate_intrinsics=estimate_intrinsics, include_sy=intr.include_sy,
#                     mat_to_vec6d=mat_to_vec6d)
#             else: # Polynomial model
#                 intr = pr.intr
#                 A_img = calculate_analytical_jacobian_polynomial_model(
#                     ec, wb, be, obj_pts_list, intr.fx, intr.fy, intr.cx, intr.cy, intr.dist_coeffs,
#                     is_target_based=is_target_based, is_scara=is_scara, estimate_ec=estimate_ec,
#                     estimate_wb=estimate_wb, estimate_be=estimate_be, estimate_intrinsics=estimate_intrinsics,
#                     mat_to_vec6d=mat_to_vec6d)

#             # 3. 전체 자코비안 A 구성
#             if estimate_be:
#                 A = np.zeros((ni_total + 6 * nr, layout['total']))
#                 A[:ni_total, :] = A_img
#                 A[ni_total:, layout.get('be', slice(0,0))] = np.eye(6 * nr)
#             else:
#                 A = A_img

#             # 4. 가중치를 적용한 자코비안 반환 (J_weighted = sqrt(P) * J)
#             # 최소화하는 함수가 f(x)일 때, jac은 df/dx.
#             # 우리 함수는 sqrt(P)*f(x) 이므로, jac은 sqrt(P)*df/dx.
#             return A * sqrt_P[:, np.newaxis]

#         # --- LM 루프를 SciPy least_squares 호출로 대체 ---
#         result = least_squares(
#             fun=objective_func,
#             x0=x_initial,
#             jac=jacobian_func, # ★★★ 분석적 자코비안 전달 ★★★
#             method='trf',
#             loss='huber',
#             xtol=term_thresh, ftol=term_thresh, gtol=term_thresh,
#             max_nfev=max_param_iter,
#             verbose=0
#         )
        
#         if not result.success:
#             print(f"  [SciPy] Optimization failed or did not converge: {result.message}")
        
#         X_EC, X_WB, T_BE_list, proj = unpack_param_vector(result.x, layout)
#         print(f"    - Cost(φ) = {2 * result.cost:.6f}, Status: {result.message}")

#         # --- VCE 업데이트 (분석적 자코비안 재사용) ---
#         print("  Updating variance components...")
#         final_jac = jacobian_func(result.x) / sqrt_P[:, np.newaxis] # 가중치 제거
#         v_hat = objective_func(result.x) / sqrt_P # 가중치 제거

#         N_mat = final_jac.T @ (Pll_diag[:, None] * final_jac)
#         try:
#             Q_xx = np.linalg.pinv(N_mat)
#             diag_Qlhat = np.einsum('ij,jk,ik->i', final_jac, Q_xx, final_jac)
#         except np.linalg.LinAlgError:
#             print("  Warning: Could not compute Q_xx for VCE update. Skipping."); continue
            
#         diag_R = 1.0 - Pll_diag * diag_Qlhat
#         v_i = v_hat[:ni_total]
#         R_i_sum = np.sum(diag_R[:ni_total])
#         sigma2_hat_i = (v_i.T @ (Pll_diag[:ni_total] * v_i) / max(R_i_sum, 1e-9))

#         sigma2_hat_a, sigma2_hat_t = 1.0, 1.0
#         if estimate_be:
#             idx_a = np.concatenate([np.arange(i, i+3) for i in range(ni_total, ni_total + 6*nr, 6)])
#             idx_t = np.concatenate([np.arange(i+3, i+6) for i in range(ni_total, ni_total + 6*nr, 6)])
#             R_a_sum, v_a = np.sum(diag_R[idx_a]), v_hat[idx_a]
#             R_t_sum, v_t = np.sum(diag_R[idx_t]), v_hat[idx_t]
#             sigma2_hat_a = (v_a.T @ (Pll_diag[idx_a] * v_a) / max(R_a_sum, 1e-9))
#             sigma2_hat_t = (v_t.T @ (Pll_diag[idx_t] * v_t) / max(R_t_sum, 1e-9))
#             print(f"  Redundancy sums: R_img={R_i_sum:.2f}, R_ang={R_a_sum:.2f}, R_trans={R_t_sum:.2f}")
#         else:
#             print(f"  Redundancy sums: R_img={R_i_sum:.2f}")

#         var_i = max(var_i * sigma2_hat_i, VAR_I_FLOOR)
#         if estimate_be:
#             var_a = max(var_a * sigma2_hat_a, VAR_A_FLOOR)
#             var_t = max(var_t * sigma2_hat_t, VAR_T_FLOOR)
#         print(f"  σ̂²_img={sigma2_hat_i:.4f}, σ̂²_ang={sigma2_hat_a:.4f}, σ̂²_trans={sigma2_hat_t:.4f}")

#         if np.allclose([sigma2_hat_i, sigma2_hat_a, sigma2_hat_t], 1.0, atol=1e-3):
#             print("  VCE converged."); break

#     # --- 최종 결과 패킹 (기존과 동일) ---
#     final_intrinsics = proj.intr.__dict__
#     print("\nOptimization Finished.")
#     return X_EC, X_WB, T_BE_list, final_intrinsics

def run_optimization_with_vce_unified(
    model_type: str,
    T_ee_cam_init: np.ndarray,
    T_base_board_init: np.ndarray,
    T_be_list_init: list,
    img_pts_list: list,
    obj_pts_list: list,
    T_be_list_obs: list,
    intrinsics_init: dict,
    sigma_image_px: float,
    sigma_angle_deg: float,
    sigma_trans_mm: float,
    max_vce_iter: int = 5,
    max_param_iter: int = 10,
    term_thresh: float = 1e-6,
    *,
    is_target_based: bool = True,
    estimate_ec: bool = True,
    estimate_wb: bool = True,
    estimate_be: bool = True,
    estimate_intrinsics: bool = False,
    is_scara: bool = False
):
    if model_type not in ['division', 'polynomial']:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    # --- state ---
    nr = len(T_be_list_init)
    X_EC, X_WB = T_ee_cam_init.copy(), T_base_board_init.copy()
    T_BE_list = [T.copy() for T in T_be_list_init]
    sy0_fixed = intrinsics_init['sy']

    # --- intrinsics dataclass + projector 생성 ---
    if model_type == 'division':
        include_sy = intrinsics_init.get('include_sy', False)
        div_intr = DivisionIntrinsics(
            c=intrinsics_init['c'],
            kappa=intrinsics_init['kappa'],
            sx=intrinsics_init['sx'],
            sy=intrinsics_init['sy'],
            cx=intrinsics_init['cx'],
            cy=intrinsics_init['cy'],
            include_sy=include_sy
        )
        proj = DivisionProjector(div_intr)
    else:
        poly_intr = PolyIntrinsics(
            fx=intrinsics_init['fx'], fy=intrinsics_init['fy'],
            cx=intrinsics_init['cx'], cy=intrinsics_init['cy'],
            dist_coeffs=np.array(intrinsics_init['dist_coeffs']).reshape(-1)
        )
        proj = PolynomialProjector(poly_intr)

    # --- 관측 벡터 ---
    l_obs_img = np.concatenate([pts.flatten() for pts in img_pts_list])
    if estimate_be:
        l_obs_pose = np.concatenate([safe_mat_to_vec6d(T) for T in T_be_list_obs])
        l_obs = np.concatenate([l_obs_img, l_obs_pose])
    else:
        l_obs = l_obs_img
    ni_total = len(l_obs_img)

    # --- 초기 분산, LM 설정 ---
    VAR_I_FLOOR, VAR_A_FLOOR, VAR_T_FLOOR = 1e-12, (np.deg2rad(1e-9))**2, (1e-9)**2
    var_i = max(sigma_image_px**2, VAR_I_FLOOR)
    var_a = max((np.deg2rad(sigma_angle_deg))**2, VAR_A_FLOOR)
    var_t = max((sigma_trans_mm/1000.0)**2, VAR_T_FLOOR)
    lam, lam_up, lam_down, lam_min, lam_max, max_ls_tries = 1e-5, 5.0, 0.25, 1e-12, 1e+8, 8
    tz0 = float(T_ee_cam_init[2,3])

    def compute_layout():
        layout = {}
        current_col = 0
        dim_ec = 5 if is_scara else 6
        if estimate_ec: layout['ec'] = slice(current_col, current_col + dim_ec); current_col += dim_ec
        if is_target_based and estimate_wb: layout['wb'] = slice(current_col, current_col + 6); current_col += 6
        if estimate_be: layout['be'] = slice(current_col, current_col + 6 * nr); current_col += 6 * nr
        if estimate_intrinsics:
            if model_type == 'division':
                num_intr_params = 6 if proj.intr.include_sy else 5
            else:
                num_intr_params = 4 + len(proj.intr.dist_coeffs.flatten())
            layout['intr'] = slice(current_col, current_col + num_intr_params)
            current_col += num_intr_params
        layout['total'] = current_col
        return layout

    def build_x_current(layout):
        vecs = []
        if estimate_ec:
            e6 = mat_to_vec6d(X_EC)
            vecs.append(e6[:5] if is_scara else e6)
        if is_target_based and estimate_wb:
            vecs.append(mat_to_vec6d(X_WB))
        if estimate_be:
            vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_BE_list]))
        if estimate_intrinsics:
            if model_type == 'division':
                intr = proj.intr
                intr_vec = [intr.c, intr.kappa, intr.sx]
                if intr.include_sy: intr_vec += [intr.sy, intr.cx, intr.cy]
                else:               intr_vec += [intr.cx, intr.cy]
                vecs.append(np.array(intr_vec, dtype=float))
            else:
                intr = proj.intr
                vecs.append(np.array([intr.fx, intr.fy, intr.cx, intr.cy, *intr.dist_coeffs.flatten()], dtype=float))
        return np.concatenate(vecs)

    def _ec_vec_to_mat(ec_vec):
        e6 = np.array([*ec_vec, tz0]) if is_scara else ec_vec
        R = Rotation.from_euler('XYZ', e6[:3], degrees=False).as_matrix()
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = e6[3:]
        return T

    print("="*50); print("Optimization Start"); print("="*50)

    for vce_iter in range(max_vce_iter):
        print(f"\n--- VCE Iteration {vce_iter + 1}/{max_vce_iter} ---")
        print(f"Variances: σ_img²={var_i:.3e}, σ_ang²={var_a:.3e}, σ_trans²={var_t:.3e}")
        wi, wa, wt = 1.0/var_i, 1.0/var_a, 1.0/var_t
        Pll_diag = np.full(ni_total, wi)
        if estimate_be:
            pose_weights = np.tile(np.concatenate([np.full(3, wa), np.full(3, wt)]), nr)
            Pll_diag = np.concatenate([Pll_diag, pose_weights])

        layout = compute_layout()
        phi_new = np.inf
        delta_x_term_thresh=1e-8

        idx_img = slice(0, ni_total)
        idx_ang, idx_trans = None, None
        if estimate_be:
            # T_BE의 6-dof 벡터에서 회전(앞 3개)과 이동(뒤 3개) 인덱스 분리
            idx_ang = np.concatenate([np.arange(ni_total + i, ni_total + i + 3) for i in range(0, 6 * nr, 6)])
            idx_trans = np.concatenate([np.arange(ni_total + i + 3, ni_total + i + 6) for i in range(0, 6 * nr, 6)])

        # << 추가: 비용 분해 계산을 위한 헬퍼 함수 >>
        def _get_cost_breakdown(w, Pll_diag):
            cost_img = np.sum(Pll_diag[idx_img] * (w[idx_img]**2))
            cost_ang = 0.0
            cost_trans = 0.0
            if estimate_be and idx_ang is not None:
                cost_ang = np.sum(Pll_diag[idx_ang] * (w[idx_ang]**2))
                cost_trans = np.sum(Pll_diag[idx_trans] * (w[idx_trans]**2))
            return cost_img, cost_ang, cost_trans
        
        # for param_iter in range(max_param_iter):
        #     # --- [수정] 루프 시작 시 현재 비용(phi_k) 계산 ---
        #     x_k = build_x_current(layout)
        #     f_pix = proj.project_dataset_flat(X_EC, X_WB, T_BE_list, obj_pts_list)
        #     if estimate_be:
        #         f_pose = np.concatenate([mat_to_vec6d(T) for T in T_BE_list])
        #         w = l_obs - np.concatenate((f_pix, f_pose))
        #     else:
        #         w = l_obs - f_pix
            
        #     if np.isnan(w).any():
        #         print(f"  [Iter {param_iter + 1}] Error: NaN in residual. Stopping.")
        #         break

        #     # print(w[:10])
            
        #     cost0_img, cost0_ang, cost0_trans = _get_cost_breakdown(w, Pll_diag)
        #     phi_k = np.sum(Pll_diag * (w**2))

        #     # --- (C) 자코비안은 기존 함수 재사용 (intr는 proj에서 꺼냄) ---
        #     if model_type == 'division':
        #         intr = proj.intr
        #         sy_cur = intr.sy if intr.include_sy else sy0_fixed
        #         A_img = calculate_analytical_jacobian_division_model(
        #             X_EC, X_WB, T_BE_list, obj_pts_list,
        #             intr.c, intr.kappa, intr.sx, sy_cur, intr.cx, intr.cy,
        #             is_target_based=is_target_based, is_scara=is_scara,
        #             estimate_ec=estimate_ec, estimate_wb=estimate_wb, estimate_be=estimate_be,
        #             estimate_intrinsics=estimate_intrinsics, include_sy=intr.include_sy,
        #             mat_to_vec6d=mat_to_vec6d
        #         )
        #     else:
        #         intr = proj.intr
        #         A_img = calculate_analytical_jacobian_polynomial_model(
        #             T_ee_cam=X_EC, T_base_board=X_WB, T_be_list=T_BE_list, obj_pts_list=obj_pts_list,
        #             fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, dist_coeffs=intr.dist_coeffs,
        #             is_target_based=is_target_based, is_scara=is_scara,
        #             estimate_ec=estimate_ec, estimate_wb=estimate_wb, estimate_be=estimate_be,
        #             estimate_intrinsics=estimate_intrinsics,
        #             mat_to_vec6d=mat_to_vec6d
        #         )

        #     # --- (D) 완성된 A 행렬 (기존과 동일) ---
        #     if estimate_be:
        #         A = np.zeros((ni_total + 6 * nr, layout['total']))
        #         A[:ni_total, :] = A_img
        #         A[ni_total:, layout.get('be', slice(0,0))] = np.eye(6 * nr)
        #     else:
        #         A = A_img
                
        #     # --- (E) LM 스텝 계산 ---
        #     N = A.T @ (Pll_diag[:,None]*A)
        #     b = A.T @ (Pll_diag*w)

        #     try:
        #         N_aug = N + lam * np.diag(np.diag(N))
        #         delta = np.linalg.solve(N_aug, b)
        #     except np.linalg.LinAlgError:
        #         print(f"  [Iter {param_iter + 1}] LinAlgError. Increasing lambda and continuing.")
        #         lam = min(lam * lam_up, lam_max)
        #         continue

        #     # --- [수정] 수렴 조건: delta의 크기가 임계값보다 작으면 종료 ---
        #     delta_norm = np.linalg.norm(delta)
        #     if delta_norm < delta_x_term_thresh:
        #         print(f"\n  [LM] Converged by delta_x norm ({delta_norm:.2e}) < {delta_x_term_thresh:.2e}")
        #         break

        #     # --- [수정] 비용 검사 없이 즉시 파라미터 업데이트 ---
        #     x_new = x_k + delta
            
        #     try:
        #         X_EC  = _ec_vec_to_mat(x_new[layout['ec']]) if estimate_ec else X_EC
        #         X_WB  = vec6d_to_mat(x_new[layout['wb']])   if (is_target_based and estimate_wb) else X_WB
        #         T_BE_list = [vec6d_to_mat(v) for v in x_new[layout['be']].reshape(nr,6)] if estimate_be else T_BE_list
        #     except Exception as e:
        #         print(f"  [Iter {param_iter + 1}] Error during parameter update: {e}. Stopping.")
        #         break

        #     # (Intrinsics 업데이트 로직은 estimate_intrinsics=True일 때만 필요하므로 생략, 필요 시 추가)

        #     # --- [수정] 업데이트 후 비용 계산 (Lambda 조절 및 로깅용) ---
        #     f_pix_new = proj.project_dataset_flat(X_EC, X_WB, T_BE_list, obj_pts_list)
        #     if estimate_be:
        #         f_pose_new = x_new[layout['be']]
        #         w_new = l_obs - np.concatenate((f_pix_new, f_pose_new))
        #     else:
        #         w_new = l_obs - f_pix_new

        #     # print(w_new[:10])

        #     # << 추가: 스텝 성공 후 비용 상세 분석 출력 >>
        #     cost_new_img, cost_new_ang, cost_new_trans = _get_cost_breakdown(w_new, Pll_diag)
        #     # print(f"      - Cost Breakdown (Img | Ang | Trans)")
        #     # print(f"        Before: {cost0_img:9.2f} | {cost0_ang:9.2f} | {cost0_trans:9.2f}")
        #     # print(f"        After:  {cost_new_img:9.2f} | {cost_new_ang:9.2f} | {cost_new_trans:9.2f}")
            
        #     phi_new = np.inf if np.isnan(w_new).any() else np.sum(Pll_diag * (w_new**2))

        #     # --- [수정] 비용 변화에 따라 다음 스텝의 lambda만 조절 ---
        #     if phi_new < phi_k:
        #         # 비용 감소: 성공적인 스텝. 다음엔 더 큰 스텝을 시도 (lambda 감소)
        #         # print(f"  [Iter {param_iter+1}] Step Accepted. Cost: {phi_k:.4f} -> {phi_new:.4f}. λ down.")
        #         lam = max(lam * lam_down, lam_min)
        #     else:
        #         # 비용 증가: 실패한 스텝. 다음엔 더 작은 스텝을 시도 (lambda 증가)
        #         # print(f"  [Iter {param_iter+1}] Uphill Step Accepted. Cost: {phi_k:.4f} -> {phi_new:.4f}. λ up.")
        #         lam = min(lam * lam_up, lam_max)
        # else:
        #     # for 루프가 break 없이 정상 종료되었을 때 (최대 반복 도달)
        #     print(f"\n  [LM] Reached max iterations ({max_param_iter}).")

        for param_iter in range(max_param_iter):
            x_k = build_x_current(layout)

            # --- (A) 현재 프로젝터로 예측 ---
            f_pix = proj.project_dataset_flat(X_EC, X_WB, T_BE_list, obj_pts_list)

            # --- (B) 잔차 ---
            if estimate_be:
                f_pose = np.concatenate([mat_to_vec6d(T) for T in T_BE_list])
                w = l_obs - np.concatenate((f_pix, f_pose))
            else:
                w = l_obs - f_pix

            if np.isnan(w).any():
                print(f"  [LM-{param_iter + 1}] Error: NaN detected in residual vector. Stopping.")
                return None, None, None, None

            # --- (C) 자코비안은 기존 함수 재사용 (intr는 proj에서 꺼냄) ---
            if model_type == 'division':
                intr = proj.intr
                sy_cur = intr.sy if intr.include_sy else sy0_fixed
                A_img = calculate_analytical_jacobian_division_model(
                    X_EC, X_WB, T_BE_list, obj_pts_list,
                    intr.c, intr.kappa, intr.sx, sy_cur, intr.cx, intr.cy,
                    is_target_based=is_target_based, is_scara=is_scara,
                    estimate_ec=estimate_ec, estimate_wb=estimate_wb, estimate_be=estimate_be,
                    estimate_intrinsics=estimate_intrinsics, include_sy=intr.include_sy,
                    mat_to_vec6d=mat_to_vec6d
                )
            else:
                intr = proj.intr
                A_img = calculate_analytical_jacobian_polynomial_model(
                    T_ee_cam=X_EC, T_base_board=X_WB, T_be_list=T_BE_list, obj_pts_list=obj_pts_list,
                    fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, dist_coeffs=intr.dist_coeffs,
                    is_target_based=is_target_based, is_scara=is_scara,
                    estimate_ec=estimate_ec, estimate_wb=estimate_wb, estimate_be=estimate_be,
                    estimate_intrinsics=estimate_intrinsics,
                    mat_to_vec6d=mat_to_vec6d
                )

            # --- (D) 완성된 A ---
            if estimate_be:
                A = np.zeros((ni_total + 6 * nr, layout['total']))
                A[:ni_total, :] = A_img
                A[ni_total:, layout.get('be', slice(0,0))] = np.eye(6 * nr)
            else:
                A = A_img

            # --- (E) LM step ---
            phi0 = np.sum(Pll_diag * (w**2))
            cost0_img, cost0_ang, cost0_trans = _get_cost_breakdown(w, Pll_diag)
            N = A.T @ (Pll_diag[:,None]*A)

            # condition_number = np.linalg.cond(N)
            # print(f"Normal Matrix Condition Number: {condition_number}")

            # try:
            #     # SVD(특이값 분해) 수행
            #     U, S, Vt = np.linalg.svd(N)

            #     s_max = S[0]
            #     s_min = S[-1]
            #     cond_svd = s_max / s_min

            #     print("\n" + "="*50)
            #     print("Normal Matrix (N) SVD Analysis")
            #     print("="*50)
            #     print(f"  - Condition Number (from SVD): {cond_svd:.4e}")
            #     print(f"  - Largest Singular Value (σ_max): {s_max:.4e}")
            #     print(f"  - Smallest Singular Value (σ_min): {s_min:.4e}")

            #     # 가장 작은 특이값에 해당하는 특이벡터 (Vt의 마지막 행)
            #     weakest_vector = Vt[-1, :]

            #     print("\n  --- Weakest Component Analysis (Vector for σ_min) ---")
            #     print("  This vector shows which parameters are most ambiguous.")

            #     # 파라미터 레이아웃을 기반으로 어떤 파라미터가 큰 값을 갖는지 분석
            #     # [버그 수정] 'total' 키를 건너뛰도록 수정
            #     layout_items = sorted(layout.items(), key=lambda item: item[1].start if isinstance(item[1], slice) else float('inf'))

            #     for name, param_slice in layout_items:
            #         # 'total' 키는 slice 객체가 아니므로 건너뜀
            #         if not isinstance(param_slice, slice):
            #             continue

            #         # 해당 파라미터 그룹의 벡터 성분들 추출
            #         param_vector_segment = weakest_vector[param_slice]
            #         energy = np.mean(param_vector_segment**2)

            #         print(f"  - Group '{name}':")
            #         print(f"    - Average Energy: {energy:.4e}")

            #         if name == 'be':
            #             be_vectors = param_vector_segment.reshape(-1, 6)
            #             trans_energy = np.mean(be_vectors[:, 3:]**2)
            #             rot_energy = np.mean(be_vectors[:, :3]**2)
            #             print(f"      - Translation Energy: {trans_energy:.4e}")
            #             print(f"      - Rotation Energy:    {rot_energy:.4e}")

            #     print("="*50 + "\n")

            # except Exception as e:
            #     print(f"N 행렬 SVD 분석 중 오류 발생: {e}")
            
            # try:
            #     # 로그 스케일로 변환하여 명암 대비를 높임
            #     # N의 절대값에 작은 값을 더해 log(0) 오류 방지
            #     N_log_abs = np.log10(np.abs(N) + 1e-9)
                
            #     plt.figure(figsize=(10, 8))
            #     plt.imshow(N_log_abs, cmap='viridis', interpolation='nearest')
            #     plt.title('Structure of the Normal Matrix N (log scale)')
            #     plt.xlabel('Parameter Index')
            #     plt.ylabel('Parameter Index')
            #     plt.colorbar(label='log10(|N_ij|)')

            #     # 파라미터 그룹별로 경계선과 라벨 추가
            #     param_boundaries = []
            #     param_labels = []
            #     current_pos = 0

            #     if layout.get('ec'):
            #         param_boundaries.append(layout['ec'].stop)
            #         param_labels.append('EC')
            #     if layout.get('wb'):
            #         param_boundaries.append(layout['wb'].stop)
            #         param_labels.append('WB')
            #     if layout.get('be'):
            #         param_boundaries.append(layout['be'].stop)
            #         param_labels.append('BE')
            #     if layout.get('intr'):
            #         param_boundaries.append(layout['intr'].stop)
            #         param_labels.append('Intrinsics')
                
            #     ticks = []
            #     tick_labels = []
            #     start = 0
            #     for label, end in zip(param_labels, param_boundaries):
            #         # 경계선 그리기
            #         plt.axvline(x=end - 0.5, color='white', linestyle='--', linewidth=0.8)
            #         plt.axhline(y=end - 0.5, color='white', linestyle='--', linewidth=0.8)
                    
            #         # 라벨 위치 계산
            #         ticks.append(start + (end - start) / 2)
            #         tick_labels.append(label)
            #         start = end

            #     plt.xticks(ticks, tick_labels, rotation=45)
            #     plt.yticks(ticks, tick_labels)
                
            #     print("✅ Normal Matrix N 시각화 그래프를 표시합니다. 창을 닫으면 최적화가 계속 진행됩니다.")
            #     plt.show()

            # except Exception as e:
            #     print(f"N 행렬 시각화 중 오류 발생: {e}")

            b = A.T @ (Pll_diag*w)

            accepted_delta_norm = np.inf
            converged_by_delta_x = False

            for ls_try in range(max_ls_tries):
                try:
                    N_aug = N + lam * np.diag(np.diag(N))
                    delta = np.linalg.solve(N_aug, b)
                except np.linalg.LinAlgError:
                    lam = min(lam * lam_up, lam_max); continue
                
                delta_norm = np.linalg.norm(delta)
                if delta_norm < delta_x_term_thresh:
                    converged_by_delta_x = True
                    # 성공적인 업데이트로 간주하기 위해 phi_new를 phi0보다 작게 설정
                    phi_new = phi0 - 1.0 
                    # 파라미터 업데이트 수행
                    x_new = x_k + delta

                x_new = x_k + delta
                if not np.isfinite(x_new).all():
                    lam = min(lam * lam_up, lam_max); continue

                # --- (E-1) 파라미터 갱신 (행렬 & intr & projector) ---
                try:
                    X_EC_new  = _ec_vec_to_mat(x_new[layout['ec']]) if estimate_ec else X_EC
                    X_WB_new  = vec6d_to_mat(x_new[layout['wb']])   if (is_target_based and estimate_wb) else X_WB
                    T_BE_list_new = [vec6d_to_mat(v) for v in x_new[layout['be']].reshape(nr,6)] if estimate_be else T_BE_list
                except Exception:
                    lam = min(lam * lam_up, lam_max); continue

                proj_new = proj  # 기본은 그대로
                if estimate_intrinsics:
                    intr_vec = x_new[layout['intr']]
                    if model_type == 'division':
                        if proj.intr.include_sy:
                            c_n,k_n,sx_n,sy_n,cx_n,cy_n = intr_vec
                            intr_new = DivisionIntrinsics(
                                c=c_n, kappa=k_n, sx=sx_n, sy=sy_n, cx=cx_n, cy=cy_n,
                                include_sy=proj.intr.include_sy
                            )
                        else:
                            c_n,k_n,sx_n,cx_n,cy_n = intr_vec
                            intr_new = DivisionIntrinsics(
                                c=c_n, kappa=k_n, sx=sx_n, sy=sy0_fixed, cx=cx_n, cy=cy_n,
                                include_sy=proj.intr.include_sy
                            )
                        proj_new = DivisionProjector(intr_new)
                    else:
                        fx_n,fy_n,cx_n,cy_n = intr_vec[:4]
                        dist_n = intr_vec[4:]
                        intr_new = PolyIntrinsics(
                            fx=fx_n, fy=fy_n, cx=cx_n, cy=cy_n, dist_coeffs=dist_n
                        )
                        proj_new = PolynomialProjector(intr_new)

                # --- (E-2) 새 프로젝션으로 평가 ---
                f_pix_new = proj_new.project_dataset_flat(X_EC_new, X_WB_new, T_BE_list_new, obj_pts_list)

                if estimate_be:
                    # f_pose_new = np.concatenate([safe_mat_to_vec6d(T) for T in T_BE_list_new])
                    f_pose_new = x_new[layout['be']]
                    w_new = l_obs - np.concatenate((f_pix_new, f_pose_new))
                else:
                    w_new = l_obs - f_pix_new

                phi_new = np.inf if np.isnan(w_new).any() else np.sum(Pll_diag * (w_new**2))

                if phi_new < phi0 or converged_by_delta_x:
                    if converged_by_delta_x:
                        print(f"    - Attempt {ls_try+1}: Converged by delta_x norm ({delta_norm:.2e}) < {delta_x_term_thresh:.2e}")
                    else:
                        print(f"    - Attempt {ls_try+1}: Success! Cost(φ) = {phi_new:.6f}, λ = {lam:.2e}")
                    
                    # << 추가: 스텝 성공 후 비용 상세 분석 출력 >>
                    cost_new_img, cost_new_ang, cost_new_trans = _get_cost_breakdown(w_new, Pll_diag)
                    print(f"      - Cost Breakdown (Img | Ang | Trans)")
                    print(f"        Before: {cost0_img:9.2f} | {cost0_ang:9.2f} | {cost0_trans:9.2f}")
                    print(f"        After:  {cost_new_img:9.2f} | {cost_new_ang:9.2f} | {cost_new_trans:9.2f}")
                    
                    # 수용
                    X_EC, X_WB, T_BE_list = X_EC_new, X_WB_new, T_BE_list_new
                    proj = proj_new
                    lam = max(lam * lam_down, lam_min)
                    
                    # << 추가: 수용된 delta의 크기 저장 >>
                    accepted_delta_norm = delta_norm
                    break
                else:
                    lam = min(lam * lam_up, lam_max)

                # if phi_new < phi0:
                #     print(f"    - Attempt {ls_try+1}: Success! Cost(φ) = {phi_new:.6f}, λ = {lam:.2e}")
                #     # 수용
                #     X_EC, X_WB, T_BE_list = X_EC_new, X_WB_new, T_BE_list_new
                #     proj = proj_new
                #     lam = max(lam * lam_down, lam_min)
                #     break
                # else:
                #     lam = min(lam * lam_up, lam_max)
            else:
                print(f"  [LM-{param_iter + 1}] Failed to improve after {max_ls_tries} attempts.")

            if abs(phi0 - phi_new) < term_thresh:
                print("  [LM] Converged. Terminating parameter search.")
                break

        # --- (F) VCE 업데이트 ---
        print("  Updating variance components using rigorous method...")
        f_pix_final = proj.project_dataset_flat(X_EC, X_WB, T_BE_list, obj_pts_list)
        x_now = build_x_current(layout)
        if estimate_be:
            # f_pose_final = np.concatenate([mat_to_vec6d(T) for T in T_BE_list])
            f_pose_final = x_now[layout['be']]
            v_hat = l_obs - np.concatenate((f_pix_final, f_pose_final))
        else:
            v_hat = l_obs - f_pix_final

        # 자코비안 다시 (프로젝터 intr에서 인자 꺼내기)
        if model_type == 'division':
            intr = proj.intr
            sy_cur = intr.sy if intr.include_sy else sy0_fixed
            A_img = calculate_analytical_jacobian_division_model(
                X_EC, X_WB, T_BE_list, obj_pts_list,
                intr.c, intr.kappa, intr.sx, sy_cur, intr.cx, intr.cy,
                is_target_based=is_target_based, is_scara=is_scara,
                estimate_ec=estimate_ec, estimate_wb=estimate_wb, estimate_be=estimate_be,
                estimate_intrinsics=estimate_intrinsics, include_sy=intr.include_sy,
                mat_to_vec6d=mat_to_vec6d
            )
        else:
            intr = proj.intr
            A_img = calculate_analytical_jacobian_polynomial_model(
                T_ee_cam=X_EC, T_base_board=X_WB, T_be_list=T_BE_list, obj_pts_list=obj_pts_list,
                fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, dist_coeffs=intr.dist_coeffs,
                is_target_based=is_target_based, is_scara=is_scara,
                estimate_ec=estimate_ec, estimate_wb=estimate_wb, estimate_be=estimate_be,
                estimate_intrinsics=estimate_intrinsics,
                mat_to_vec6d=mat_to_vec6d
            )

        # --- (D) 완성된 A ---
        if estimate_be:
            A = np.zeros((ni_total + 6 * nr, layout['total']))
            A[:ni_total, :] = A_img
            A[ni_total:, layout.get('be', slice(0,0))] = np.eye(6 * nr)
        else:
            A = A_img

        N_mat = A.T @ (Pll_diag[:,None]*A)
        try:
            from scipy.linalg import cho_factor, cho_solve
            cF2, lower2 = cho_factor(N_mat, check_finite=False)
            diag_Qlhat = np.empty(ni_total + (6*nr if estimate_be else 0), dtype=float)
            for i in range(diag_Qlhat.size):
                ai = A[i,:]
                y = cho_solve((cF2, lower2), ai, check_finite=False)
                diag_Qlhat[i] = float(ai @ y)
        except Exception:
            Q_xx = np.linalg.pinv(N_mat)
            diag_Qlhat = np.einsum('ij,jk,ik->i', A, Q_xx, A)

        diag_R = 1.0 - Pll_diag * diag_Qlhat

        idx_i_end = ni_total
        R_i_sum = np.sum(diag_R[:idx_i_end])
        v_i = v_hat[:idx_i_end]
        sigma2_hat_i = (v_i.T @ (Pll_diag[:idx_i_end]*v_i) / max(R_i_sum, 1e-9))

        sigma2_hat_a, sigma2_hat_t = 1.0, 1.0
        if estimate_be:
            idx_a = np.concatenate([np.arange(i, i+3) for i in range(idx_i_end, idx_i_end + 6*nr, 6)])
            idx_t = np.concatenate([np.arange(i+3, i+6) for i in range(idx_i_end, idx_i_end + 6*nr, 6)])

            v_img = v_hat[:idx_i_end]
            v_ang = v_hat[idx_a]
            v_trn = v_hat[idx_t]

            print("img rmse [px]:", np.sqrt(np.mean(v_img**2)))
            print("ang std [deg]:", np.rad2deg(v_ang).std(), "mean:", np.rad2deg(v_ang).mean())
            print("trn std [mm]: ", v_trn.std(), "mean:", v_trn.mean())
            
            R_a_sum = np.sum(diag_R[idx_a]); v_a = v_hat[idx_a]
            R_t_sum = np.sum(diag_R[idx_t]); v_t = v_hat[idx_t]
            sigma2_hat_a = (v_a.T @ (Pll_diag[idx_a]*v_a) / max(R_a_sum, 1e-9))
            sigma2_hat_t = (v_t.T @ (Pll_diag[idx_t]*v_t) / max(R_t_sum, 1e-9))
            print(f"  Redundancy sums: R_img={R_i_sum:.2f}, R_ang={R_a_sum}, R_trans={R_t_sum}")
        else:
            print(f"  Redundancy sums: R_img={R_i_sum:.2f}")

        var_i = max(var_i * sigma2_hat_i, VAR_I_FLOOR)
        if estimate_be:
            var_a = max(var_a * sigma2_hat_a, VAR_A_FLOOR)
            var_t = max(var_t * sigma2_hat_t, VAR_T_FLOOR)

        print(f"  σ̂²_img={sigma2_hat_i:.4f}, σ̂²_ang={sigma2_hat_a:.4f}, σ̂²_trans={sigma2_hat_t:.4f}")

        if np.allclose([sigma2_hat_i, sigma2_hat_a, sigma2_hat_t], 1.0, atol=1e-3):
            print("  VCE converged.")
            break

    # --- 결과 intrinsics dict로 패킹 ---
    if model_type == 'division':
        intr = proj.intr
        final_intrinsics = {
            'c': intr.c, 'kappa': intr.kappa,
            'sx': intr.sx, 'sy': (intr.sy if intr.include_sy else sy0_fixed),
            'cx': intr.cx, 'cy': intr.cy
        }
    else:
        intr = proj.intr
        final_intrinsics = {
            'fx': intr.fx, 'fy': intr.fy, 'cx': intr.cx, 'cy': intr.cy,
            'dist_coeffs': intr.dist_coeffs.copy()
        }

    print("\nOptimization Finished.")
    return X_EC, X_WB, T_BE_list, final_intrinsics


def _project_point_division(pc, c, kappa, sx, sy, cx, cy):
    """
    pc: (3,) in C1 frame
    return: (u,v)
    """
    X, Y, Z = pc
    if Z <= 1e-12:
        return np.nan, np.nan
    ux, uy = c * (X/Z), c * (Y/Z)          # pu
    ru2 = ux*ux + uy*uy
    if abs(kappa) < 1e-15 or ru2 < 1e-24:
        xd, yd = ux, uy
    else:
        g = 1.0 - 4.0*kappa*ru2
        if g <= 0.0:
            return np.nan, np.nan
        Delta = np.sqrt(g)
        ru = np.sqrt(ru2)
        rd = (1.0 - Delta) / (2.0*kappa*ru)
        s = rd / (ru + 1e-12)
        xd, yd = s*ux, s*uy
    u = xd/sx + cx
    v = yd/sy + cy
    return u, v


def _project_dataset_flat_dual(
    X1_EC, T_B1B2, E2_C2,
    T_E1B1_list, T_B2E2_list, T_C2B_list,
    obj_pts_list,
    intr
):
    """체인: B --(C2B_i)--> C2 --(E2C2)--> E2 --(B2E2_i)--> B2 --(B1B2)--> B1 --(E1B1_i)--> E1 --(X1EC)--> C1"""
    c, kappa, sx, sy, cx, cy = intr['c'], intr['kappa'], intr['sx'], intr['sy'], intr['cx'], intr['cy']
    uv_all = []
    nr = len(obj_pts_list)
    for i in range(nr):
        R_c2b, t_c2b   = T_C2B_list[i][:3,:3], T_C2B_list[i][:3,3]
        R_e2c2, t_e2c2 = E2_C2[:3,:3], E2_C2[:3,3]
        R_b2e2, t_b2e2 = T_B2E2_list[i][:3,:3], T_B2E2_list[i][:3,3]
        R_b1b2, t_b1b2 = T_B1B2[:3,:3], T_B1B2[:3,3]
        R_e1b1, t_e1b1 = T_E1B1_list[i][:3,:3], T_E1B1_list[i][:3,3]
        R_x1,   t_x1   = X1_EC[:3,:3],  X1_EC[:3,3]

        for pw in obj_pts_list[i]:
            # s0=pw
            s1 = R_c2b @ pw + t_c2b           # C2
            s2 = R_e2c2 @ s1 + t_e2c2         # E2
            s3 = R_b2e2 @ s2 + t_b2e2         # B2
            s4 = R_b1b2 @ s3 + t_b1b2         # B1
            s5 = R_e1b1 @ s4 + t_e1b1         # E1
            pc = R_x1   @ s5 + t_x1           # C1
            u, v = _project_point_division(pc, c, kappa, sx, sy, cx, cy)
            uv_all.append(u); uv_all.append(v)
    return np.array(uv_all, dtype=float)


import matplotlib.pyplot as plt

# ---- 네 자코비안의 열 레이아웃을 재현 (듀얼 세팅 기준) ----
def build_layout_dual(nr,
                      estimate_x1ec=True, estimate_b1b2=True, estimate_e2c2=True,
                      estimate_e1b1=True, estimate_b2e2=True, estimate_c2b=False,
                      estimate_intrinsics=True, include_sy=False, is_scara_x1=False):
    layout, col = {}, 0
    def add(name, n):
        nonlocal col
        layout[name] = slice(col, col+n); col += n
    if estimate_x1ec: add('x1ec', 5 if is_scara_x1 else 6)
    if estimate_b1b2: add('b1b2', 6)
    if estimate_e2c2: add('e2c2', 6)
    if estimate_e1b1: add('e1b1', 6*nr)
    if estimate_b2e2: add('b2e2', 6*nr)
    if estimate_c2b:  add('c2b' , 6*nr)
    if estimate_intrinsics:
        add('intr', 6 if include_sy else 5)  # [c,κ,sx,(sy),cx,cy]
    layout['total'] = col
    return layout

def _draw_block_lines(ax, layout, labels_order=('x1ec','b1b2','e2c2','e1b1','b2e2','c2b','intr')):
    # 수직 경계선 + 라벨
    xmax = layout['total']
    x_centers = []
    x_ticks = []
    for k in labels_order:
        if k in layout:
            s = layout[k]
            ax.axvline(s.start, linewidth=0.7)
            ax.axvline(s.stop,  linewidth=0.7)
            x_centers.append(0.5*(s.start+s.stop))
            x_ticks.append(k)
    ax.set_xlim(0, xmax)
    if x_centers:
        ax.set_xticks(x_centers, x_ticks, rotation=0)

# ---- 1) 스파시티(0/비0) ----
def plot_jacobian_sparsity(A, layout, outfile=None, title='Jacobian sparsity'):
    M = np.nan_to_num(A, nan=0.0)
    mask = (M != 0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.spy(mask, markersize=1)
    _draw_block_lines(ax, layout)
    ax.set_title(title)
    ax.set_xlabel('parameters')
    ax.set_ylabel('residual rows')
    fig.tight_layout()
    if outfile: fig.savefig(outfile, dpi=200)
    plt.show()

# ---- 2) 히트맵(값 크기 시각화; log10|A| 권장) ----
def plot_jacobian_heatmap(A, layout, outfile=None, title='Jacobian |values| (log10)'):
    M = np.log10(np.abs(A) + 1e-12)  # outlier 완화
    # 보기 좋게 클리핑(하위/상위 분위수)
    lo, hi = np.nanpercentile(M[np.isfinite(M)], [5, 95])
    M = np.clip(M, lo, hi)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(M, aspect='auto', origin='upper', interpolation='nearest')
    _draw_block_lines(ax, layout)
    ax.set_title(title)
    ax.set_xlabel('parameters')
    ax.set_ylabel('residual rows')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if outfile: fig.savefig(outfile, dpi=200)
    plt.show()

# ---- 3) 열 노름(파라미터 민감도 한눈에) ----
def plot_jacobian_colnorms(A, layout, outfile=None, title='Column L2 norms of J'):
    col_norms = np.linalg.norm(np.nan_to_num(A, nan=0.0), axis=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(col_norms)
    _draw_block_lines(ax, layout)
    ax.set_title(title)
    ax.set_xlabel('parameter index')
    ax.set_ylabel('||col||₂')
    fig.tight_layout()
    if outfile: fig.savefig(outfile, dpi=200)
    plt.show()

def run_optimization_with_vce_dual(
    # 모델/데이터
    model_type: str,   # 'division'만 지원 (placeholder로 'polynomial' 경고)
    X1_EC_init: np.ndarray,      # ^C1 T_E1
    T_B1B2_init: np.ndarray,     # ^B1 T_B2
    E2_C2_init: np.ndarray,      # ^E2 T_C2
    T_E1B1_list_init: list,      # [^E1 T_B1]_i
    T_E2B2_list_init: list,      # [^E2 T_B2]_i   # <<< CHG: E2B2로 입력
    T_C2B_list_init: list,       # [^C2 T_B ]_i
    img_pts_list: list,          # cam1 관측 [(Ni,2)]_i
    obj_pts_list: list,          # 보드 좌표 [(Ni,3)]_i
    # 포즈 관측(옵션: fictitious obs, VCE용)
    T_E1B1_list_obs: list = None,
    T_E2B2_list_obs: list = None,  # <<< CHG: obs도 E2B2로 받음
    T_C2B_list_obs: list = None,
    # intrinsics
    intrinsics_init: dict = None,  # {'c','kappa','sx','sy','cx','cy','include_sy'}
    # 노이즈 (초기 분산)
    sigma_image_px: float = 0.1,
    sigma_angle_deg: float = 0.1,
    sigma_trans_mm: float = 1.0,
    # 반복/LM
    max_vce_iter: int = 5,
    max_param_iter: int = 10,
    term_thresh: float = 1e-6,
    # 추정 플래그
    estimate_x1ec: bool = True,
    estimate_b1b2: bool = True,
    estimate_e2c2: bool = True,
    estimate_e1b1: bool = True,   # per-pose
    estimate_b2e2: bool = True,   # per-pose  (키는 b2e2로 유지, 내용은 E2B2)
    estimate_c2b:  bool = False,  # per-pose (보통 False: Charuco PnP 고정)
    estimate_intrinsics: bool = False,
    include_sy: bool = False,
    is_scara_x1: bool = False,
    # --- VCE log 옵션 (구버전 파이썬 호환: 타입힌트 생략)
    vce_log=None,              # 이벤트 dict를 append할 리스트 (옵션)
    collect_vce_hist: bool = False,
):
    assert model_type in ['division', 'polynomial'], "지원 모델: 'division' 또는 'polynomial'(placeholder)"
    if model_type != 'division':
        print("[경고] 현재 듀얼 솔버는 division 모델만 구현되었습니다.")

    # --- VCE gain control (클리핑/스무딩) ---
    VCE_GAIN_MIN = 0.5
    VCE_GAIN_MAX = 2.0
    VCE_ETA      = 0.5
    MIN_REDUNDANCY = 10.0  # 자유도가 너무 작으면 업데이트 스킵

    def inv4(T: np.ndarray):
        """4x4 동차 변환 행렬의 역행렬을 계산합니다."""
        R, t = T[:3, :3], T[:3, 3]
        Ti = np.eye(4)
        Ti[:3, :3] = R.T
        Ti[:3, 3] = -R.T @ t
        return Ti

    def _clipped_gain(g, gmin, gmax, eta):
        if not np.isfinite(g):
            return 1.0
        g_clip = float(np.clip(g, gmin, gmax))
        return g_clip ** eta

    nr = len(obj_pts_list)

    # --- 상태 변수 초기화 ---
    X1_EC   = X1_EC_init.copy()
    T_B1B2  = T_B1B2_init.copy()
    E2_C2   = E2_C2_init.copy()
    T_E1B1_list = [T.copy() for T in T_E1B1_list_init]
    T_E2B2_list = [T.copy() for T in T_E2B2_list_init]   # <<< CHG: 상태는 E2B2로 유지
    T_C2B_list  = [T.copy() for T in T_C2B_list_init]

    # --- intrinsics 상태 ---
    if intrinsics_init is None:
        intrinsics_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=(1.0 if include_sy else 1.0), cx=0.0, cy=0.0, include_sy=include_sy)
    intr = {
        'c': intrinsics_init['c'],
        'kappa': intrinsics_init['kappa'],
        'sx': intrinsics_init['sx'],
        'sy': intrinsics_init['sy'],
        'cx': intrinsics_init['cx'],
        'cy': intrinsics_init['cy'],
        'include_sy': include_sy
    }

    # --- 관측 벡터 구성 ---
    l_obs_img = np.concatenate([pts.reshape(-1) for pts in img_pts_list])
    obs_blocks = []
    if estimate_e1b1 and (T_E1B1_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]))
    if estimate_b2e2 and (T_E2B2_list_obs is not None):    # <<< CHG: E2B2 직접 비교
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs]))
    if estimate_c2b and (T_C2B_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs]))
    l_obs = np.concatenate([l_obs_img, *obs_blocks]) if len(obs_blocks)>0 else l_obs_img
    ni_img = len(l_obs_img)

    # --- 초기 분산/LM 파라미터 ---
    VAR_I_FLOOR, VAR_A_FLOOR, VAR_T_FLOOR = 1e-12, (np.deg2rad(1e-9))**2, (1e-9)**2
    var_i = max(sigma_image_px**2, VAR_I_FLOOR)
    var_a = max(np.deg2rad(sigma_angle_deg)**2, VAR_A_FLOOR)
    var_t = max((sigma_trans_mm / 1000.0)**2, VAR_T_FLOOR)
    lam, lam_up, lam_down, lam_min, lam_max, max_ls_tries = 1e-2, 5.0, 0.25, 1e-12, 1e+8, 8

    # --- 레이아웃 (키는 'b2e2' 유지) ---
    def compute_layout():
        layout = {}; col = 0
        dim_x1 = 5 if is_scara_x1 else 6
        if estimate_x1ec: layout['x1ec'] = slice(col, col+dim_x1); col += dim_x1
        if estimate_b1b2: layout['b1b2'] = slice(col, col+6); col += 6
        if estimate_e2c2: layout['e2c2'] = slice(col, col+6); col += 6
        if estimate_e1b1: layout['e1b1'] = slice(col, col+6*nr); col += 6*nr
        if estimate_b2e2: layout['b2e2'] = slice(col, col+6*nr); col += 6*nr  # 내용은 E2B2
        if estimate_c2b:  layout['c2b']  = slice(col, col+6*nr); col += 6*nr
        if estimate_intrinsics:
            layout['intr'] = slice(col, col + (6 if include_sy else 5)); col += (6 if include_sy else 5)
        layout['total'] = col
        return layout

    def build_x_current(layout):
        vecs = []
        if estimate_x1ec:
            e6 = mat_to_vec6d(X1_EC); vecs.append(e6[:5] if is_scara_x1 else e6)
        if estimate_b1b2: vecs.append(mat_to_vec6d(T_B1B2))
        if estimate_e2c2: vecs.append(mat_to_vec6d(E2_C2))
        if estimate_e1b1: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
        if estimate_b2e2: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list]))  # <<< CHG: E2B2
        if estimate_c2b:  vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_C2B_list]))
        if estimate_intrinsics:
            if include_sy:
                vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['sy'], intr['cx'], intr['cy']], dtype=float))
            else:
                vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['cx'], intr['cy']], dtype=float))
        return np.concatenate(vecs) if len(vecs)>0 else np.zeros(0)

    print("="*50); print("Dual-Arm Division Optimization Start (Single-Cam)"); print("="*50)

    # ── vlog 준비 ───────────────────────────────────────────
    have_log = vce_log is not None
    if have_log:
        clip_counts = {'img': {'under':0,'over':0}, 'ang': {'under':0,'over':0}, 'trans': {'under':0,'over':0}}
    if collect_vce_hist and have_log:
        sigma0_hist = {'img': [], 'ang': [], 'trans': []}

    for vce_iter in range(max_vce_iter):
        print(f"\n--- VCE Iteration {vce_iter+1}/{max_vce_iter} ---")
        print(f"Variances: σ_img²={var_i:.3e}, σ_ang²={var_a:.3e}, σ_trans²={var_t:.3e}")
        
        # 관측 가중치 대각
        wi = 1.0/var_i
        Pll_diag_list = [np.full(ni_img, wi)]
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            Pll_diag_list.append(np.tile(np.concatenate([np.full(3, 1.0/var_a), np.full(3, 1.0/var_t)]), nr))
        if estimate_b2e2 and (T_E2B2_list_obs is not None):  # <<< CHG
            Pll_diag_list.append(np.tile(np.concatenate([np.full(3, 1.0/var_a), np.full(3, 1.0/var_t)]), nr))
        if estimate_c2b and (T_C2B_list_obs is not None):
            Pll_diag_list.append(np.tile(np.concatenate([np.full(3, 1.0/var_a), np.full(3, 1.0/var_t)]), nr))
        Pll_diag = np.concatenate(Pll_diag_list) if len(Pll_diag_list)>1 else Pll_diag_list[0]
        
        layout = compute_layout()
        phi_best = np.inf

        # ------------ LM 내부 반복 ------------
        for param_iter in range(max_param_iter):
            x_k = build_x_current(layout)

            # (A) 현재 파라미터로 예측 픽셀
            #     자코비안/프로젝션 함수는 B2E2를 기대하므로 on-the-fly로 inv
            T_B2E2_list_for_proj = [inv4(T) for T in T_E2B2_list]  # <<< CHG
            f_pix = _project_dataset_flat_dual(
                X1_EC, T_B1B2, E2_C2, T_E1B1_list, T_B2E2_list_for_proj, T_C2B_list,
                obj_pts_list, intr
            )

            # (B) 잔차 w 구성
            w_list = [l_obs_img - f_pix]
            idx_after = ni_img
            if estimate_e1b1 and (T_E1B1_list_obs is not None):
                f_e1b1 = x_k[layout['e1b1']] if estimate_e1b1 else np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list])
                obs_e1b1 = np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs])
                w_list.append(obs_e1b1 - f_e1b1)
                idx_after += 6*nr
            if estimate_b2e2 and (T_E2B2_list_obs is not None):  # <<< CHG
                f_e2b2 = x_k[layout['b2e2']] if estimate_b2e2 else np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list])
                obs_e2b2 = np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs])
                w_list.append(obs_e2b2 - f_e2b2)
                idx_after += 6*nr
            if estimate_c2b and (T_C2B_list_obs is not None):
                f_c2b = x_k[layout['c2b']] if estimate_c2b else np.concatenate([mat_to_vec6d(T) for T in T_C2B_list])
                obs_c2b = np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs])
                w_list.append(obs_c2b - f_c2b)
                idx_after += 6*nr
            w = np.concatenate(w_list)
            if np.isnan(w).any():
                print(f"  [LM-{param_iter+1}] NaN residual detected. Stop.")
                break

            # (C) 이미지 자코비안 (듀얼 전용 해석 자코비안) — B2E2 기대 → inv 적용
            A_img = calculate_analytical_jacobian_division_model_dual(
                X1_EC=X1_EC, T_B1B2=T_B1B2, E2_C2=E2_C2,
                T_E1B1_list=T_E1B1_list,
                T_E2B2_list=T_E2B2_list,
                T_C2B_list=T_C2B_list,
                obj_pts_list=obj_pts_list,
                c=intr['c'], kappa=intr['kappa'], sx=intr['sx'], sy=intr['sy'], cx=intr['cx'], cy=intr['cy'],
                estimate_x1ec=estimate_x1ec, estimate_e1b1=estimate_e1b1,
                estimate_b1b2=estimate_b1b2, estimate_b2e2=estimate_b2e2,
                estimate_e2c2=estimate_e2c2, estimate_c2b=estimate_c2b,
                estimate_intrinsics=estimate_intrinsics,
                include_sy=include_sy, is_scara_x1=is_scara_x1,
                mat_to_vec6d=mat_to_vec6d
            )

            # (D) 전체 A (이미지 + 포즈 항 I)
            A = np.zeros((w.size, layout['total']), dtype=float)
            A[:ni_img, :] = A_img
            row_ptr = ni_img
            if estimate_e1b1 and (T_E1B1_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
            if estimate_b2e2 and (T_E2B2_list_obs is not None):  # <<< CHG
                A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr
            if estimate_c2b and (T_C2B_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['c2b']] = np.eye(6*nr); row_ptr += 6*nr

            # (E) LM step
            phi0 = float(np.sum(Pll_diag * (w**2)))
            N = A.T @ (Pll_diag[:,None]*A)
            b = A.T @ (Pll_diag*w)

            success = False
            for ls_try in range(max_ls_tries):
                try:
                    N_aug = N + lam * np.diag(np.diag(N))
                    delta = np.linalg.solve(N_aug, b)
                except np.linalg.LinAlgError:
                    lam = min(lam * lam_up, lam_max); continue

                x_new = x_k + delta
                if not np.isfinite(x_new).all():
                    lam = min(lam * lam_up, lam_max); continue

                # (E-1) 파라미터 갱신 (상태는 E2B2 유지)
                try:
                    X1_EC_new = X1_EC
                    if estimate_x1ec:
                        v = x_new[layout['x1ec']]
                        if is_scara_x1:
                            base = mat_to_vec6d(X1_EC); base[:5] = v; v6 = np.array([*v, base[5]])
                            X1_EC_new = vec6d_to_mat(v6)
                        else:
                            X1_EC_new = vec6d_to_mat(v)
                    T_B1B2_new = vec6d_to_mat(x_new[layout['b1b2']]) if estimate_b1b2 else T_B1B2
                    E2_C2_new  = vec6d_to_mat(x_new[layout['e2c2']])  if estimate_e2c2  else E2_C2

                    T_E1B1_list_new = T_E1B1_list if not estimate_e1b1 else [vec6d_to_mat(a) for a in x_new[layout['e1b1']].reshape(nr,6)]
                    T_E2B2_list_new = T_E2B2_list if not estimate_b2e2 else [vec6d_to_mat(a) for a in x_new[layout['b2e2']].reshape(nr,6)]  # <<< CHG
                    T_C2B_list_new  = T_C2B_list  if not estimate_c2b  else [vec6d_to_mat(a) for a in x_new[layout['c2b'] ].reshape(nr,6)]

                    intr_new = intr
                    if estimate_intrinsics:
                        iv = x_new[layout['intr']]
                        if include_sy:
                            intr_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':iv[3],'cx':iv[4],'cy':iv[5],'include_sy':True}
                        else:
                            intr_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':intr['sy'],'cx':iv[3],'cy':iv[4],'include_sy':False}
                except Exception:
                    lam = min(lam * lam_up, lam_max); continue

                # (E-2) 새 잔차 평가 (프로젝션은 B2E2 필요 → inv)
                f_pix_new = _project_dataset_flat_dual(
                    X1_EC_new, T_B1B2_new, E2_C2_new,
                    T_E1B1_list_new, [inv4(T) for T in T_E2B2_list_new], T_C2B_list_new,  # <<< CHG
                    obj_pts_list, intr_new
                )
                w_list_new = [l_obs_img - f_pix_new]
                if estimate_e1b1 and (T_E1B1_list_obs is not None):
                    obs_e1b1 = np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs])
                    w_list_new.append(obs_e1b1 - x_new[layout['e1b1']])
                if estimate_b2e2 and (T_E2B2_list_obs is not None):  # <<< CHG
                    obs_e2b2 = np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs])
                    w_list_new.append(obs_e2b2 - x_new[layout['b2e2']])
                if estimate_c2b and (T_C2B_list_obs is not None):
                    obs_c2b = np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs])
                    w_list_new.append(obs_c2b - x_new[layout['c2b']])

                w_new = np.concatenate(w_list_new)
                if np.isnan(w_new).any():
                    lam = min(lam * lam_up, lam_max); continue
                phi_new = float(np.sum(Pll_diag * (w_new**2)))

                if phi_new < phi0:
                    X1_EC, T_B1B2, E2_C2 = X1_EC_new, T_B1B2_new, E2_C2_new
                    T_E1B1_list, T_E2B2_list, T_C2B_list = T_E1B1_list_new, T_E2B2_list_new, T_C2B_list_new
                    intr = intr_new
                    lam = max(lam * lam_down, lam_min)
                    phi_best = phi_new
                    success = True
                    break
                else:
                    lam = min(lam * lam_up, lam_max)

            if not success:
                print(f"  [LM-{param_iter+1}] No improvement after {max_ls_tries} tries.")
                break
            if abs(phi0 - phi_best) < term_thresh:
                print("  [LM] Converged. Stop parameter search.")
                break

        # ------------ (F) VCE 업데이트 ------------
        print("  Updating variance components...")
        f_pix_final = _project_dataset_flat_dual(
            X1_EC, T_B1B2, E2_C2,
            T_E1B1_list, [inv4(T) for T in T_E2B2_list], T_C2B_list,  # <<< CHG
            obj_pts_list, intr
        )
        v_list = [l_obs_img - f_pix_final]
        x_now = build_x_current(layout)
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            obs_e1b1 = np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs])
            v_list.append(obs_e1b1 - x_now[layout['e1b1']])
        if estimate_b2e2 and (T_E2B2_list_obs is not None):  # <<< CHG
            obs_e2b2 = np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs])
            v_list.append(obs_e2b2 - x_now[layout['b2e2']])
        if estimate_c2b and (T_C2B_list_obs is not None):
            obs_c2b = np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs])
            v_list.append(obs_c2b - x_now[layout['c2b']])
        v_hat = np.concatenate(v_list)

        # 자코비안 재계산 (최종 파라미터) — B2E2 기대 → inv
        A_img = calculate_analytical_jacobian_division_model_dual(
            X1_EC=X1_EC, T_B1B2=T_B1B2, E2_C2=E2_C2,
            T_E1B1_list=T_E1B1_list,
            T_E2B2_list=T_E2B2_list,   # <<< CHG
            T_C2B_list=T_C2B_list,
            obj_pts_list=obj_pts_list,
            c=intr['c'], kappa=intr['kappa'], sx=intr['sx'], sy=intr['sy'], cx=intr['cx'], cy=intr['cy'],
            estimate_x1ec=estimate_x1ec, estimate_e1b1=estimate_e1b1,
            estimate_b1b2=estimate_b1b2, estimate_b2e2=estimate_b2e2,
            estimate_e2c2=estimate_e2c2, estimate_c2b=estimate_c2b,
            estimate_intrinsics=estimate_intrinsics,
            include_sy=include_sy, is_scara_x1=is_scara_x1,
            mat_to_vec6d=mat_to_vec6d
        )
        A = np.zeros((v_hat.size, layout['total']), dtype=float)
        A[:ni_img, :] = A_img
        row_ptr = ni_img
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
        if estimate_b2e2 and (T_E2B2_list_obs is not None):   # <<< CHG
            A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr
        if estimate_c2b and (T_C2B_list_obs is not None):
            A[row_ptr:row_ptr+6*nr, layout['c2b']] = np.eye(6*nr); row_ptr += 6*nr

        # N = A^T P A, redundancy
        N_mat = A.T @ (Pll_diag[:,None]*A)
        try:
            from scipy.linalg import cho_factor, cho_solve
            cF, lower = cho_factor(N_mat, check_finite=False)
            diag_Qlhat = np.empty(v_hat.size, dtype=float)
            for i in range(v_hat.size):
                ai = A[i,:]
                y = cho_solve((cF, lower), ai, check_finite=False)
                diag_Qlhat[i] = float(ai @ y)
        except Exception:
            Q_xx = np.linalg.pinv(N_mat)
            diag_Qlhat = np.einsum('ij,jk,ik->i', A, Q_xx, A)

        diag_R = 1.0 - Pll_diag * diag_Qlhat

        # --- 그룹별 σ̂0² ---
        R_img = float(np.sum(diag_R[:ni_img]))
        v_img = v_hat[:ni_img]
        sigma0_sq_img = float(v_img.T @ (Pll_diag[:ni_img] * v_img) / max(R_img, 1e-9))

        idx_ang_all, idx_trans_all = [], []
        offset = ni_img
        def _collect_pose_indices(enabled):
            nonlocal offset
            if not enabled: return
            base = offset
            for p in range(nr):
                b = base + 6*p
                idx_ang_all.extend([b+0, b+1, b+2]); idx_trans_all.extend([b+3, b+4, b+5])
            offset += 6*nr
        _collect_pose_indices(estimate_e1b1 and (T_E1B1_list_obs is not None))
        _collect_pose_indices(estimate_b2e2 and (T_E2B2_list_obs is not None))  # <<< CHG
        _collect_pose_indices(estimate_c2b  and (T_C2B_list_obs  is not None))

        idx_ang = np.array(idx_ang_all, int); idx_trn = np.array(idx_trans_all, int)
        sigma0_sq_ang = float((v_hat[idx_ang].T @ (Pll_diag[idx_ang]*v_hat[idx_ang])) / max(np.sum(diag_R[idx_ang]), 1e-9)) if idx_ang.size else None
        sigma0_sq_trn = float((v_hat[idx_trn].T @ (Pll_diag[idx_trn]*v_hat[idx_trn])) / max(np.sum(diag_R[idx_trn]), 1e-9)) if idx_trn.size else None

        # (선택) 히스토리 기록
        if collect_vce_hist and have_log:
            sigma0_hist['img'].append(float(sigma0_sq_img))
            if sigma0_sq_ang is not None: sigma0_hist['ang'].append(float(sigma0_sq_ang))
            if sigma0_sq_trn is not None: sigma0_hist['trans'].append(float(sigma0_sq_trn))

        # 클리핑 이벤트 로그
        def _check_clip_and_log(group_name, raw_val):
            if not have_log: return
            if not np.isfinite(raw_val):
                vce_log.append({'iter': int(vce_iter), 'group': group_name, 'raw_sigma0_sq': None, 'status': 'nan_or_inf'})
                return
            if raw_val < VCE_GAIN_MIN:
                clip_counts[group_name]['under'] += 1
                vce_log.append({'iter': int(vce_iter), 'group': group_name, 'raw_sigma0_sq': float(raw_val),
                                'clip_min': float(VCE_GAIN_MIN), 'clip_max': float(VCE_GAIN_MAX), 'status': 'under'})
                print(f"  [VCE-clip] iter={vce_iter} group={group_name} raw={raw_val:.4f} → < {VCE_GAIN_MIN}")
            elif raw_val > VCE_GAIN_MAX:
                clip_counts[group_name]['over'] += 1
                vce_log.append({'iter': int(vce_iter), 'group': group_name, 'raw_sigma0_sq': float(raw_val),
                                'clip_min': float(VCE_GAIN_MIN), 'clip_max': float(VCE_GAIN_MAX), 'status': 'over'})
                print(f"  [VCE-clip] iter={vce_iter} group={group_name} raw={raw_val:.4f} → > {VCE_GAIN_MAX}")

        _check_clip_and_log('img', sigma0_sq_img)
        if sigma0_sq_ang is not None: _check_clip_and_log('ang', sigma0_sq_ang)
        if sigma0_sq_trn is not None: _check_clip_and_log('trans', sigma0_sq_trn)

        print("  σ̂0² (should → 1): "
              f"img={sigma0_sq_img:.4f}"
              + (f", ang={sigma0_sq_ang:.4f}" if sigma0_sq_ang is not None else "")
              + (f", trans={sigma0_sq_trn:.4f}" if sigma0_sq_trn is not None else ""))

        # --- (클리핑/스무딩) 분산 업데이트 ---
        old_i, old_a, old_t = var_i, var_a, var_t
        g_img = _clipped_gain(sigma0_sq_img, VCE_GAIN_MIN, VCE_GAIN_MAX, VCE_ETA)
        var_i = max(var_i * g_img, VAR_I_FLOOR)
        if sigma0_sq_ang is not None:
            g_ang = _clipped_gain(sigma0_sq_ang, VCE_GAIN_MIN, VCE_GAIN_MAX, VCE_ETA)
            var_a = max(var_a * g_ang, VAR_A_FLOOR)
        if sigma0_sq_trn is not None:
            g_trn = _clipped_gain(sigma0_sq_trn, VCE_GAIN_MIN, VCE_GAIN_MAX, VCE_ETA)
            var_t = max(var_t * g_trn, VAR_T_FLOOR)

        print(f"  σ² update (clipped): img {old_i:.3e}→{var_i:.3e}"
              + (f" | ang {old_a:.3e}→{var_a:.3e}" if sigma0_sq_ang is not None else "")
              + (f" | trans {old_t:.3e}→{var_t:.3e}" if sigma0_sq_trn is not None else ""))

    # 요약 로그
    if have_log:
        summary = (f"[VCE-clip-summary] img(u/o)={clip_counts['img']['under']}/{clip_counts['img']['over']}, "
                   f"ang(u/o)={clip_counts['ang']['under']}/{clip_counts['ang']['over']}, "
                   f"trans(u/o)={clip_counts['trans']['under']}/{clip_counts['trans']['over']}")
        print(summary)
        vce_log.append({'summary': summary})
        if collect_vce_hist:
            vce_log.append({'sigma0_hist': sigma0_hist})

    # 결과 intrinsics dict
    final_intrinsics = {'c': intr['c'], 'kappa': intr['kappa'], 'sx': intr['sx'], 'sy': intr['sy'], 'cx': intr['cx'], 'cy': intr['cy'], 'include_sy': include_sy}
    print("\nDual-Arm Division Optimization Finished.")
    return X1_EC, T_B1B2, E2_C2, T_E1B1_list, T_E2B2_list, T_C2B_list, final_intrinsics

def _is_close_to_one(x, tol=1e-3):
    """x가 None이면 자동 충족, 숫자면 |x-1|<tol"""
    return (x is None) or (abs(x - 1.0) < tol)

import copy

def verify_jacobian(
    h=1e-7,  # 수치 미분을 위한 미소 변화량
    # --- run_optimization_with_vce_dual_bicamera의 모든 인자를 그대로 전달 ---
    **kwargs 
):
    """
    해석적 야코비안과 수치적 야코비안을 직접 비교하여 정확성을 검증합니다.
    기존 최적화 함수와 동일한 인자를 받습니다.
    """
    print(" Jacobian Verification Start ".center(80, "="))

    # --- 0. 최적화 함수 내부의 유틸리티 함수들을 이 함수 내에 재정의 ---
    # (캡슐화를 위해 내부에 재정의하는 것이 가장 안전합니다)
    
    nr = len(kwargs['obj_pts_list'])
    is_scara_x1 = kwargs.get('is_scara_x1', False)
    
    # 필요한 유틸리티 함수들을 kwargs에서 가져옵니다.
    # 이 함수들이 스크립트 내에 정의되어 있어야 합니다.
    # from your_project.utils import mat_to_vec6d, vec6d_to_mat
    # from your_project.camera_model import _project_point_division
    # mat_to_vec6d = calculate_analytical_jacobian_division_model_dual_bicamera.__globals__['mat_to_vec6d']
    # vec6d_to_mat = calculate_analytical_jacobian_division_model_dual_bicamera.__globals__['vec6d_to_mat']
    # _project_point_division = calculate_analytical_jacobian_division_model_dual_bicamera.__globals__['_project_point_division']
    
    def _inv_T(T):
        R, t = T[:3, :3], T[:3, 3]
        RT = R.T
        Tinv = np.eye(4, dtype=float)
        Tinv[:3, :3] = RT
        Tinv[:3, 3] = -RT @ t
        return Tinv

    def _project_dataset_flat_cam1(X1_EC_, T_B1B2_, E2_C2_, T_E1B1_list_, T_E2B2_list_, T_C2B_list_, obj_pts_list_, intr_):
        c, kappa, sx, sy, cx, cy = intr_['c'], intr_['kappa'], intr_['sx'], intr_['sy'], intr_['cx'], intr_['cy']
        out = []
        R_x1, t_x1 = X1_EC_[:3,:3], X1_EC_[:3,3]
        R_b12,t_b12= T_B1B2_[:3,:3], T_B1B2_[:3,3]
        R_e2c2,t_e2c2=E2_C2_[:3,:3], E2_C2_[:3,3]
        for i in range(nr):
            R_c2b,t_c2b   = T_C2B_list_[i][:3,:3], T_C2B_list_[i][:3,3]
            R_e1b1,t_e1b1 = T_E1B1_list_[i][:3,:3], T_E1B1_list_[i][:3,3]
            T_b2e2_i      = _inv_T(T_E2B2_list_[i])
            R_b2e2,t_b2e2 = T_b2e2_i[:3,:3], T_b2e2_i[:3,3]
            for pw in obj_pts_list_[i]:
                s1 = R_c2b @ pw + t_c2b; s2 = R_e2c2 @ s1 + t_e2c2
                s3 = R_b2e2 @ s2 + t_b2e2; s4 = R_b12  @ s3 + t_b12
                s5 = R_e1b1 @ s4 + t_e1b1; pc = R_x1   @ s5 + t_x1
                u,v = _project_point_division(pc, c, kappa, sx, sy, cx, cy)
                out.extend((u,v))
        return np.array(out, float)

    def _project_dataset_flat_cam2(X1_EC_, T_B1B2_, E2_C2_, T_E1B1_list_, T_E2B2_list_, T_C1B_list_, obj_pts_list_, intr_):
        c, kappa, sx, sy, cx, cy = intr_['c'], intr_['kappa'], intr_['sx'], intr_['sy'], intr_['cx'], intr_['cy']
        out = []
        R_x1, t_x1 = X1_EC_[:3,:3], X1_EC_[:3,3]; RT_x1 = R_x1.T
        R_b12,t_b12= T_B1B2_[:3,:3], T_B1B2_[:3,3]; RT_b12 = R_b12.T
        R_e2c2,t_e2c2=E2_C2_[:3,:3], E2_C2_[:3,3]; RT_e2c2 = R_e2c2.T
        for i in range(nr):
            R_c1b,t_c1b   = T_C1B_list_[i][:3,:3], T_C1B_list_[i][:3,3]
            R_e1b1,t_e1b1 = T_E1B1_list_[i][:3,:3], T_E1B1_list_[i][:3,3]; RT_e1b1 = R_e1b1.T
            R_e2b2,t_e2b2 = T_E2B2_list_[i][:3,:3], T_E2B2_list_[i][:3,3]
            for pw in obj_pts_list_[i]:
                q1 = R_c1b @ pw + t_c1b; q2 = RT_x1 @ (q1 - t_x1)
                q3 = RT_e1b1 @ (q2 - t_e1b1); q4 = RT_b12 @ (q3 - t_b12)
                q5 = R_e2b2 @ q4 + t_e2b2; pc = RT_e2c2 @ (q5 - t_e2c2)
                u,v = _project_point_division(pc, c, kappa, sx, sy, cx, cy)
                out.extend((u,v))
        return np.array(out, float)

    def _project_dataset_flat_bicamera(X1_EC_, T_B1B2_, E2_C2_, T_E1B1_list_, T_E2B2_list_, T_C2B_list_, T_C1B_list_, obj_pts_list_, intr1_, intr2_):
        f1 = _project_dataset_flat_cam1(X1_EC_, T_B1B2_, E2_C2_, T_E1B1_list_, T_E2B2_list_, T_C2B_list_, obj_pts_list_, intr1_)
        f2 = _project_dataset_flat_cam2(X1_EC_, T_B1B2_, E2_C2_, T_E1B1_list_, T_E2B2_list_, T_C1B_list_, obj_pts_list_, intr2_)
        return np.concatenate([f1, f2])

    def compute_layout():
        layout = {}; col = 0
        dim_x1 = 5 if is_scara_x1 else 6
        if kwargs['estimate_x1ec']: layout['x1ec'] = slice(col, col+dim_x1); col += dim_x1
        if kwargs['estimate_b1b2']: layout['b1b2'] = slice(col, col+6); col += 6
        if kwargs['estimate_e2c2']: layout['e2c2'] = slice(col, col+6); col += 6
        if kwargs['estimate_e1b1']: layout['e1b1'] = slice(col, col+6*nr); col += 6*nr
        if kwargs['estimate_b2e2']: layout['b2e2'] = slice(col, col+6*nr); col += 6*nr
        if kwargs['estimate_c2b']:  layout['c2b']  = slice(col, col+6*nr); col += 6*nr
        if kwargs['estimate_c1b']:  layout['c1b']  = slice(col, col+6*nr); col += 6*nr
        if kwargs['estimate_intr1']: layout['intr1'] = slice(col, col+(6 if kwargs['include_sy1'] else 5)); col += (6 if kwargs['include_sy1'] else 5)
        if kwargs['estimate_intr2']: layout['intr2'] = slice(col, col+(6 if kwargs['include_sy2'] else 5)); col += (6 if kwargs['include_sy2'] else 5)
        layout['total'] = col
        return layout

    def build_x_current(layout, X1_EC, T_B1B2, E2_C2, T_E1B1_list, T_E2B2_list, T_C2B_list, T_C1B_list, intr1, intr2):
        vecs = []
        if kwargs['estimate_x1ec']:
            e6 = mat_to_vec6d(X1_EC); vecs.append(e6[:5] if is_scara_x1 else e6)
        if kwargs['estimate_b1b2']: vecs.append(mat_to_vec6d(T_B1B2))
        if kwargs['estimate_e2c2']: vecs.append(mat_to_vec6d(E2_C2))
        if kwargs['estimate_e1b1']: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
        if kwargs['estimate_b2e2']: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list]))
        if kwargs['estimate_c2b']:  vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_C2B_list]))
        if kwargs['estimate_c1b']:  vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_C1B_list]))
        if kwargs['estimate_intr1']:
            v = [intr1['c'], intr1['kappa'], intr1['sx']]
            if kwargs['include_sy1']: v.append(intr1['sy'])
            v.extend([intr1['cx'], intr1['cy']]); vecs.append(np.array(v, float))
        if kwargs['estimate_intr2']:
            v = [intr2['c'], intr2['kappa'], intr2['sx']]
            if kwargs['include_sy2']: v.append(intr2['sy'])
            v.extend([intr2['cx'], intr2['cy']]); vecs.append(np.array(v, float))
        return np.concatenate(vecs) if len(vecs)>0 else np.zeros(0)

    # --- 1. 해석적 야코비안 계산 ---
    J_analytical, layout = calculate_analytical_jacobian_division_model_dual_bicamera(
        X1_EC=kwargs['X1_EC_init'], T_B1B2=kwargs['T_B1B2_init'], E2_C2=kwargs['E2_C2_init'],
        T_E1B1_list=kwargs['T_E1B1_list_init'], T_E2B2_list=kwargs['T_E2B2_list_init'],
        T_C2B_list=kwargs['T_C2B_list_init'], T_C1B_list=kwargs['T_C1B_list_init'],
        obj_pts_list=kwargs['obj_pts_list'],
        c1=kwargs['intr1_init']['c'], kappa1=kwargs['intr1_init']['kappa'], sx1=kwargs['intr1_init']['sx'], sy1=kwargs['intr1_init']['sy'], cx1=kwargs['intr1_init']['cx'], cy1=kwargs['intr1_init']['cy'],
        c2=kwargs['intr2_init']['c'], kappa2=kwargs['intr2_init']['kappa'], sx2=kwargs['intr2_init']['sx'], sy2=kwargs['intr2_init']['sy'], cx2=kwargs['intr2_init']['cx'], cy2=kwargs['intr2_init']['cy'],
        estimate_x1ec=kwargs['estimate_x1ec'], estimate_b1b2=kwargs['estimate_b1b2'], estimate_e2c2=kwargs['estimate_e2c2'],
        estimate_e1b1=kwargs['estimate_e1b1'], estimate_b2e2=kwargs['estimate_b2e2'],
        estimate_c2b=kwargs['estimate_c2b'], estimate_c1b=kwargs['estimate_c1b'],
        estimate_intr1=kwargs['estimate_intr1'], estimate_intr2=kwargs['estimate_intr2'],
        include_sy1=kwargs['include_sy1'], include_sy2=kwargs['include_sy2'],
        is_scara_x1=is_scara_x1,
        mat_to_vec6d=mat_to_vec6d
    )
    print("1. Analytical Jacobian calculated.")

    # --- 2. 수치적 야코비안 계산 (Ground Truth) ---
    def project_wrapper(params_flat):
        X1_EC_temp = copy.deepcopy(kwargs['X1_EC_init']); T_B1B2_temp = copy.deepcopy(kwargs['T_B1B2_init'])
        E2_C2_temp = copy.deepcopy(kwargs['E2_C2_init']); T_E1B1_list_temp = copy.deepcopy(kwargs['T_E1B1_list_init'])
        T_E2B2_list_temp = copy.deepcopy(kwargs['T_E2B2_list_init']); T_C2B_list_temp = copy.deepcopy(kwargs['T_C2B_list_init'])
        T_C1B_list_temp = copy.deepcopy(kwargs['T_C1B_list_init']); intr1_temp = copy.deepcopy(kwargs['intr1_init'])
        intr2_temp = copy.deepcopy(kwargs['intr2_init'])

        if kwargs['estimate_x1ec']: X1_EC_temp = vec6d_to_mat(params_flat[layout['x1ec']])
        if kwargs['estimate_b1b2']: T_B1B2_temp = vec6d_to_mat(params_flat[layout['b1b2']])
        if kwargs['estimate_e2c2']: E2_C2_temp = vec6d_to_mat(params_flat[layout['e2c2']])
        if kwargs['estimate_e1b1']: T_E1B1_list_temp = [vec6d_to_mat(v) for v in params_flat[layout['e1b1']].reshape(-1, 6)]
        if kwargs['estimate_b2e2']: T_E2B2_list_temp = [vec6d_to_mat(v) for v in params_flat[layout['b2e2']].reshape(-1, 6)]
        
        if kwargs['estimate_intr1']:
            v_intr = params_flat[layout['intr1']]
            intr1_temp['c'] = v_intr[0]
            intr1_temp['kappa'] = v_intr[1]
            intr1_temp['sx'] = v_intr[2]
            if kwargs['include_sy1']:
                intr1_temp['sy'] = v_intr[3]
                intr1_temp['cx'] = v_intr[4]
                intr1_temp['cy'] = v_intr[5]
            else:
                intr1_temp['cx'] = v_intr[3]
                intr1_temp['cy'] = v_intr[4]

        if kwargs['estimate_intr2']:
            v_intr = params_flat[layout['intr2']]
            intr2_temp['c'] = v_intr[0]
            intr2_temp['kappa'] = v_intr[1]
            intr2_temp['sx'] = v_intr[2]
            if kwargs['include_sy2']:
                intr2_temp['sy'] = v_intr[3]
                intr2_temp['cx'] = v_intr[4]
                intr2_temp['cy'] = v_intr[5]
            else:
                intr2_temp['cx'] = v_intr[3]
                intr2_temp['cy'] = v_intr[4]

        return _project_dataset_flat_bicamera(
            X1_EC_temp, T_B1B2_temp, E2_C2_temp, T_E1B1_list_temp, T_E2B2_list_temp,
            T_C2B_list_temp, T_C1B_list_temp, kwargs['obj_pts_list'], intr1_temp, intr2_temp
        )

    params_initial_flat = build_x_current(
        layout, kwargs['X1_EC_init'], kwargs['T_B1B2_init'], kwargs['E2_C2_init'],
        kwargs['T_E1B1_list_init'], kwargs['T_E2B2_list_init'], kwargs['T_C2B_list_init'],
        kwargs['T_C1B_list_init'], kwargs['intr1_init'], kwargs['intr2_init']
    )
    J_numerical = np.zeros_like(J_analytical)
    
    print("2. Calculating Numerical Jacobian (this may take a while)...")
    f_base = project_wrapper(params_initial_flat)
    for i in range(len(params_initial_flat)):
        params_plus_h = params_initial_flat.copy()
        params_plus_h[i] += h
        f_plus_h = project_wrapper(params_plus_h)
        J_numerical[:, i] = (f_plus_h - f_base) / h
        if (i+1) % 20 == 0: print(f"   ... processed {i+1}/{len(params_initial_flat)} columns")
    print("   ... Numerical Jacobian calculated.")

    # --- 3. 결과 비교 및 분석 ---
    print("\n" + " 3. Comparison Results ".center(80, "="))
    abs_error_matrix = np.abs(J_analytical - J_numerical)
    norm_J_num = np.linalg.norm(J_numerical)
    
    total_relative_error = np.linalg.norm(J_analytical - J_numerical) / norm_J_num if norm_J_num > 1e-9 else np.inf
    print(f"\nOverall Relative Error: {total_relative_error:.6e}")
    if total_relative_error < 1e-5: print("✅ SUCCESS: Analytical Jacobian seems correct!")
    else: print("❌ FAILURE: Analytical Jacobian has significant errors!")

    print("\n" + " Per-Parameter Block Analysis ".center(80, "-"))
    for name, col_slice in layout.items():
        if name == 'total' or col_slice.start == col_slice.stop: continue
        
        J_an_block, J_num_block = J_analytical[:, col_slice], J_numerical[:, col_slice]
        abs_err_block = abs_error_matrix[:, col_slice]
        norm_num = np.linalg.norm(J_num_block)
        
        rel_err = np.linalg.norm(J_an_block - J_num_block) / norm_num if norm_num > 1e-9 else np.inf
        status = "✅ OK" if rel_err < 1e-5 else "❌ ERROR"
        
        print(f"[{status}] Parameter Block '{name}':")
        print(f"  - Relative Error: {rel_err:.6e}")
        print(f"  - Max Absolute Error in block: {np.max(abs_err_block):.6e}")
    print("="*80)

def verify_jacobian_dual(
    h=1e-7,                # 수치 미분 step
    **kwargs               # run_optimization_with_vce_dual 과 동일 인자
):
    """
    단일카메라 Dual 체인용 자코비안 검증.
    - 입력 per-pose는 T_E2B2_list (E2→B2).
    - 투영/자코비안은 체인에서 B2E2 = inv(E2B2)를 on-the-fly로 사용.
    필요한 kwargs (대표):
      X1_EC_init, T_B1B2_init, E2_C2_init,
      T_E1B1_list_init, T_E2B2_list_init, T_C2B_list_init,
      obj_pts_list,
      intrinsics_init,
      estimate_x1ec, estimate_b1b2, estimate_e2c2,
      estimate_e1b1, estimate_b2e2, estimate_c2b,
      estimate_intrinsics, include_sy, is_scara_x1
    """
    print(" Jacobian Verification (Dual, E2B2 input) ".center(80, "="))

    # 필수 옵션/크기
    nr = len(kwargs['obj_pts_list'])
    is_scara_x1   = kwargs.get('is_scara_x1', False)
    include_sy    = kwargs.get('include_sy', False)

    # ---- 보조: inv(T) ----
    def _inv_T(T):
        R, t = T[:3,:3], T[:3,3]
        RT = R.T
        Tinvt = np.eye(4, dtype=float)
        Tinvt[:3,:3] = RT
        Tinvt[:3,3]  = -RT @ t
        return Tinvt

    # ---- 투영 (Dual, E2B2 입력; 체인에서 B2E2=inv(E2B2)) ----
    def _project_dataset_flat_dual_E2B2(
        X1_EC, T_B1B2, E2_C2, T_E1B1_list, T_E2B2_list, T_C2B_list,
        obj_pts_list, intr
    ):
        """B --(C2B_i)--> C2 --(E2C2)--> E2 --(B2E2=inv(E2B2)_i)--> B2 --(B1B2)--> B1 --(E1B1_i)--> E1 --(X1EC)--> C1"""
        c, kappa, sx, sy, cx, cy = intr['c'], intr['kappa'], intr['sx'], intr['sy'], intr['cx'], intr['cy']
        uv = []
        R_x1, t_x1   = X1_EC[:3,:3], X1_EC[:3,3]
        R_b12, t_b12 = T_B1B2[:3,:3], T_B1B2[:3,3]
        R_e2c2, t_e2c2 = E2_C2[:3,:3], E2_C2[:3,3]

        for i in range(nr):
            R_c2b, t_c2b   = T_C2B_list[i][:3,:3], T_C2B_list[i][:3,3]
            R_e1b1, t_e1b1 = T_E1B1_list[i][:3,:3], T_E1B1_list[i][:3,3]

            # B2E2 = inv(E2B2)
            R_e2b2, t_e2b2 = T_E2B2_list[i][:3,:3], T_E2B2_list[i][:3,3]
            R_b2e2 = R_e2b2.T
            t_b2e2 = -R_e2b2.T @ t_e2b2

            for pw in kwargs['obj_pts_list'][i]:
                s1 = R_c2b  @ pw + t_c2b
                s2 = R_e2c2 @ s1 + t_e2c2
                s3 = R_b2e2 @ s2 + t_b2e2
                s4 = R_b12  @ s3 + t_b12
                s5 = R_e1b1 @ s4 + t_e1b1
                pc = R_x1   @ s5 + t_x1
                u, v = _project_point_division(pc, c, kappa, sx, sy, cx, cy)
                uv.append(u); uv.append(v)
        return np.array(uv, dtype=float)

    # ---- 파라미터 벡터 레이아웃 (dual) ----
    def compute_layout():
        layout = {}; col = 0
        dim_x1 = 5 if is_scara_x1 else 6
        if kwargs['estimate_x1ec']: layout['x1ec'] = slice(col, col+dim_x1); col += dim_x1
        if kwargs['estimate_b1b2']: layout['b1b2'] = slice(col, col+6); col += 6
        if kwargs['estimate_e2c2']: layout['e2c2'] = slice(col, col+6); col += 6
        if kwargs['estimate_e1b1']: layout['e1b1'] = slice(col, col+6*nr); col += 6*nr
        if kwargs['estimate_b2e2']: layout['b2e2'] = slice(col, col+6*nr); col += 6*nr   # ← E2B2 상태 그대로
        if kwargs['estimate_c2b']:  layout['c2b']  = slice(col, col+6*nr); col += 6*nr
        if kwargs['estimate_intrinsics']:
            layout['intr'] = slice(col, col + (6 if include_sy else 5)); col += (6 if include_sy else 5)
        layout['total'] = col
        return layout

    def build_x_current(layout, X1_EC, T_B1B2, E2_C2, T_E1B1_list, T_E2B2_list, T_C2B_list, intr):
        vecs = []
        if kwargs['estimate_x1ec']:
            e6 = mat_to_vec6d(X1_EC); vecs.append(e6[:5] if is_scara_x1 else e6)
        if kwargs['estimate_b1b2']:
            vecs.append(mat_to_vec6d(T_B1B2))
        if kwargs['estimate_e2c2']:
            vecs.append(mat_to_vec6d(E2_C2))
        if kwargs['estimate_e1b1']:
            vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
        if kwargs['estimate_b2e2']:
            vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list]))  # ← E2B2 그대로
        if kwargs['estimate_c2b']:
            vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_C2B_list]))
        if kwargs['estimate_intrinsics']:
            if include_sy:
                vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['sy'], intr['cx'], intr['cy']], float))
            else:
                vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['cx'], intr['cy']], float))
        return np.concatenate(vecs) if vecs else np.zeros(0)

    # ---- 1) 해석 자코비안 ----
    J_analytical = calculate_analytical_jacobian_division_model_dual(
        X1_EC=kwargs['X1_EC_init'],
        T_B1B2=kwargs['T_B1B2_init'],
        E2_C2=kwargs['E2_C2_init'],
        T_E1B1_list=kwargs['T_E1B1_list_init'],
        T_E2B2_list=kwargs['T_E2B2_list_init'],    # ← E2B2 입력
        T_C2B_list=kwargs['T_C2B_list_init'],
        obj_pts_list=kwargs['obj_pts_list'],
        c=kwargs['intrinsics_init']['c'], kappa=kwargs['intrinsics_init']['kappa'],
        sx=kwargs['intrinsics_init']['sx'],
        sy=(kwargs['intrinsics_init']['sy'] if include_sy else kwargs['intrinsics_init']['sx']),
        cx=kwargs['intrinsics_init']['cx'], cy=kwargs['intrinsics_init']['cy'],
        estimate_x1ec=kwargs['estimate_x1ec'],
        estimate_e1b1=kwargs['estimate_e1b1'],
        estimate_b1b2=kwargs['estimate_b1b2'],
        estimate_b2e2=kwargs['estimate_b2e2'],     # ← inverse-rule 적용되어 있어야 함
        estimate_e2c2=kwargs['estimate_e2c2'],
        estimate_c2b=kwargs['estimate_c2b'],
        estimate_intrinsics=kwargs['estimate_intrinsics'],
        include_sy=include_sy,
        is_scara_x1=is_scara_x1,
        mat_to_vec6d=mat_to_vec6d
    )
    print("1. Analytical Jacobian calculated.")

    # ---- 2) 수치 자코비안 (forward diff) ----
    layout = compute_layout()

    # intr 준비
    intr0 = {
        'c': kwargs['intrinsics_init']['c'],
        'kappa': kwargs['intrinsics_init']['kappa'],
        'sx': kwargs['intrinsics_init']['sx'],
        'sy': (kwargs['intrinsics_init']['sy'] if include_sy else kwargs['intrinsics_init']['sx']),
        'cx': kwargs['intrinsics_init']['cx'],
        'cy': kwargs['intrinsics_init']['cy'],
        'include_sy': include_sy
    }

    params0 = build_x_current(
        layout,
        kwargs['X1_EC_init'], kwargs['T_B1B2_init'], kwargs['E2_C2_init'],
        kwargs['T_E1B1_list_init'], kwargs['T_E2B2_list_init'], kwargs['T_C2B_list_init'],
        intr0
    )

    def unpack_and_project(params_flat):
        X1 = kwargs['X1_EC_init']; B12 = kwargs['T_B1B2_init']; E2C2 = kwargs['E2_C2_init']
        E1B1_list = kwargs['T_E1B1_list_init']; E2B2_list = kwargs['T_E2B2_list_init']; C2B_list = kwargs['T_C2B_list_init']
        intr = dict(intr0)

        # 글로벌
        if kwargs['estimate_x1ec']:
            v = params_flat[layout['x1ec']]
            if is_scara_x1:
                base = mat_to_vec6d(X1); base[:5] = v; v6 = np.array([*v, base[5]])
                X1 = vec6d_to_mat(v6)
            else:
                X1 = vec6d_to_mat(v)
        if kwargs['estimate_b1b2']:
            B12 = vec6d_to_mat(params_flat[layout['b1b2']])
        if kwargs['estimate_e2c2']:
            E2C2 = vec6d_to_mat(params_flat[layout['e2c2']])

        # per-pose
        if kwargs['estimate_e1b1']:
            E1B1_list = [vec6d_to_mat(a) for a in params_flat[layout['e1b1']].reshape(nr,6)]
        if kwargs['estimate_b2e2']:
            E2B2_list = [vec6d_to_mat(a) for a in params_flat[layout['b2e2']].reshape(nr,6)]  # ← E2B2 상태
        if kwargs['estimate_c2b']:
            C2B_list = [vec6d_to_mat(a) for a in params_flat[layout['c2b']].reshape(nr,6)]

        # intr
        if kwargs['estimate_intrinsics']:
            iv = params_flat[layout['intr']]
            if include_sy:
                intr = {'c':iv[0], 'kappa':iv[1], 'sx':iv[2], 'sy':iv[3], 'cx':iv[4], 'cy':iv[5], 'include_sy':True}
            else:
                intr = {'c':iv[0], 'kappa':iv[1], 'sx':iv[2], 'sy':iv[2], 'cx':iv[3], 'cy':iv[4], 'include_sy':False}

        return _project_dataset_flat_dual_E2B2(
            X1, B12, E2C2, E1B1_list, E2B2_list, C2B_list, kwargs['obj_pts_list'], intr
        )

    print("2. Calculating Numerical Jacobian...")
    f0 = unpack_and_project(params0)
    J_num = np.zeros_like(J_analytical)
    for i in range(params0.size):
        p = params0.copy(); p[i] += h
        fp = unpack_and_project(p)
        J_num[:, i] = (fp - f0) / h
        if (i+1) % 20 == 0: print(f"   ... {i+1}/{params0.size} cols")

    # ---- 3) 비교 ----
    print("\n" + " 3. Comparison Results ".center(80, "="))
    norm_num = np.linalg.norm(J_num)
    rel_err_all = np.linalg.norm(J_analytical - J_num) / (norm_num if norm_num>1e-12 else 1.0)
    print(f"Overall Relative Error: {rel_err_all:.6e}")
    print("✅ SUCCESS" if rel_err_all < 1e-5 else "❌ FAILURE")

    print("\n" + " Per-Parameter Block Analysis ".center(80, "-"))
    for name, sl in layout.items():
        if name == 'total' or sl.start == sl.stop: continue
        Ja = J_analytical[:, sl]; Jn = J_num[:, sl]
        n = np.linalg.norm(Jn)
        rel = np.linalg.norm(Ja - Jn) / (n if n>1e-12 else 1.0)
        mx  = np.max(np.abs(Ja - Jn))
        print(f"[{'OK' if rel<1e-5 else 'ERR'}] {name:>8s}  rel={rel:.3e}, maxAbs={mx:.3e}")
    print("="*80)

# V2
def run_optimization_with_vce_dual_bicamera(
    # 모델/데이터
    model_type: str,   # 'division'만 지원 (polynomial은 경고만)
    # 전역 변환 (공유)
    X1_EC_init: np.ndarray,      # ^C1 T_E1
    T_B1B2_init: np.ndarray,     # ^B1 T_B2
    E2_C2_init: np.ndarray,      # ^E2 T_C2
    # per-pose 변환 (공유)
    T_E1B1_list_init: list,      # [^E1 T_B1]_i     (노이즈 가정 변수)
    T_E2B2_list_init: list,      # [^E2 T_B2]_i     (노이즈 가정 변수)  <-- 이름 변경
    # 보드 관측(카메라별 앵커)
    T_C2B_list_init: list,       # [^C2 T_B]_i  (cam1 블록)
    T_C1B_list_init: list,       # [^C1 T_B]_i  (cam2 블록)
    # 포인트/이미지 관측
    obj_pts_list: list,          # [(Ni,3)]_i
    img1_pts_list: list,         # cam1 [(Ni,2)]_i  (C2→...→C1 체인으로 투영)
    img2_pts_list: list,         # cam2 [(Ni,2)]_i  (C1→...→C2 역체인으로 투영)
    # 포즈 관측(옵션: fictitious obs, VCE용)  <-- obs도 E2B2로 받음
    T_E1B1_list_obs: list = None,
    T_E2B2_list_obs: list = None,
    T_C2B_list_obs: list = None,
    T_C1B_list_obs: list = None,
    # intrinsics (카메라별)
    intr1_init: dict = None,     # {'c','kappa','sx','sy','cx','cy','include_sy'}
    intr2_init: dict = None,     # {'c','kappa','sx','sy','cx','cy','include_sy'}
    # 노이즈 (초기 분산)
    sigma_image_px: float = 0.1,
    sigma_angle_deg: float = 0.1,
    sigma_trans_mm: float = 1.0,
    # 반복/LM
    max_vce_iter: int = 5,
    max_param_iter: int = 10,
    term_thresh: float = 1e-6,
    # 추정 플래그
    estimate_x1ec: bool = True,
    estimate_b1b2: bool = True,
    estimate_e2c2: bool = True,
    estimate_e1b1: bool = True,   # per-pose
    estimate_b2e2: bool = True,   # per-pose (슬라이스 이름은 b2e2로 유지하되 내용은 E2B2)
    estimate_c2b:  bool = False,  # per-pose
    estimate_c1b:  bool = False,  # per-pose
    estimate_intr1: bool = False,
    estimate_intr2: bool = False,
    include_sy1: bool = False,
    include_sy2: bool = False,
    is_scara_x1: bool = False,
    vce_log = None,      # 이벤트를 append할 리스트 (옵션)
    collect_vce_hist: bool = False,   # σ̂0² 히스토리 수집 여부 (옵션)
):
    """
    양카메라(같은 보드) 동시 최적화.
    - 상태/프라이어/VCE의 노이즈 가정 변수: T_E1B1_list, T_E2B2_list (그대로 유지)
    - 잔차/자코비안에서만 필요 구간 inverse 적용:
        * cam1 체인의 B2E2 자리는 (E2B2)^{-1}를 on-the-fly로 사용
        * cam2 역체인의 inv(B2E2) 자리는 E2B2를 그대로 사용
    """
    assert model_type in ['division', 'polynomial'], "지원 모델: 'division' 또는 'polynomial'(placeholder)"
    if model_type != 'division':
        print("[경고] 듀얼-바이카메라 솔버는 division 모델만 구현되었습니다.")

    nr = len(obj_pts_list)
    assert len(img1_pts_list) == nr and len(img2_pts_list) == nr
    assert len(T_C2B_list_init) == nr and len(T_C1B_list_init) == nr
    assert len(T_E1B1_list_init) == nr and len(T_E2B2_list_init) == nr

    # --- 상태 변수 초기화 (노이즈 프레임 유지) ---
    X1_EC   = X1_EC_init.copy()
    T_B1B2  = T_B1B2_init.copy()
    E2_C2   = E2_C2_init.copy()
    T_E1B1_list = [T.copy() for T in T_E1B1_list_init]
    T_E2B2_list = [T.copy() for T in T_E2B2_list_init]   # <-- 상태는 E2B2로 유지
    T_C2B_list  = [T.copy() for T in T_C2B_list_init]
    T_C1B_list  = [T.copy() for T in T_C1B_list_init]

    # --- intrinsics 상태 ---
    if intr1_init is None:
        intr1_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=(1.0 if include_sy1 else 1.0), cx=0.0, cy=0.0, include_sy=include_sy1)
    if intr2_init is None:
        intr2_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=(1.0 if include_sy2 else 1.0), cx=0.0, cy=0.0, include_sy=include_sy2)

    intr1 = {'c':intr1_init['c'],'kappa':intr1_init['kappa'],'sx':intr1_init['sx'],
             'sy':intr1_init['sy'],'cx':intr1_init['cx'],'cy':intr1_init['cy'],'include_sy':include_sy1}
    intr2 = {'c':intr2_init['c'],'kappa':intr2_init['kappa'],'sx':intr2_init['sx'],
             'sy':intr2_init['sy'],'cx':intr2_init['cx'],'cy':intr2_init['cy'],'include_sy':include_sy2}

    # --- 보조: 역변환 ---
    def _inv_T(T):
        R, t = T[:3,:3], T[:3,3]
        RT = R.T
        Tinvt = np.eye(4, dtype=float)
        Tinvt[:3,:3] = RT
        Tinvt[:3,3]  = -RT @ t
        return Tinvt

    # --- 프로젝션(픽셀) 유틸: cam1(정방향 체인), cam2(역체인) ---
    def _project_dataset_flat_cam1(X1_EC_, T_B1B2_, E2_C2_,
                                   T_E1B1_list_, T_E2B2_list_, T_C2B_list_,
                                   obj_pts_list_, intr_):
        """C2→E2→B2→B1→E1→C1 체인으로 cam1 픽셀.
           B2→E2 자리는 (E2B2)^{-1}를 on-the-fly로 사용."""
        c, kappa, sx, sy, cx, cy = intr_['c'], intr_['kappa'], intr_['sx'], intr_['sy'], intr_['cx'], intr_['cy']
        out = []
        R_x1, t_x1 = X1_EC_[:3,:3], X1_EC_[:3,3]
        R_b12,t_b12= T_B1B2_[:3,:3], T_B1B2_[:3,3]
        R_e2c2,t_e2c2=E2_C2_[:3,:3], E2_C2_[:3,3]
        for i in range(nr):
            R_c2b,t_c2b   = T_C2B_list_[i][:3,:3], T_C2B_list_[i][:3,3]
            R_e1b1,t_e1b1 = T_E1B1_list_[i][:3,:3], T_E1B1_list_[i][:3,3]
            # (E2B2)^{-1} = B2E2
            T_b2e2_i      = _inv_T(T_E2B2_list_[i])
            R_b2e2,t_b2e2 = T_b2e2_i[:3,:3], T_b2e2_i[:3,3]
            for pw in obj_pts_list_[i]:
                s1 = R_c2b @ pw + t_c2b
                s2 = R_e2c2 @ s1 + t_e2c2
                s3 = R_b2e2 @ s2 + t_b2e2
                s4 = R_b12  @ s3 + t_b12
                s5 = R_e1b1 @ s4 + t_e1b1
                pc = R_x1   @ s5 + t_x1
                u,v = _project_point_division(pc, c, kappa, sx, sy, cx, cy)
                out.extend((u,v))
        return np.array(out, float)

    def _project_dataset_flat_cam2(X1_EC_, T_B1B2_, E2_C2_,
                                   T_E1B1_list_, T_E2B2_list_, T_C1B_list_,
                                   obj_pts_list_, intr_):
        """B→C1→E1(inv X1_EC)→B1(inv E1B1)→B2(inv B1B2)→E2(inv B2E2)→C2(inv E2_C2)
           여기서 inv(B2E2)=E2B2이므로, 해당 단계는 E2B2를 정방향으로 적용."""
        c, kappa, sx, sy, cx, cy = intr_['c'], intr_['kappa'], intr_['sx'], intr_['sy'], intr_['cx'], intr_['cy']
        out = []
        R_x1, t_x1 = X1_EC_[:3,:3], X1_EC_[:3,3]
        R_b12,t_b12= T_B1B2_[:3,:3], T_B1B2_[:3,3]
        R_e2c2,t_e2c2=E2_C2_[:3,:3], E2_C2_[:3,3]
        RT_x1 = R_x1.T; RT_b12 = R_b12.T; RT_e2c2 = R_e2c2.T
        for i in range(nr):
            R_c1b,t_c1b   = T_C1B_list_[i][:3,:3], T_C1B_list_[i][:3,3]
            R_e1b1,t_e1b1 = T_E1B1_list_[i][:3,:3], T_E1B1_list_[i][:3,3]
            R_e2b2,t_e2b2 = T_E2B2_list_[i][:3,:3], T_E2B2_list_[i][:3,3]  # 정방향
            RT_e1b1 = R_e1b1.T
            for pw in obj_pts_list_[i]:
                q1 = R_c1b @ pw + t_c1b
                q2 = RT_x1   @ (q1 - t_x1)
                q3 = RT_e1b1 @ (q2 - t_e1b1)
                q4 = RT_b12  @ (q3 - t_b12)
                q5 = R_e2b2  @ q4 + t_e2b2        # inv(B2E2) = E2B2 (정방향)
                pc = RT_e2c2 @ (q5 - t_e2c2)
                u,v = _project_point_division(pc, c, kappa, sx, sy, cx, cy)
                out.extend((u,v))
        return np.array(out, float)

    def _project_dataset_flat_bicamera(X1_EC_, T_B1B2_, E2_C2_,
                                       T_E1B1_list_, T_E2B2_list_,
                                       T_C2B_list_, T_C1B_list_,
                                       obj_pts_list_, intr1_, intr2_):
        f1 = _project_dataset_flat_cam1(X1_EC_, T_B1B2_, E2_C2_,
                                        T_E1B1_list_, T_E2B2_list_, T_C2B_list_,
                                        obj_pts_list_, intr1_)
        f2 = _project_dataset_flat_cam2(X1_EC_, T_B1B2_, E2_C2_,
                                        T_E1B1_list_, T_E2B2_list_, T_C1B_list_,
                                        obj_pts_list_, intr2_)
        return np.concatenate([f1, f2])

    # --- 관측 벡터 (cam1→cam2 순서) ---
    l_obs_img = np.concatenate([pts.reshape(-1) for pts in img1_pts_list] +
                               [pts.reshape(-1) for pts in img2_pts_list])
    ni_img = len(l_obs_img)

    # --- 포즈 관측 블록 (노이즈 프레임: E1B1, E2B2) ---
    obs_blocks = []
    if estimate_e1b1 and (T_E1B1_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]))
    if estimate_b2e2 and (T_E2B2_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs]))  # ← E2B2 직접 비교
    if estimate_c2b and (T_C2B_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs]))
    if estimate_c1b and (T_C1B_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_C1B_list_obs]))
    l_obs = np.concatenate([l_obs_img, *obs_blocks]) if len(obs_blocks)>0 else l_obs_img

    # --- 초기 분산/LM 파라미터 ---
    VAR_I_FLOOR, VAR_A_FLOOR, VAR_T_FLOOR = 1e-12, (np.deg2rad(1e-9))**2, (1e-9)**2
    var_i = max(sigma_image_px**2, VAR_I_FLOOR)
    var_a = max(np.deg2rad(sigma_angle_deg)**2, VAR_A_FLOOR)
    var_t = max((sigma_trans_mm / 1000.0)**2, VAR_T_FLOOR)
    lam, lam_up, lam_down, lam_min, lam_max, max_ls_tries = 1e-2, 5.0, 0.25, 1e-12, 1e+8, 8
    # --- VCE gain control (클리핑/스무딩) ---
    VCE_GAIN_MIN = 0.2   # 최소 배율
    VCE_GAIN_MAX = 5.0   # 최대 배율
    VCE_ETA      = 0.5   # 스무딩(0~1): 1.0=그대로, 0.5=sqrt(gain)
    MIN_REDUNDANCY = 10.0  # 자유도 너무 작으면 업데이트 스킵

    # --- 레이아웃 (키 이름은 호환을 위해 'b2e2' 유지, 내용은 E2B2) ---
    def compute_layout():
        layout = {}; col = 0
        dim_x1 = 5 if is_scara_x1 else 6
        if estimate_x1ec: layout['x1ec'] = slice(col, col+dim_x1); col += dim_x1
        if estimate_b1b2: layout['b1b2'] = slice(col, col+6); col += 6
        if estimate_e2c2: layout['e2c2'] = slice(col, col+6); col += 6
        if estimate_e1b1: layout['e1b1'] = slice(col, col+6*nr); col += 6*nr
        if estimate_b2e2: layout['b2e2'] = slice(col, col+6*nr); col += 6*nr   # ← 여기에 E2B2가 들어감
        if estimate_c2b:  layout['c2b']  = slice(col, col+6*nr); col += 6*nr
        if estimate_c1b:  layout['c1b']  = slice(col, col+6*nr); col += 6*nr
        if estimate_intr1: layout['intr1'] = slice(col, col+(6 if include_sy1 else 5)); col += (6 if include_sy1 else 5)
        if estimate_intr2: layout['intr2'] = slice(col, col+(6 if include_sy2 else 5)); col += (6 if include_sy2 else 5)
        layout['total'] = col
        return layout

    def build_x_current(layout):
        vecs = []
        if estimate_x1ec:
            e6 = mat_to_vec6d(X1_EC); vecs.append(e6[:5] if is_scara_x1 else e6)
        if estimate_b1b2: vecs.append(mat_to_vec6d(T_B1B2))
        if estimate_e2c2: vecs.append(mat_to_vec6d(E2_C2))
        if estimate_e1b1: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
        if estimate_b2e2: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list]))  # ← E2B2
        if estimate_c2b:  vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_C2B_list]))
        if estimate_c1b:  vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_C1B_list]))
        if estimate_intr1:
            if include_sy1:
                vecs.append(np.array([intr1['c'], intr1['kappa'], intr1['sx'], intr1['sy'], intr1['cx'], intr1['cy']], float))
            else:
                vecs.append(np.array([intr1['c'], intr1['kappa'], intr1['sx'], intr1['cx'], intr1['cy']], float))
        if estimate_intr2:
            if include_sy2:
                vecs.append(np.array([intr2['c'], intr2['kappa'], intr2['sx'], intr2['sy'], intr2['cx'], intr2['cy']], float))
            else:
                vecs.append(np.array([intr2['c'], intr2['kappa'], intr2['sx'], intr2['cx'], intr2['cy']], float))
        return np.concatenate(vecs) if len(vecs)>0 else np.zeros(0)

    print("="*50); print("Dual-Arm Division Optimization Start (Bi-Camera)"); print("="*50)

    for vce_iter in range(max_vce_iter):
        print(f"\n--- VCE Iteration {vce_iter+1}/{max_vce_iter} ---")
        print(f"Variances: σ_img²={var_i:.3e}, σ_ang²={var_a:.3e}, σ_trans²={var_t:.3e}")

        # 관측 가중치 대각
        Pll_diag_list = [np.full(ni_img, 1.0/var_i)]
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            Pll_diag_list.append(np.tile(np.concatenate([np.full(3,1.0/var_a), np.full(3,1.0/var_t)]), nr))
        if estimate_b2e2 and (T_E2B2_list_obs is not None):
            Pll_diag_list.append(np.tile(np.concatenate([np.full(3,1.0/var_a), np.full(3,1.0/var_t)]), nr))  # E2B2
        if estimate_c2b and (T_C2B_list_obs is not None):
            Pll_diag_list.append(np.tile(np.concatenate([np.full(3,1.0/var_a), np.full(3,1.0/var_t)]), nr))
        if estimate_c1b and (T_C1B_list_obs is not None):
            Pll_diag_list.append(np.tile(np.concatenate([np.full(3,1.0/var_a), np.full(3,1.0/var_t)]), nr))
        Pll_diag = np.concatenate(Pll_diag_list) if len(Pll_diag_list)>1 else Pll_diag_list[0]

        layout = compute_layout()
        phi_best = np.inf

        # ------------ LM 내부 반복 ------------
        for param_iter in range(max_param_iter):
            x_k = build_x_current(layout)

            # (A) 예측 픽셀 (cam1+cam2 스택)
            f_pix = _project_dataset_flat_bicamera(
                X1_EC, T_B1B2, E2_C2,
                T_E1B1_list, T_E2B2_list,     # ← E2B2
                T_C2B_list, T_C1B_list,
                obj_pts_list, intr1, intr2
            )

            # (B) 잔차 (이미지 + 포즈 프라이어)
            w_list = [l_obs_img - f_pix]
            if estimate_e1b1 and (T_E1B1_list_obs is not None):
                f_e1b1 = x_k[layout['e1b1']] if estimate_e1b1 else np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list])
                obs_e1b1 = np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs])
                w_list.append(obs_e1b1 - f_e1b1)
            if estimate_b2e2 and (T_E2B2_list_obs is not None):
                f_e2b2 = x_k[layout['b2e2']] if estimate_b2e2 else np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list])
                obs_e2b2 = np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs])
                w_list.append(obs_e2b2 - f_e2b2)   # ← E2B2 직접 비교
            if estimate_c2b and (T_C2B_list_obs is not None):
                f_c2b = x_k[layout['c2b']] if estimate_c2b else np.concatenate([mat_to_vec6d(T) for T in T_C2B_list])
                obs_c2b = np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs])
                w_list.append(obs_c2b - f_c2b)
            if estimate_c1b and (T_C1B_list_obs is not None):
                f_c1b = x_k[layout['c1b']] if estimate_c1b else np.concatenate([mat_to_vec6d(T) for T in T_C1B_list])
                obs_c1b = np.concatenate([mat_to_vec6d(T) for T in T_C1B_list_obs])
                w_list.append(obs_c1b - f_c1b)
            w = np.concatenate(w_list)

            if np.isnan(w).any():
                print(f"  [LM-{param_iter+1}] NaN residual detected. Stop.")
                break

            # (C) 해석 자코비안 (cam1+cam2 세로 스택)
            #  └ 자코비안 구현은 E2B2 파라미터화를 지원해야 함:
            #     - cam1 블록의 E2B2: inverse rule
            #     - cam2 블록의 E2B2: forward rule
            A_img, layout_chk = calculate_analytical_jacobian_division_model_dual_bicamera(
                X1_EC=X1_EC, T_B1B2=T_B1B2, E2_C2=E2_C2,
                T_E1B1_list=T_E1B1_list, T_E2B2_list=T_E2B2_list,   # ← 인자명/내부 처리 업데이트 필요
                T_C2B_list=T_C2B_list, T_C1B_list=T_C1B_list,
                obj_pts_list=obj_pts_list,
                c1=intr1['c'], kappa1=intr1['kappa'], sx1=intr1['sx'], sy1=intr1['sy'], cx1=intr1['cx'], cy1=intr1['cy'],
                c2=intr2['c'], kappa2=intr2['kappa'], sx2=intr2['sx'], sy2=intr2['sy'], cx2=intr2['cx'], cy2=intr2['cy'],
                estimate_x1ec=estimate_x1ec, estimate_b1b2=estimate_b1b2, estimate_e2c2=estimate_e2c2,
                estimate_e1b1=estimate_e1b1, estimate_b2e2=estimate_b2e2,   # 키는 유지(호환)
                estimate_c2b=estimate_c2b, estimate_c1b=estimate_c1b,
                estimate_intr1=estimate_intr1, estimate_intr2=estimate_intr2,
                include_sy1=include_sy1, include_sy2=include_sy2,
                is_scara_x1=is_scara_x1,
                mat_to_vec6d=mat_to_vec6d,
            )

            # (D) 전체 A (이미지 + 포즈 항 I)
            A = np.zeros((w.size, layout['total']), float)
            A[:ni_img, :] = A_img
            row_ptr = ni_img
            if estimate_e1b1 and (T_E1B1_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
            if estimate_b2e2 and (T_E2B2_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr  # ← E2B2
            if estimate_c2b and (T_C2B_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['c2b']] = np.eye(6*nr); row_ptr += 6*nr
            if estimate_c1b and (T_C1B_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['c1b']] = np.eye(6*nr); row_ptr += 6*nr

            # (E) LM step
            phi0 = float(np.sum(Pll_diag * (w**2)))
            N = A.T @ (Pll_diag[:,None]*A)
            b = A.T @ (Pll_diag*w)

            success = False
            for ls_try in range(max_ls_tries):
                try:
                    N_aug = N + lam * np.diag(np.diag(N))
                    delta = np.linalg.solve(N_aug, b)
                except np.linalg.LinAlgError:
                    lam = min(lam * lam_up, lam_max); continue

                x_new = x_k + delta
                if not np.isfinite(x_new).all():
                    lam = min(lam * lam_up, lam_max); continue

                # (E-1) 파라미터 갱신 (E2B2를 상태로 유지)
                try:
                    X1_EC_new = X1_EC
                    if estimate_x1ec:
                        v = x_new[layout['x1ec']]
                        if is_scara_x1:
                            base = mat_to_vec6d(X1_EC); base[:5] = v; v6 = np.array([*v, base[5]])
                            X1_EC_new = vec6d_to_mat(v6)
                        else:
                            X1_EC_new = vec6d_to_mat(v)
                    T_B1B2_new = vec6d_to_mat(x_new[layout['b1b2']]) if estimate_b1b2 else T_B1B2
                    E2_C2_new  = vec6d_to_mat(x_new[layout['e2c2']])  if estimate_e2c2  else E2_C2
                    T_E1B1_list_new = T_E1B1_list if not estimate_e1b1 else [vec6d_to_mat(a) for a in x_new[layout['e1b1']].reshape(nr,6)]
                    T_E2B2_list_new = T_E2B2_list if not estimate_b2e2 else [vec6d_to_mat(a) for a in x_new[layout['b2e2']].reshape(nr,6)]  # ← E2B2
                    T_C2B_list_new  = T_C2B_list  if not estimate_c2b  else [vec6d_to_mat(a) for a in x_new[layout['c2b'] ].reshape(nr,6)]
                    T_C1B_list_new  = T_C1B_list  if not estimate_c1b  else [vec6d_to_mat(a) for a in x_new[layout['c1b'] ].reshape(nr,6)]

                    intr1_new = intr1
                    if estimate_intr1:
                        iv = x_new[layout['intr1']]
                        if include_sy1:
                            intr1_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':iv[3],'cx':iv[4],'cy':iv[5],'include_sy':True}
                        else:
                            intr1_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':intr1['sy'],'cx':iv[3],'cy':iv[4],'include_sy':False}
                    intr2_new = intr2
                    if estimate_intr2:
                        iv = x_new[layout['intr2']]
                        if include_sy2:
                            intr2_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':iv[3],'cx':iv[4],'cy':iv[5],'include_sy':True}
                        else:
                            intr2_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':intr2['sy'],'cx':iv[3],'cy':iv[4],'include_sy':False}
                except Exception:
                    lam = min(lam * lam_up, lam_max); continue

                # (E-2) 새 잔차 평가
                f_pix_new = _project_dataset_flat_bicamera(
                    X1_EC_new, T_B1B2_new, E2_C2_new,
                    T_E1B1_list_new, T_E2B2_list_new,   # ← E2B2
                    T_C2B_list_new, T_C1B_list_new,
                    obj_pts_list, intr1_new, intr2_new
                )
                w_list_new = [l_obs_img - f_pix_new]
                if estimate_e1b1 and (T_E1B1_list_obs is not None):
                    obs_e1b1 = np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs])
                    w_list_new.append(obs_e1b1 - x_new[layout['e1b1']])
                if estimate_b2e2 and (T_E2B2_list_obs is not None):
                    obs_e2b2 = np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs])
                    w_list_new.append(obs_e2b2 - x_new[layout['b2e2']])   # ← E2B2
                if estimate_c2b and (T_C2B_list_obs is not None):
                    obs_c2b = np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs])
                    w_list_new.append(obs_c2b - x_new[layout['c2b']])
                if estimate_c1b and (T_C1B_list_obs is not None):
                    obs_c1b = np.concatenate([mat_to_vec6d(T) for T in T_C1B_list_obs])
                    w_list_new.append(obs_c1b - x_new[layout['c1b']])

                w_new = np.concatenate(w_list_new)
                if np.isnan(w_new).any():
                    lam = min(lam * lam_up, lam_max); continue
                phi_new = float(np.sum(Pll_diag * (w_new**2)))

                if phi_new < phi0:
                    X1_EC, T_B1B2, E2_C2 = X1_EC_new, T_B1B2_new, E2_C2_new
                    T_E1B1_list, T_E2B2_list = T_E1B1_list_new, T_E2B2_list_new
                    T_C2B_list, T_C1B_list   = T_C2B_list_new, T_C1B_list_new
                    intr1, intr2 = intr1_new, intr2_new
                    lam = max(lam * lam_down, lam_min)
                    phi_best = phi_new
                    success = True
                    print(f"    - LM {param_iter+1} / try {ls_try+1}: φ {phi0:.6f} → {phi_new:.6f}, λ={lam:.2e}")
                    break
                else:
                    lam = min(lam * lam_up, lam_max)

            if not success:
                print(f"  [LM-{param_iter+1}] No improvement after {max_ls_tries} tries.")
                break
            if abs(phi0 - phi_best) < term_thresh:
                print("  [LM] Converged. Stop parameter search.")
                break

        # ------------ (F) VCE 업데이트 ------------
        print("  Updating variance components...")
        f_pix_final = _project_dataset_flat_bicamera(
            X1_EC, T_B1B2, E2_C2,
            T_E1B1_list, T_E2B2_list,  # ← E2B2
            T_C2B_list, T_C1B_list,
            obj_pts_list, intr1, intr2
        )
        v_list = [l_obs_img - f_pix_final]
        x_now = build_x_current(layout)
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            obs_e1b1 = np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs])
            v_list.append(obs_e1b1 - x_now[layout['e1b1']])
        if estimate_b2e2 and (T_E2B2_list_obs is not None):
            obs_e2b2 = np.concatenate([mat_to_vec6d(T) for T in T_E2B2_list_obs])
            v_list.append(obs_e2b2 - x_now[layout['b2e2']])  # ← E2B2
        if estimate_c2b and (T_C2B_list_obs is not None):
            obs_c2b = np.concatenate([mat_to_vec6d(T) for T in T_C2B_list_obs])
            v_list.append(obs_c2b - x_now[layout['c2b']])
        if estimate_c1b and (T_C1B_list_obs is not None):
            obs_c1b = np.concatenate([mat_to_vec6d(T) for T in T_C1B_list_obs])
            v_list.append(obs_c1b - x_now[layout['c1b']])
        v_hat = np.concatenate(v_list)

        # 자코비안(최종 파라미터) — 동일 플래그로 재호출
        A_img, _ = calculate_analytical_jacobian_division_model_dual_bicamera(
            X1_EC=X1_EC, T_B1B2=T_B1B2, E2_C2=E2_C2,
            T_E1B1_list=T_E1B1_list, T_E2B2_list=T_E2B2_list,   # ← E2B2
            T_C2B_list=T_C2B_list, T_C1B_list=T_C1B_list,
            obj_pts_list=obj_pts_list,
            c1=intr1['c'], kappa1=intr1['kappa'], sx1=intr1['sx'], sy1=intr1['sy'], cx1=intr1['cx'], cy1=intr1['cy'],
            c2=intr2['c'], kappa2=intr2['kappa'], sx2=intr2['sx'], sy2=intr2['sy'], cx2=intr2['cx'], cy2=intr2['cy'],
            estimate_x1ec=estimate_x1ec, estimate_b1b2=estimate_b1b2, estimate_e2c2=estimate_e2c2,
            estimate_e1b1=estimate_e1b1, estimate_b2e2=estimate_b2e2,
            estimate_c2b=estimate_c2b, estimate_c1b=estimate_c1b,
            estimate_intr1=estimate_intr1, estimate_intr2=estimate_intr2,
            include_sy1=include_sy1, include_sy2=include_sy2,
            is_scara_x1=is_scara_x1,
            mat_to_vec6d=mat_to_vec6d,
        )
        A = np.zeros((v_hat.size, layout['total']), float)
        A[:ni_img, :] = A_img
        row_ptr = ni_img
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
        if estimate_b2e2 and (T_E2B2_list_obs is not None):
            A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr  # ← E2B2
        if estimate_c2b and (T_C2B_list_obs is not None):
            A[row_ptr:row_ptr+6*nr, layout['c2b']] = np.eye(6*nr); row_ptr += 6*nr
        if estimate_c1b and (T_C1B_list_obs is not None):
            A[row_ptr:row_ptr+6*nr, layout['c1b']] = np.eye(6*nr); row_ptr += 6*nr

        # N = A^T P A, redundancy
        N_mat = A.T @ (Pll_diag[:,None]*A)
        try:
            cF, lower = cho_factor(N_mat, check_finite=False)
            diag_Qlhat = np.empty(v_hat.size, float)
            for i in range(v_hat.size):
                ai = A[i,:]
                y = cho_solve((cF, lower), ai, check_finite=False)
                diag_Qlhat[i] = float(ai @ y)
        except Exception:
            Q_xx = np.linalg.pinv(N_mat)
            diag_Qlhat = np.einsum('ij,jk,ik->i', A, Q_xx, A)

        diag_R = 1.0 - Pll_diag * diag_Qlhat

        # --- 그룹별 σ̂0² (이미지 하나의 그룹 + 포즈 각/이동 묶음) ---
        R_img = float(np.sum(diag_R[:ni_img]))
        v_img = v_hat[:ni_img]
        sigma0_sq_img = float(v_img.T @ (Pll_diag[:ni_img] * v_img) / max(R_img, 1e-9))

        idx_ang_all, idx_trans_all = [], []
        offset = ni_img
        def _collect_pose_indices(enabled):
            nonlocal offset
            if not enabled: return
            base = offset
            for p in range(nr):
                b = base + 6*p
                idx_ang_all.extend([b+0, b+1, b+2]); idx_trans_all.extend([b+3, b+4, b+5])
            offset += 6*nr
        _collect_pose_indices(estimate_e1b1 and (T_E1B1_list_obs is not None))
        _collect_pose_indices(estimate_b2e2 and (T_E2B2_list_obs is not None))
        _collect_pose_indices(estimate_c2b  and (T_C2B_list_obs  is not None))
        _collect_pose_indices(estimate_c1b  and (T_C1B_list_obs  is not None))

        idx_ang = np.array(idx_ang_all, int); idx_trn = np.array(idx_trans_all, int)
        sigma0_sq_ang = float((v_hat[idx_ang].T @ (Pll_diag[idx_ang]*v_hat[idx_ang])) / max(np.sum(diag_R[idx_ang]), 1e-9)) if idx_ang.size else None
        sigma0_sq_trn = float((v_hat[idx_trn].T @ (Pll_diag[idx_trn]*v_hat[idx_trn])) / max(np.sum(diag_R[idx_trn]), 1e-9)) if idx_trn.size else None

        print("  σ̂0² (should → 1): "
              f"img={sigma0_sq_img:.4f}"
              + (f", ang={sigma0_sq_ang:.4f}" if sigma0_sq_ang is not None else "")
              + (f", trans={sigma0_sq_trn:.4f}" if sigma0_sq_trn is not None else ""))

        if vce_iter == 0:
            _have_log = vce_log is not None
            # 간단 카운터
            clip_counts = {'img': {'under':0,'over':0},
                        'ang': {'under':0,'over':0},
                        'trans': {'under':0,'over':0}}
            # 히스토리(원하면)
            if collect_vce_hist:
                sigma0_hist = {'img': [], 'ang': [], 'trans': []}

        # ──(B) 이번 반복의 σ̂0² 기록(선택)──
        if collect_vce_hist:
            sigma0_hist['img'].append(float(sigma0_sq_img))
            if sigma0_sq_ang is not None:  sigma0_hist['ang'].append(float(sigma0_sq_ang))
            if sigma0_sq_trn is not None:  sigma0_hist['trans'].append(float(sigma0_sq_trn))

        # ──(C) 클리핑 초과 감지 & 이벤트 로그──
        def _check_clip_and_log(group_name: str, raw_val: float):
            if not np.isfinite(raw_val):  # NaN/Inf는 따로 표기
                if vce_log is not None:
                    vce_log.append({
                        'iter': int(vce_iter),
                        'group': group_name,
                        'raw_sigma0_sq': None,
                        'status': 'nan_or_inf'
                    })
                return
            under = (raw_val < VCE_GAIN_MIN)
            over  = (raw_val > VCE_GAIN_MAX)
            if under or over:
                clip_counts[group_name]['under'] += int(under)
                clip_counts[group_name]['over']  += int(over)
                if vce_log is not None:
                    vce_log.append({
                        'iter': int(vce_iter),
                        'group': group_name,
                        'raw_sigma0_sq': float(raw_val),
                        'clip_min': float(VCE_GAIN_MIN),
                        'clip_max': float(VCE_GAIN_MAX),
                        'status': 'under' if under else 'over'
                    })
                print(f"  [VCE-clip] iter={vce_iter} group={group_name} "
                    f"raw={raw_val:.4f} → clipped to [{VCE_GAIN_MIN}, {VCE_GAIN_MAX}]")

        # 세 그룹에 대해 실행
        _check_clip_and_log('img',   sigma0_sq_img)
        if sigma0_sq_ang  is not None: _check_clip_and_log('ang',   sigma0_sq_ang)
        if sigma0_sq_trn  is not None: _check_clip_and_log('trans', sigma0_sq_trn)
        
        if (_is_close_to_one(sigma0_sq_img) and
            _is_close_to_one(sigma0_sq_ang) and
            _is_close_to_one(sigma0_sq_trn)):
            print("  ✅ VCE converged: all σ̂0² ≈ 1.0. Stopping.")
            break

        if R_img < MIN_REDUNDANCY:
            sigma0_sq_img = 1.0  # 이미지 그룹은 이번 회 차 업데이트 스킵 의미
        if idx_ang.size and np.sum(diag_R[idx_ang]) < MIN_REDUNDANCY:
            sigma0_sq_ang = None
        if idx_trn.size and np.sum(diag_R[idx_trn]) < MIN_REDUNDANCY:
            sigma0_sq_trn = None

        def _clipped_gain(g, gmin, gmax, eta):
            if not np.isfinite(g):
                return 1.0  # NaN/Inf면 업데이트 스킵
            g_clip = float(np.clip(g, gmin, gmax))  # 클리핑
            return g_clip  # 스무딩(eta=1 → 그대로, 0.5 → sqrt)

        old_i, old_a, old_t = var_i, var_a, var_t

        # img
        g_img = _clipped_gain(sigma0_sq_img, VCE_GAIN_MIN, VCE_GAIN_MAX, VCE_ETA)
        var_i = max(var_i * g_img, VAR_I_FLOOR)

        # ang
        if sigma0_sq_ang is not None:
            g_ang = _clipped_gain(sigma0_sq_ang, VCE_GAIN_MIN, VCE_GAIN_MAX, VCE_ETA)
            var_a = max(var_a * g_ang, VAR_A_FLOOR)

        # trans
        if sigma0_sq_trn is not None:
            g_trn = _clipped_gain(sigma0_sq_trn, VCE_GAIN_MIN, VCE_GAIN_MAX, VCE_ETA)
            var_t = max(var_t * g_trn, VAR_T_FLOOR)

        print(f"  σ² update (clipped): img {old_i:.3e}→{var_i:.3e}"
            + (f" | ang {old_a:.3e}→{var_a:.3e}" if sigma0_sq_ang is not None else "")
            + (f" | trans {old_t:.3e}→{var_t:.3e}" if sigma0_sq_trn is not None else ""))
        
        print("  σ̂0² (raw): "
            f"img={sigma0_sq_img:.4f}"
            + (f", ang={sigma0_sq_ang:.4f}" if sigma0_sq_ang is not None else "")
            + (f", trans={sigma0_sq_trn:.4f}" if sigma0_sq_trn is not None else ""))
        print(f"  gains(clipped^η): img={g_img:.3f}"
            + (f", ang={g_ang:.3f}" if sigma0_sq_ang is not None else "")
            + (f", trans={g_trn:.3f}" if sigma0_sq_trn is not None else ""))

        # # --- 분산 업데이트 (Eq.24) ---
        # old_i, old_a, old_t = var_i, var_a, var_t
        # var_i = max(var_i * sigma0_sq_img, VAR_I_FLOOR)
        # if sigma0_sq_ang is not None: var_a = max(var_a * sigma0_sq_ang, VAR_A_FLOOR)
        # if sigma0_sq_trn is not None: var_t = max(var_t * sigma0_sq_trn, VAR_T_FLOOR)
        # print(f"  σ² update: img {old_i:.3e}→{var_i:.3e}"
        #       + (f" | ang {old_a:.3e}→{var_a:.3e}" if sigma0_sq_ang is not None else "")
        #       + (f" | trans {old_t:.3e}→{var_t:.3e}" if sigma0_sq_trn is not None else ""))

    # 결과 intrinsics dict (카메라별 반환)
    intr1_final = {'c':intr1['c'],'kappa':intr1['kappa'],'sx':intr1['sx'],
                   'sy':intr1['sy'],'cx':intr1['cx'],'cy':intr1['cy'],'include_sy':include_sy1}
    intr2_final = {'c':intr2['c'],'kappa':intr2['kappa'],'sx':intr2['sx'],
                   'sy':intr2['sy'],'cx':intr2['cx'],'cy':intr2['cy'],'include_sy':include_sy2}

    if 'clip_counts' in locals():
        summary = (f"[VCE-clip-summary] img(u/o)={clip_counts['img']['under']}/{clip_counts['img']['over']}, "
                f"ang(u/o)={clip_counts['ang']['under']}/{clip_counts['ang']['over']}, "
                f"trans(u/o)={clip_counts['trans']['under']}/{clip_counts['trans']['over']}")
        print(summary)
        if vce_log is not None:
            vce_log.append({'summary': summary})
        if collect_vce_hist and vce_log is not None:
            vce_log.append({'sigma0_hist': sigma0_hist})

    print("\nDual-Arm Division Optimization Finished (Bi-Camera).")
    return (X1_EC, T_B1B2, E2_C2,
            T_E1B1_list, T_E2B2_list,    # ← E2B2로 반환
            T_C2B_list, T_C1B_list,
            intr1_final, intr2_final)


# --- FD 비교용: 파라미터 벡터 팩/언팩 -----------------------------

def pack_params_dual(X1_EC, T_B1B2, E2_C2,
                     T_E1B1_list, T_B2E2_list, T_C2B_list,
                     intr, layout, include_sy=False, is_scara_x1=False, mat_to_vec6d=mat_to_vec6d):
    vecs = []
    if 'x1ec' in layout:
        e6 = mat_to_vec6d(X1_EC)
        vecs.append(e6[:5] if is_scara_x1 else e6)
    if 'b1b2' in layout: vecs.append(mat_to_vec6d(T_B1B2))
    if 'e2c2' in layout:  vecs.append(mat_to_vec6d(E2_C2))
    if 'e1b1' in layout:  vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
    if 'b2e2' in layout:  vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list]))
    if 'c2b'  in layout and len(T_C2B_list)>0:
        vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_C2B_list]))
    if 'intr' in layout:
        if include_sy:
            vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['sy'], intr['cx'], intr['cy']], dtype=float))
        else:
            vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['cx'], intr['cy']], dtype=float))
    return np.concatenate(vecs) if len(vecs)>0 else np.zeros(0)

def unpack_params_dual(x, X1_EC_ref, layout, nr,
                       include_sy=False, is_scara_x1=False):
    def blk(name): return layout.get(name, slice(0,0))

    # X1_EC
    if 'x1ec' in layout:
        v = x[blk('x1ec')]
        if is_scara_x1:
            base = mat_to_vec6d(X1_EC_ref)
            base[:5] = v
            X1_EC = vec6d_to_mat(np.array([*v, base[5]]))
        else:
            X1_EC = vec6d_to_mat(v)
    else:
        X1_EC = X1_EC_ref

    # 글로벌
    T_B1B2 = vec6d_to_mat(x[blk('b1b2')]) if 'b1b2' in layout else None
    E2_C2  = vec6d_to_mat(x[blk('e2c2')])  if 'e2c2'  in layout else None

    # per-pose
    def split6(v): return [v[i:i+6] for i in range(0, len(v), 6)]
    T_E1B1_list = [vec6d_to_mat(v6) for v6 in split6(x[blk('e1b1')])] if 'e1b1' in layout else []
    T_B2E2_list = [vec6d_to_mat(v6) for v6 in split6(x[blk('b2e2')])] if 'b2e2' in layout else []
    T_C2B_list  = [vec6d_to_mat(v6) for v6 in split6(x[blk('c2b')])]  if 'c2b'  in layout else []

    # intrinsics
    if 'intr' in layout:
        iv = x[blk('intr')]
        if include_sy:
            intr = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':iv[3],'cx':iv[4],'cy':iv[5],'include_sy':True}
        else:
            intr = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':iv[2],'cx':iv[3],'cy':iv[4],'include_sy':False}
    else:
        intr = None

    return X1_EC, T_B1B2, E2_C2, T_E1B1_list, T_B2E2_list, T_C2B_list, intr

# --- 수치 자코비안(중앙차분) -------------------------------------

def numeric_jacobian_cdiff_dual(x0, layout, X1_EC_ref, nr,
                                obj_pts_list, T_C2B_list,
                                include_sy=False, is_scara_x1=False,
                                h_rot=1e-7, h_trans=1e-6,
                                h_c=1e-7, h_kappa=1e-8, h_sx=1e-8, h_cxcy=1e-6):
    # 파라미터별 스텝 벡터
    h = np.full(layout['total'], 1e-7, dtype=float)

    def set_block_steps(name, step_rot=h_rot, step_trans=h_trans):
        if name not in layout: return
        s = layout[name]; n = s.stop - s.start
        if n == 0: return
        if name == 'x1ec' and ((n == 5) or is_scara_x1):
            h[s] = step_rot
            return
        for off in range(s.start, s.stop, 6):
            h[off:off+3]   = step_rot
            h[off+3:off+6] = step_trans

    for nm in ['x1ec','b1b2','e2c2','e1b1','b2e2','c2b']:
        set_block_steps(nm)

    if 'intr' in layout:
        s = layout['intr']
        if include_sy:
            h[s.start+0] = h_c
            h[s.start+1] = h_kappa
            h[s.start+2] = h_sx
            h[s.start+3] = h_sx
            h[s.start+4] = h_cxcy
            h[s.start+5] = h_cxcy
        else:
            h[s.start+0] = h_c
            h[s.start+1] = h_kappa
            h[s.start+2] = h_sx
            h[s.start+3] = h_cxcy
            h[s.start+4] = h_cxcy

    # f(x): 현재 x에서 전체 픽셀 예측 (이미지 잔차와 동일 차원)
    def _f(xvec):
        X1_EC, T_B1B2, E2_C2, T_E1B1_list, T_B2E2_list, T_C2B_list_cur, intr = \
            unpack_params_dual(xvec, X1_EC_ref, layout, nr, include_sy, is_scara_x1)
        return _project_dataset_flat_dual(
            X1_EC, T_B1B2, E2_C2,
            T_E1B1_list, T_B2E2_list,
            (T_C2B_list if len(T_C2B_list_cur)==0 else T_C2B_list_cur),
            obj_pts_list, intr
        )

    y0 = _f(x0)
    m, n = y0.size, x0.size
    J_num = np.empty((m, n), dtype=float)

    for i in range(n):
        hi = h[i]
        xp = x0.copy(); xp[i] += hi
        xm = x0.copy(); xm[i] -= hi
        yp = _f(xp);    ym = _f(xm)
        J_num[:, i] = (yp - ym) / (2.0 * hi)

    return J_num

# --- 비교/시각화 요약 --------------------------------------------

def summarize_jacobian_diff(J_ana, J_num, layout, tag="A_img vs FD"):
    mask = np.isfinite(J_ana) & np.isfinite(J_num)
    A = np.where(mask, J_ana, 0.0)
    N = np.where(mask, J_num, 0.0)
    D = A - N
    denom = np.maximum(np.abs(N), 1e-12)
    rel = np.abs(D) / denom

    abs_err = np.abs(D)
    print(f"[{tag}] abs err  mean={abs_err.mean():.3e}, median={np.median(abs_err):.3e}, max={abs_err.max():.3e}")
    print(f"[{tag}] rel err  mean={rel.mean():.3e},  median={np.median(rel):.3e}, 95%={np.quantile(rel,0.95):.3e}, max={rel.max():.3e}")

    # 블록별 상대오차 요약
    for nm in ['x1ec','b1b2','e2c2','e1b1','b2e2','c2b','intr']:
        if nm in layout:
            s = layout[nm]
            num = np.linalg.norm(N[:, s], axis=0) + 1e-12
            blk = np.linalg.norm(D[:, s], axis=0) / num
            print(f"  - {nm:<5} mean col-rel = {np.mean(blk):.3e}, max col-rel = {np.max(blk):.3e}")

    # 히트맵 (원하시면 주석 해제)
    # plot_jacobian_heatmap(N, layout, title=f'{tag} - FD (log10|.|)')
    # plot_jacobian_heatmap(A, layout, title=f'{tag} - Analytical (log10|.|)')
    # plot_jacobian_heatmap(D, layout, title=f'{tag} - AbsDiff (log10|.|)')


def run_optimization_with_vce_shared_target_v2(
    model_type: str,
    T_B1_Board_init: np.ndarray, T_C1E1_init: np.ndarray, T_B2B1_init: np.ndarray, T_C2E2_init: np.ndarray,
    T_E1B1_list_init: list, T_B2E2_list_init: list,
    obj_pts_list: list, img1_pts_list: list, img2_pts_list: list,
    T_E1B1_list_obs: list = None, T_B2E2_list_obs: list = None,
    intr1_init: dict = None, intr2_init: dict = None,
    sigma_image_px: float = 0.1, sigma_angle_deg: float = 0.1, sigma_trans_mm: float = 1.0,
    max_vce_iter: int = 5, max_param_iter: int = 15, term_thresh: float = 1e-6,
    estimate_b1board: bool = True, estimate_c1e1: bool = True, estimate_b2b1: bool = True,
    estimate_c2e2: bool = True, estimate_e1b1: bool = True, estimate_b2e2: bool = True,
    estimate_intr1: bool = False, estimate_intr2: bool = False,
    include_sy1: bool = False, include_sy2: bool = False, is_scara_c1e1: bool = False,
):
    assert model_type == 'division'
    nr = len(obj_pts_list)

    T_B1_Board, T_C1E1, T_B2B1, T_C2E2 = T_B1_Board_init.copy(), T_C1E1_init.copy(), T_B2B1_init.copy(), T_C2E2_init.copy()
    T_E1B1_list = [T.copy() for T in T_E1B1_list_init]
    T_B2E2_list = [T.copy() for T in T_B2E2_list_init]

    if intr1_init is None: intr1_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=1.0, cx=0.0, cy=0.0)
    if intr2_init is None: intr2_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=1.0, cx=0.0, cy=0.0)
    intr1, intr2 = {**intr1_init, 'include_sy': include_sy1}, {**intr2_init, 'include_sy': include_sy2}

    PROJ_KEYS = ['c', 'kappa', 'sx', 'sy', 'cx', 'cy']
    proj_intr1 = {k: intr1[k] for k in PROJ_KEYS}
    proj_intr2 = {k: intr2[k] for k in PROJ_KEYS}

    def _project(T_B1_Board_, T_C1E1_, T_B2B1_, T_C2E2_, T_E1B1_list_, T_B2E2_list_, obj_pts_list_, prj_intr1_, prj_intr2_):
        out1, out2 = [], []
        for i in range(nr):
            T_C1_Board = T_C1E1_ @ T_E1B1_list_[i] @ T_B1_Board_
            for pw in obj_pts_list_[i]:
                pc1 = T_C1_Board[:3, :3] @ pw + T_C1_Board[:3, 3]
                u, v = _project_point_division(pc1, **prj_intr1_); out1.extend((u, v))
            T_C2_Board = T_C2E2_ @ T_B2E2_list_[i] @ T_B2B1_ @ T_B1_Board_
            for pw in obj_pts_list_[i]:
                pc2 = T_C2_Board[:3, :3] @ pw + T_C2_Board[:3, 3]
                u, v = _project_point_division(pc2, **prj_intr2_); out2.extend((u, v))
        return np.concatenate([np.array(out1, float), np.array(out2, float)])

    l_obs_img = np.concatenate([p.ravel() for p in img1_pts_list] + [p.ravel() for p in img2_pts_list])
    ni_img = len(l_obs_img)
    obs_blocks = []
    if estimate_e1b1 and T_E1B1_list_obs: obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]))
    if estimate_b2e2 and T_B2E2_list_obs: obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]))
    l_obs = np.concatenate([l_obs_img, *obs_blocks]) if obs_blocks else l_obs_img

    VAR_I_FLOOR, VAR_A_FLOOR, VAR_T_FLOOR = 1e-12, (np.deg2rad(1e-9))**2, (1e-9)**2
    var_i, var_a, var_t = max(sigma_image_px**2, VAR_I_FLOOR), max(np.deg2rad(sigma_angle_deg)**2, VAR_A_FLOOR), max((sigma_trans_mm/1000.0)**2, VAR_T_FLOOR)
    lam, lam_up, lam_down, lam_min, lam_max, max_ls_tries = 1e-2, 5.0, 0.25, 1e-12, 1e+8, 8

    def compute_layout():
        layout, col = {}, 0
        dim_c1e1 = 5 if is_scara_c1e1 else 6
        if estimate_b1board: layout['b1board'] = slice(col, col+6); col += 6
        if estimate_c1e1:  layout['c1e1']  = slice(col, col+dim_c1e1); col += dim_c1e1
        if estimate_b2b1:  layout['b2b1']  = slice(col, col+6); col += 6
        if estimate_c2e2:  layout['c2e2']  = slice(col, col+6); col += 6
        if estimate_e1b1:  layout['e1b1']  = slice(col, col+6*nr); col += 6*nr
        if estimate_b2e2:  layout['b2e2']  = slice(col, col+6*nr); col += 6*nr
        if estimate_intr1: layout['intr1'] = slice(col, col+(6 if include_sy1 else 5)); col += (6 if include_sy1 else 5)
        if estimate_intr2: layout['intr2'] = slice(col, col+(6 if include_sy2 else 5)); col += (6 if include_sy2 else 5)
        layout['total'] = col
        return layout

    def build_x(layout):
        vecs = []
        if estimate_b1board: vecs.append(mat_to_vec6d(T_B1_Board))
        if estimate_c1e1: e6 = mat_to_vec6d(T_C1E1); vecs.append(e6[:5] if is_scara_c1e1 else e6)
        if estimate_b2b1: vecs.append(mat_to_vec6d(T_B2B1))
        if estimate_c2e2: vecs.append(mat_to_vec6d(T_C2E2))
        if estimate_e1b1: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
        if estimate_b2e2: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list]))
        if estimate_intr1: vecs.append(np.array([intr1[k] for k in (['c','kappa','sx','sy','cx','cy'] if include_sy1 else ['c','kappa','sx','cx','cy'])]))
        if estimate_intr2: vecs.append(np.array([intr2[k] for k in (['c','kappa','sx','sy','cx','cy'] if include_sy2 else ['c','kappa','sx','cx','cy'])]))
        return np.concatenate(vecs) if vecs else np.zeros(0)

    print("="*50); print("Dual-Arm Optimization Start (Shared Target V2)"); print("="*50)

    for vce_iter in range(max_vce_iter):
        print(f"\n--- VCE Iteration {vce_iter+1}/{max_vce_iter} ---")
        print(f"Variances: σ_img²={var_i:.3e}, σ_ang²={var_a:.3e}, σ_trans²={var_t:.3e}")

        Pll_diag_list = [np.full(ni_img, 1.0/var_i)]
        pose_tile = np.tile([1.0/var_a]*3 + [1.0/var_t]*3, nr)
        if estimate_e1b1 and T_E1B1_list_obs: Pll_diag_list.append(pose_tile)
        if estimate_b2e2 and T_B2E2_list_obs: Pll_diag_list.append(pose_tile)
        Pll_diag = np.concatenate(Pll_diag_list) if len(Pll_diag_list) > 1 else Pll_diag_list[0]

        layout = compute_layout()
        phi_best = np.inf

        for param_iter in range(max_param_iter):
            x_k = build_x(layout)
            f_pix = _project(T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, obj_pts_list, proj_intr1, proj_intr2)
            w_list = [l_obs_img - f_pix]
            if estimate_e1b1 and T_E1B1_list_obs: w_list.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]) - x_k[layout['e1b1']])
            if estimate_b2e2 and T_B2E2_list_obs: w_list.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]) - x_k[layout['b2e2']])
            w = np.concatenate(w_list)

            if np.isnan(w).any(): print(f"  [LM-{param_iter+1}] NaN residual detected. Stop."); break
            
            A_img, _ = calculate_analytical_jacobian_shared_target_v2(
                T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, obj_pts_list,
                c1=proj_intr1['c'], kappa1=proj_intr1['kappa'], sx1=proj_intr1['sx'], sy1=proj_intr1['sy'], cx1=proj_intr1['cx'], cy1=proj_intr1['cy'],
                c2=proj_intr2['c'], kappa2=proj_intr2['kappa'], sx2=proj_intr2['sx'], sy2=proj_intr2['sy'], cx2=proj_intr2['cx'], cy2=proj_intr2['cy'],
                estimate_b1board=estimate_b1board, estimate_c1e1=estimate_c1e1, 
                estimate_b2b1=estimate_b2b1, estimate_c2e2=estimate_c2e2, estimate_e1b1=estimate_e1b1, 
                estimate_b2e2=estimate_b2e2, estimate_intr1=estimate_intr1, estimate_intr2=estimate_intr2,
                include_sy1=include_sy1, include_sy2=include_sy2, is_scara_c1e1=is_scara_c1e1,
                mat_to_vec6d=mat_to_vec6d)
            
            A = np.zeros((w.size, layout['total'])); A[:ni_img, :] = A_img
            row_ptr = ni_img
            if estimate_e1b1 and T_E1B1_list_obs: A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
            if estimate_b2e2 and T_B2E2_list_obs: A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr
            
            # plot_jacobian(A, layout, title=f"Jacobian at VCE iter {vce_iter+1}, LM iter {param_iter+1}")

            phi0 = w.T @ (Pll_diag * w)
            N = A.T @ (Pll_diag[:, None] * A); b = A.T @ (Pll_diag * w)

            print(f"  [LM-{param_iter+1}] Start: φ={phi0:.4f}, ||b||={np.linalg.norm(b):.4e}")

            success = False
            for ls_try in range(max_ls_tries):
                try: N_aug = N + lam * np.diag(np.diag(N)); delta = np.linalg.solve(N_aug, b)
                except np.linalg.LinAlgError: lam = min(lam*lam_up, lam_max); continue
                
                # print(f"      - Try {ls_try+1}/{max_ls_tries}: λ={lam:.2e}, ||Δx||={np.linalg.norm(delta):.4e}")

                x_new = x_k + delta
                if not np.isfinite(x_new).all(): lam = min(lam*lam_up, lam_max); continue
                try:
                    T_B1_Board_new = vec6d_to_mat(x_new[layout['b1board']]) if estimate_b1board else T_B1_Board
                    if estimate_c1e1: v=x_new[layout['c1e1']]; T_C1E1_new = vec6d_to_mat(np.array([*v,mat_to_vec6d(T_C1E1)[5]])) if is_scara_c1e1 else vec6d_to_mat(v)
                    else: T_C1E1_new = T_C1E1
                    T_B2B1_new = vec6d_to_mat(x_new[layout['b2b1']]) if estimate_b2b1 else T_B2B1
                    T_C2E2_new = vec6d_to_mat(x_new[layout['c2e2']]) if estimate_c2e2 else T_C2E2
                    T_E1B1_list_new = [vec6d_to_mat(a) for a in x_new[layout['e1b1']].reshape(nr, 6)] if estimate_e1b1 else T_E1B1_list
                    T_B2E2_list_new = [vec6d_to_mat(a) for a in x_new[layout['b2e2']].reshape(nr, 6)] if estimate_b2e2 else T_B2E2_list
                    intr1_new, intr2_new = intr1, intr2
                    if estimate_intr1: iv=x_new[layout['intr1']]; intr1_new={**intr1,'c':iv[0],'k':iv[1],'sx':iv[2],'sy':(iv[3] if include_sy1 else iv[2]),'cx':iv[-2],'cy':iv[-1]}
                    if estimate_intr2: iv=x_new[layout['intr2']]; intr2_new={**intr2,'c':iv[0],'k':iv[1],'sx':iv[2],'sy':(iv[3] if include_sy2 else iv[2]),'cx':iv[-2],'cy':iv[-1]}
                    proj_intr1_new = {k: intr1_new[k] for k in PROJ_KEYS}
                    proj_intr2_new = {k: intr2_new[k] for k in PROJ_KEYS}
                except Exception: lam = min(lam * lam_up, lam_max); continue

                f_pix_new = _project(T_B1_Board_new, T_C1E1_new, T_B2B1_new, T_C2E2_new, T_E1B1_list_new, T_B2E2_list_new, obj_pts_list, proj_intr1_new, proj_intr2_new)
                w_list_new = [l_obs_img-f_pix_new]
                if estimate_e1b1 and T_E1B1_list_obs: w_list_new.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]) - x_new[layout['e1b1']])
                if estimate_b2e2 and T_B2E2_list_obs: w_list_new.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]) - x_new[layout['b2e2']])
                w_new = np.concatenate(w_list_new)
                if np.isnan(w_new).any(): lam=min(lam*lam_up,lam_max); continue
                phi_new = w_new.T @ (Pll_diag * w_new)

                # print(f"        -> φ_current={phi0:.4f}, φ_new={phi_new:.4f}, diff={phi0 - phi_new:.4f}")

                if phi_new < phi0:
                    T_B1_Board, T_C1E1, T_B2B1, T_C2E2 = T_B1_Board_new, T_C1E1_new, T_B2B1_new, T_C2E2_new
                    T_E1B1_list, T_B2E2_list = T_E1B1_list_new, T_B2E2_list_new
                    intr1, intr2 = intr1_new, intr2_new
                    lam = max(lam*lam_down, lam_min); phi_best=phi_new; success=True
                    # print(f"    - LM {param_iter+1}/{ls_try+1}: φ {phi0:.6f}→{phi_new:.6f}, λ={lam:.2e}"); break
                else:
                    lam = min(lam*lam_up, lam_max)
                    # print(f"        ❌ Step REJECTED. Increasing λ to {lam:.2e}.")
            
            if not success: print(f"  [LM-{param_iter+1}] No improvement."); break
            if abs(phi0 - phi_best) < term_thresh: print("  [LM] Converged."); break
        
        # --- VCE 업데이트 로직 ---
        print("  Updating variance components...")
        f_pix_final = _project(T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, obj_pts_list, proj_intr1, proj_intr2)
        v_list = [l_obs_img - f_pix_final]
        x_final = build_x(layout)
        if estimate_e1b1 and T_E1B1_list_obs: v_list.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]) - x_final[layout['e1b1']])
        if estimate_b2e2 and T_B2E2_list_obs: v_list.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]) - x_final[layout['b2e2']])
        v_hat = np.concatenate(v_list)

        A_final, _ = calculate_analytical_jacobian_shared_target_v2(
            T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, obj_pts_list,
            c1=proj_intr1['c'], kappa1=proj_intr1['kappa'], sx1=proj_intr1['sx'], sy1=proj_intr1['sy'], cx1=proj_intr1['cx'], cy1=proj_intr1['cy'],
            c2=proj_intr2['c'], kappa2=proj_intr2['kappa'], sx2=proj_intr2['sx'], sy2=proj_intr2['sy'], cx2=proj_intr2['cx'], cy2=proj_intr2['cy'],
            estimate_b1board=estimate_b1board, estimate_c1e1=estimate_c1e1, 
            estimate_b2b1=estimate_b2b1, estimate_c2e2=estimate_c2e2, estimate_e1b1=estimate_e1b1, 
            estimate_b2e2=estimate_b2e2, estimate_intr1=estimate_intr1, estimate_intr2=estimate_intr2,
            include_sy1=include_sy1, include_sy2=include_sy2, is_scara_c1e1=is_scara_c1e1,
            mat_to_vec6d=mat_to_vec6d)
        A = np.zeros((v_hat.size, layout['total'])); A[:ni_img, :] = A_final
        row_ptr = ni_img
        if estimate_e1b1 and T_E1B1_list_obs: A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
        if estimate_b2e2 and T_B2E2_list_obs: A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr

        try:
            N_mat = A.T @ (Pll_diag[:,None]*A)
            cF, lower = cho_factor(N_mat, check_finite=False)
            diag_Qlhat = np.array([ A[i,:] @ cho_solve((cF, lower), A[i,:], check_finite=False) for i in range(v_hat.size) ])
        except Exception:
            Q_xx = np.linalg.pinv(N_mat)
            diag_Qlhat = np.einsum('ij,jk,ik->i', A, Q_xx, A)
        diag_R = 1.0 - Pll_diag * diag_Qlhat

        R_img = float(np.sum(diag_R[:ni_img]))
        v_img = v_hat[:ni_img]
        sigma0_sq_img = (v_img.T @ (Pll_diag[:ni_img] * v_img)) / max(R_img, 1e-9)

        idx_ang_all, idx_trans_all = [], []
        offset = ni_img
        def _collect(enabled):
            nonlocal offset
            if not enabled: return
            base = offset
            for p in range(nr): b=base+6*p; idx_ang_all.extend(range(b,b+3)); idx_trans_all.extend(range(b+3,b+6))
            offset += 6*nr
        _collect(estimate_e1b1 and T_E1B1_list_obs); _collect(estimate_b2e2 and T_B2E2_list_obs)

        sigma0_sq_ang, sigma0_sq_trn = None, None
        if idx_ang_all: R_a=np.sum(diag_R[idx_ang_all]); v_a=v_hat[idx_ang_all]; sigma0_sq_ang = (v_a.T@(Pll_diag[idx_ang_all]*v_a))/max(R_a,1e-9)
        if idx_trans_all: R_t=np.sum(diag_R[idx_trans_all]); v_t=v_hat[idx_trans_all]; sigma0_sq_trn = (v_t.T@(Pll_diag[idx_trans_all]*v_t))/max(R_t,1e-9)

        print(f"  σ̂0²: img={sigma0_sq_img:.4f}" + (f", ang={sigma0_sq_ang:.4f}" if sigma0_sq_ang else "") + (f", trans={sigma0_sq_trn:.4f}" if sigma0_sq_trn else ""))
        old_i, old_a, old_t = var_i, var_a, var_t
        var_i = max(var_i * sigma0_sq_img, VAR_I_FLOOR)
        if sigma0_sq_ang: var_a = max(var_a * sigma0_sq_ang, VAR_A_FLOOR)
        if sigma0_sq_trn: var_t = max(var_t * sigma0_sq_trn, VAR_T_FLOOR)
        print(f"  σ² update: img {old_i:.3e}→{var_i:.3e}" + (f" | ang {old_a:.3e}→{var_a:.3e}" if sigma0_sq_ang else "") + (f" | trans {old_t:.3e}→{var_t:.3e}" if sigma0_sq_trn else ""))

    intr1_final, intr2_final = {**intr1}, {**intr2}
    print("\nDual-Arm Division Optimization Finished (Shared Target V2).")
    return T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, intr1_final, intr2_final

def run_optimization_with_vce_shared_target_v3(
    model_type: str,
    T_B1_Board_init: np.ndarray, T_C1E1_init: np.ndarray, T_B2B1_init: np.ndarray, T_C2E2_init: np.ndarray,
    T_E1B1_list_init: list, T_B2E2_list_init: list,
    obj_pts_list: list, img1_pts_list: list, img2_pts_list: list,
    T_E1B1_list_obs: list = None, T_B2E2_list_obs: list = None,
    intr1_init: dict = None, intr2_init: dict = None,
    sigma_image_px: float = 0.1, sigma_angle_deg: float = 0.1, sigma_trans_mm: float = 1.0,
    max_vce_iter: int = 5, max_param_iter: int = 15, term_thresh: float = 1e-6,
    estimate_b1board: bool = True, estimate_c1e1: bool = True, estimate_b2b1: bool = True,
    estimate_c2e2: bool = True, estimate_e1b1: bool = True, estimate_b2e2: bool = True,
    estimate_intr1: bool = False, estimate_intr2: bool = False,
    include_sy1: bool = False, include_sy2: bool = False, is_scara_c1e1: bool = False,
    use_schur: bool = True,         # ★ v3: Schur 보완 사용 스위치(기본 on)
    schur_damping: float = 0.0      # ★ v3: Hll 역이 불안정할 때 안정화를 위한 작은 댐핑(0~1e-6 권장)
):
    """
    v3 변경점:
      - LM 내부에서 로컬 변수(e1b1, b2e2)를 Schur 보완으로 제거하고 전역변수만 먼저 갱신.
      - 이후 로컬 증분은 back-substitution으로 복구.
      - 나머지 인터페이스/이름/흐름(VCE 포함)은 v2와 동일.
    """
    assert model_type == 'division'
    nr = len(obj_pts_list)

    # --- 초기화 ---
    T_B1_Board, T_C1E1, T_B2B1, T_C2E2 = (
        T_B1_Board_init.copy(), T_C1E1_init.copy(), T_B2B1_init.copy(), T_C2E2_init.copy()
    )
    T_E1B1_list = [T.copy() for T in T_E1B1_list_init]
    T_B2E2_list = [T.copy() for T in T_B2E2_list_init]

    if intr1_init is None: intr1_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=1.0, cx=0.0, cy=0.0)
    if intr2_init is None: intr2_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=1.0, cx=0.0, cy=0.0)
    intr1, intr2 = {**intr1_init, 'include_sy': include_sy1}, {**intr2_init, 'include_sy': include_sy2}

    PROJ_KEYS = ['c', 'kappa', 'sx', 'sy', 'cx', 'cy']
    proj_intr1 = {k: intr1[k] for k in PROJ_KEYS}
    proj_intr2 = {k: intr2[k] for k in PROJ_KEYS}

    def _project(T_B1_Board_, T_C1E1_, T_B2B1_, T_C2E2_, T_E1B1_list_, T_B2E2_list_, obj_pts_list_, prj_intr1_, prj_intr2_):
        out1, out2 = [], []
        for i in range(nr):
            T_C1_Board = T_C1E1_ @ T_E1B1_list_[i] @ T_B1_Board_
            for pw in obj_pts_list_[i]:
                pc1 = T_C1_Board[:3, :3] @ pw + T_C1_Board[:3, 3]
                u, v = _project_point_division(pc1, **prj_intr1_); out1.extend((u, v))
            T_C2_Board = T_C2E2_ @ T_B2E2_list_[i] @ T_B2B1_ @ T_B1_Board_
            for pw in obj_pts_list_[i]:
                pc2 = T_C2_Board[:3, :3] @ pw + T_C2_Board[:3, 3]
                u, v = _project_point_division(pc2, **prj_intr2_); out2.extend((u, v))
        return np.concatenate([np.array(out1, float), np.array(out2, float)])

    # 관측 벡터 구성 (이미지 + 선택적 포즈관측)
    l_obs_img = np.concatenate([p.ravel() for p in img1_pts_list] + [p.ravel() for p in img2_pts_list])
    ni_img = len(l_obs_img)
    obs_blocks = []
    if estimate_e1b1 and T_E1B1_list_obs: obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]))
    if estimate_b2e2 and T_B2E2_list_obs: obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]))
    l_obs = np.concatenate([l_obs_img, *obs_blocks]) if obs_blocks else l_obs_img

    # 분산 초기치 및 LM 파라미터
    VAR_I_FLOOR, VAR_A_FLOOR, VAR_T_FLOOR = 1e-12, (np.deg2rad(1e-9))**2, (1e-9)**2
    var_i = max(sigma_image_px**2, VAR_I_FLOOR)
    var_a = max(np.deg2rad(sigma_angle_deg)**2, VAR_A_FLOOR)
    var_t = max((sigma_trans_mm/1000.0)**2, VAR_T_FLOOR)
    lam, lam_up, lam_down, lam_min, lam_max, max_ls_tries = 1e-2, 5.0, 0.25, 1e-12, 1e+8, 8

    # --- 레이아웃 도우미 ---
    def compute_layout():
        layout, col = {}, 0
        dim_c1e1 = 5 if is_scara_c1e1 else 6
        if estimate_b1board: layout['b1board'] = slice(col, col+6); col += 6
        if estimate_c1e1:   layout['c1e1']   = slice(col, col+dim_c1e1); col += dim_c1e1
        if estimate_b2b1:   layout['b2b1']   = slice(col, col+6); col += 6
        if estimate_c2e2:   layout['c2e2']   = slice(col, col+6); col += 6
        if estimate_e1b1:   layout['e1b1']   = slice(col, col+6*nr); col += 6*nr
        if estimate_b2e2:   layout['b2e2']   = slice(col, col+6*nr); col += 6*nr
        if estimate_intr1:  layout['intr1']  = slice(col, col+(6 if include_sy1 else 5)); col += (6 if include_sy1 else 5)
        if estimate_intr2:  layout['intr2']  = slice(col, col+(6 if include_sy2 else 5)); col += (6 if include_sy2 else 5)
        layout['total'] = col
        # 전역 vs 로컬 인덱스 (Schur용)
        global_slices = []
        if estimate_b1board: global_slices.append(layout['b1board'])
        if estimate_c1e1:    global_slices.append(layout['c1e1'])
        if estimate_b2b1:    global_slices.append(layout['b2b1'])
        if estimate_c2e2:    global_slices.append(layout['c2e2'])
        if estimate_intr1:   global_slices.append(layout['intr1'])
        if estimate_intr2:   global_slices.append(layout['intr2'])
        layout['global_cols'] = np.concatenate([np.arange(s.start, s.stop) for s in global_slices]) if global_slices else np.array([], dtype=int)

        local_slices = []
        if estimate_e1b1: local_slices.append(layout['e1b1'])
        if estimate_b2e2: local_slices.append(layout['b2e2'])
        layout['local_cols'] = np.concatenate([np.arange(s.start, s.stop) for s in local_slices]) if local_slices else np.array([], dtype=int)
        return layout

    def build_x(layout):
        vecs = []
        if estimate_b1board: vecs.append(mat_to_vec6d(T_B1_Board))
        if estimate_c1e1:
            e6 = mat_to_vec6d(T_C1E1); vecs.append(e6[:5] if is_scara_c1e1 else e6)
        if estimate_b2b1: vecs.append(mat_to_vec6d(T_B2B1))
        if estimate_c2e2: vecs.append(mat_to_vec6d(T_C2E2))
        if estimate_e1b1: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
        if estimate_b2e2: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list]))
        if estimate_intr1: vecs.append(np.array([intr1[k] for k in (['c','kappa','sx','sy','cx','cy'] if include_sy1 else ['c','kappa','sx','cx','cy'])]))
        if estimate_intr2: vecs.append(np.array([intr2[k] for k in (['c','kappa','sx','sy','cx','cy'] if include_sy2 else ['c','kappa','sx','cx','cy'])]))
        return np.concatenate(vecs) if vecs else np.zeros(0)

    print("="*50); print("Dual-Arm Optimization Start (Shared Target V3; Schur)"); print("="*50)

    for vce_iter in range(max_vce_iter):
        print(f"\n--- VCE Iteration {vce_iter+1}/{max_vce_iter} ---")
        print(f"Variances: σ_img²={var_i:.3e}, σ_ang²={var_a:.3e}, σ_trans²={var_t:.3e}")

        # 가중치 대각 (단순형: 스칼라 그룹)
        Pll_diag_list = [np.full(ni_img, 1.0/var_i)]
        pose_tile = np.tile([1.0/var_a]*3 + [1.0/var_t]*3, nr)
        if estimate_e1b1 and T_E1B1_list_obs: Pll_diag_list.append(pose_tile)
        if estimate_b2e2 and T_B2E2_list_obs: Pll_diag_list.append(pose_tile)
        Pll_diag = np.concatenate(Pll_diag_list) if len(Pll_diag_list) > 1 else Pll_diag_list[0]

        layout = compute_layout()
        phi_best = np.inf

        for param_iter in range(max_param_iter):
            x_k = build_x(layout)

            # --- 예측 & 잔차 ---
            f_pix = _project(T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, obj_pts_list, proj_intr1, proj_intr2)
            w_list = [l_obs_img - f_pix]
            if estimate_e1b1 and T_E1B1_list_obs:
                w_list.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]) - x_k[layout['e1b1']])
            if estimate_b2e2 and T_B2E2_list_obs:
                w_list.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]) - x_k[layout['b2e2']])
            w = np.concatenate(w_list)
            if np.isnan(w).any():
                print(f"  [LM-{param_iter+1}] NaN residual detected. Stop."); break

            # --- 자코비안 ---
            A_img, _ = calculate_analytical_jacobian_shared_target_v2(
                T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, obj_pts_list,
                c1=proj_intr1['c'], kappa1=proj_intr1['kappa'], sx1=proj_intr1['sx'], sy1=proj_intr1['sy'], cx1=proj_intr1['cx'], cy1=proj_intr1['cy'],
                c2=proj_intr2['c'], kappa2=proj_intr2['kappa'], sx2=proj_intr2['sx'], sy2=proj_intr2['sy'], cx2=proj_intr2['cx'], cy2=proj_intr2['cy'],
                estimate_b1board=estimate_b1board, estimate_c1e1=estimate_c1e1,
                estimate_b2b1=estimate_b2b1, estimate_c2e2=estimate_c2e2, estimate_e1b1=estimate_e1b1,
                estimate_b2e2=estimate_b2e2, estimate_intr1=estimate_intr1, estimate_intr2=estimate_intr2,
                include_sy1=include_sy1, include_sy2=include_sy2, is_scara_c1e1=is_scara_c1e1,
                mat_to_vec6d=mat_to_vec6d
            )

            A = np.zeros((w.size, layout['total']))
            A[:ni_img, :] = A_img
            row_ptr = ni_img
            if estimate_e1b1 and T_E1B1_list_obs:
                A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
            if estimate_b2e2 and T_B2E2_list_obs:
                A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr

            # --- Normal equations ---
            # (가중치 Pll_diag는 대각으로 가정)
            phi0 = w.T @ (Pll_diag * w)
            # W*A 및 W*w
            WA = (Pll_diag[:, None] * A)
            Ww = (Pll_diag * w)
            # H = A^T W A, g = A^T W w
            H = A.T @ WA
            g = A.T @ Ww

            # --- LM damping (각 블록에 동일 적용) ---
            # diag(H)를 기준으로 LM 적용
            H_damped = H.copy()
            diagH = np.diag(H)
            np.fill_diagonal(H_damped, diagH + lam * np.maximum(diagH, 1.0))

            success = False

            # --- Schur 보완 or Full solve ---
            for ls_try in range(max_ls_tries):
                try:
                    if use_schur and layout['local_cols'].size > 0:
                        # 인덱스 준비
                        idx_g = layout['global_cols']
                        idx_l = layout['local_cols']

                        # 블록 추출
                        Hgg = H_damped[np.ix_(idx_g, idx_g)]
                        Hgl = H_damped[np.ix_(idx_g, idx_l)]
                        Hlg = H_damped[np.ix_(idx_l, idx_g)]
                        Hll = H_damped[np.ix_(idx_l, idx_l)]
                        gg  = g[idx_g]
                        gl  = g[idx_l]

                        # 안정화(필요 시 Hll에 소량 댐핑 추가)
                        if schur_damping > 0.0:
                            Hll = Hll + schur_damping * np.eye(Hll.shape[0])

                        # Hll^{-1} * v 풀기 (치올레스키 우선)
                        try:
                            cF_ll, lower_ll = cho_factor(Hll, check_finite=False)
                            Hll_inv_gl = cho_solve((cF_ll, lower_ll), gl, check_finite=False)
                            Hll_inv_Hlg = cho_solve((cF_ll, lower_ll), Hlg, check_finite=False)
                        except Exception:
                            Hll_inv = np.linalg.pinv(Hll)
                            Hll_inv_gl = Hll_inv @ gl
                            Hll_inv_Hlg = Hll_inv @ Hlg

                        # Schur 보완 S = Hgg - Hgl Hll^{-1} Hlg
                        S = Hgg - Hgl @ Hll_inv_Hlg
                        rhs = gg - Hgl @ Hll_inv_gl

                        # 전역 증분 Δx_g
                        try:
                            cF_s, lower_s = cho_factor(S, check_finite=False)
                            delta_g = cho_solve((cF_s, lower_s), rhs, check_finite=False)
                        except Exception:
                            delta_g = np.linalg.pinv(S) @ rhs

                        # 로컬 증분 Δx_l = Hll^{-1}(g_l - H_lg Δx_g)
                        delta_l_rhs = gl - Hlg @ delta_g
                        try:
                            if 'cF_ll' in locals():
                                delta_l = cho_solve((cF_ll, lower_ll), delta_l_rhs, check_finite=False)
                            else:
                                # 위에서 실패해 pinv 사용한 경우
                                delta_l = Hll_inv @ delta_l_rhs
                        except Exception:
                            delta_l = np.linalg.pinv(Hll) @ delta_l_rhs

                        # 전체 Δx 조립
                        delta = np.zeros(layout['total'])
                        delta[idx_g] = delta_g
                        delta[idx_l] = delta_l
                    else:
                        # 전체 풀기 (v2와 동일 경로)
                        delta = np.linalg.solve(H_damped, g)
                except np.linalg.LinAlgError:
                    lam = min(lam*lam_up, lam_max); continue

                # --- 스텝 적용 & 평가 ---
                x_new = x_k + delta
                if not np.isfinite(x_new).all():
                    lam = min(lam*lam_up, lam_max); continue

                try:
                    T_B1_Board_new = vec6d_to_mat(x_new[layout['b1board']]) if estimate_b1board else T_B1_Board
                    if estimate_c1e1:
                        v = x_new[layout['c1e1']]
                        T_C1E1_new = vec6d_to_mat(np.array([*v, mat_to_vec6d(T_C1E1)[5]])) if is_scara_c1e1 else vec6d_to_mat(v)
                    else:
                        T_C1E1_new = T_C1E1
                    T_B2B1_new = vec6d_to_mat(x_new[layout['b2b1']]) if estimate_b2b1 else T_B2B1
                    T_C2E2_new = vec6d_to_mat(x_new[layout['c2e2']]) if estimate_c2e2 else T_C2E2
                    T_E1B1_list_new = [vec6d_to_mat(a) for a in x_new[layout['e1b1']].reshape(nr, 6)] if estimate_e1b1 else T_E1B1_list
                    T_B2E2_list_new = [vec6d_to_mat(a) for a in x_new[layout['b2e2']].reshape(nr, 6)] if estimate_b2e2 else T_B2E2_list
                    intr1_new, intr2_new = intr1, intr2
                    if estimate_intr1:
                        iv = x_new[layout['intr1']]
                        intr1_new = {**intr1, 'c':iv[0],'k':iv[1],'sx':iv[2],'sy':(iv[3] if include_sy1 else iv[2]),'cx':iv[-2],'cy':iv[-1]}
                    if estimate_intr2:
                        iv = x_new[layout['intr2']]
                        intr2_new = {**intr2, 'c':iv[0],'k':iv[1],'sx':iv[2],'sy':(iv[3] if include_sy2 else iv[2]),'cx':iv[-2],'cy':iv[-1]}
                    proj_intr1_new = {k: intr1_new[k] for k in PROJ_KEYS}
                    proj_intr2_new = {k: intr2_new[k] for k in PROJ_KEYS}
                except Exception:
                    lam = min(lam*lam_up, lam_max); continue

                # 새 목적함수 값
                f_pix_new = _project(T_B1_Board_new, T_C1E1_new, T_B2B1_new, T_C2E2_new,
                                     T_E1B1_list_new, T_B2E2_list_new, obj_pts_list, proj_intr1_new, proj_intr2_new)
                w_list_new = [l_obs_img - f_pix_new]
                if estimate_e1b1 and T_E1B1_list_obs:
                    w_list_new.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]) - x_new[layout['e1b1']])
                if estimate_b2e2 and T_B2E2_list_obs:
                    w_list_new.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]) - x_new[layout['b2e2']])
                w_new = np.concatenate(w_list_new)
                if np.isnan(w_new).any():
                    lam = min(lam*lam_up, lam_max); continue
                phi_new = w_new.T @ (Pll_diag * w_new)

                print(f"  [LM-{param_iter+1}] φ={phi0:.6f} → {phi_new:.6f} (Δ={phi0-phi_new:.3e}), λ={lam:.2e}")

                if phi_new < phi0:
                    # 수락
                    T_B1_Board, T_C1E1, T_B2B1, T_C2E2 = T_B1_Board_new, T_C1E1_new, T_B2B1_new, T_C2E2_new
                    T_E1B1_list, T_B2E2_list = T_E1B1_list_new, T_B2E2_list_new
                    intr1, intr2 = intr1_new, intr2_new
                    lam = max(lam*lam_down, lam_min)
                    phi_best = phi_new
                    success = True
                    break
                else:
                    # 거부: damping 증가
                    lam = min(lam*lam_up, lam_max)

            if not success:
                print(f"  [LM-{param_iter+1}] No improvement."); break
            if abs(phi0 - phi_best) < term_thresh:
                print("  [LM] Converged."); break

        # --- VCE 업데이트 (v2와 동일 로직 유지) ---
        print("  Updating variance components...")
        f_pix_final = _project(T_B1_Board, T_C1E1, T_B2B1, T_C2E2,
                               T_E1B1_list, T_B2E2_list, obj_pts_list, proj_intr1, proj_intr2)
        v_list = [l_obs_img - f_pix_final]
        x_final = build_x(layout)
        if estimate_e1b1 and T_E1B1_list_obs:
            v_list.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]) - x_final[layout['e1b1']])
        if estimate_b2e2 and T_B2E2_list_obs:
            v_list.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]) - x_final[layout['b2e2']])
        v_hat = np.concatenate(v_list)

        A_final, _ = calculate_analytical_jacobian_shared_target_v2(
            T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, obj_pts_list,
            c1=proj_intr1['c'], kappa1=proj_intr1['kappa'], sx1=proj_intr1['sx'], sy1=proj_intr1['sy'], cx1=proj_intr1['cx'], cy1=proj_intr1['cy'],
            c2=proj_intr2['c'], kappa2=proj_intr2['kappa'], sx2=proj_intr2['sx'], sy2=proj_intr2['sy'], cx2=proj_intr2['cx'], cy2=proj_intr2['cy'],
            estimate_b1board=estimate_b1board, estimate_c1e1=estimate_c1e1,
            estimate_b2b1=estimate_b2b1, estimate_c2e2=estimate_c2e2, estimate_e1b1=estimate_e1b1,
            estimate_b2e2=estimate_b2e2, estimate_intr1=estimate_intr1, estimate_intr2=estimate_intr2,
            include_sy1=include_sy1, include_sy2=include_sy2, is_scara_c1e1=is_scara_c1e1,
            mat_to_vec6d=mat_to_vec6d
        )
        A = np.zeros((v_hat.size, layout['total']))
        A[:ni_img, :] = A_final
        row_ptr = ni_img
        if estimate_e1b1 and T_E1B1_list_obs:
            A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
        if estimate_b2e2 and T_B2E2_list_obs:
            A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr

        # diag_R 계산
        try:
            N_mat = A.T @ (Pll_diag[:, None] * A)
            cF, lower = cho_factor(N_mat, check_finite=False)
            # diag(A Qxx A^T) 효율 계산
            diag_Qlhat = np.array([ A[i,:] @ cho_solve((cF, lower), A[i,:], check_finite=False) for i in range(v_hat.size) ])
        except Exception:
            Q_xx = np.linalg.pinv(N_mat)
            diag_Qlhat = np.einsum('ij,jk,ik->i', A, Q_xx, A)
        diag_R = 1.0 - Pll_diag * diag_Qlhat

        # 그룹별 σ̂0²
        R_img = float(np.sum(diag_R[:ni_img])); v_img = v_hat[:ni_img]
        sigma0_sq_img = (v_img.T @ (Pll_diag[:ni_img] * v_img)) / max(R_img, 1e-9)

        idx_ang_all, idx_trans_all = [], []
        offset = ni_img
        def _collect(enabled):
            nonlocal offset
            if not enabled: return
            base = offset
            for p in range(nr):
                b = base + 6*p
                idx_ang_all.extend(range(b, b+3))
                idx_trans_all.extend(range(b+3, b+6))
            offset += 6*nr
        _collect(estimate_e1b1 and T_E1B1_list_obs)
        _collect(estimate_b2e2 and T_B2E2_list_obs)

        sigma0_sq_ang, sigma0_sq_trn = None, None
        if idx_ang_all:
            R_a = float(np.sum(diag_R[idx_ang_all]))
            v_a = v_hat[idx_ang_all]
            sigma0_sq_ang = float((v_a.T @ (Pll_diag[idx_ang_all] * v_a)) / max(R_a, 1e-9))
        if idx_trans_all:
            R_t = float(np.sum(diag_R[idx_trans_all]))
            v_t = v_hat[idx_trans_all]
            sigma0_sq_trn = float((v_t.T @ (Pll_diag[idx_trans_all] * v_t)) / max(R_t, 1e-9))

        print(f"  σ̂0²: img={sigma0_sq_img:.4f}" + (f", ang={sigma0_sq_ang:.4f}" if sigma0_sq_ang is not None else "") + (f", trans={sigma0_sq_trn:.4f}" if sigma0_sq_trn is not None else ""))
        old_i, old_a, old_t = var_i, var_a, var_t
        var_i = max(var_i * sigma0_sq_img, VAR_I_FLOOR)
        if sigma0_sq_ang is not None: var_a = max(var_a * sigma0_sq_ang, VAR_A_FLOOR)
        if sigma0_sq_trn is not None: var_t = max(var_t * sigma0_sq_trn, VAR_T_FLOOR)
        print(f"  σ² update: img {old_i:.3e}→{var_i:.3e}" + (f" | ang {old_a:.3e}→{var_a:.3e}" if sigma0_sq_ang is not None else "") + (f" | trans {old_t:.3e}→{var_t:.3e}" if sigma0_sq_trn is not None else ""))

    intr1_final, intr2_final = {**intr1}, {**intr2}
    print("\nDual-Arm Division Optimization Finished (Shared Target V3; Schur).")
    return T_B1_Board, T_C1E1, T_B2B1, T_C2E2, T_E1B1_list, T_B2E2_list, intr1_final, intr2_final


def plot_jacobian(A, layout, title="Jacobian Matrix"):
    """
    주어진 자코비안 행렬 A와 layout을 사용하여 sparsity와 magnitude를 시각화합니다.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    
    # --- 1. Sparsity Pattern (어떤 원소가 0이 아닌가) ---
    ax = axs[0]
    ax.spy(A, markersize=0.5)
    ax.set_title("Sparsity Pattern (Non-zero elements)")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Observations")

    # --- 2. Magnitude Heatmap (원소의 크기는 얼마인가) ---
    ax = axs[1]
    # 0인 값을 마스킹하고, 로그 스케일로 크기를 표현하여 가시성을 높임
    A_abs = np.abs(A)
    A_log = np.log10(np.where(A_abs > 1e-12, A_abs, np.nan))
    
    im = ax.imshow(A_log, cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=ax, label="log10(|Jacobian Value|)")
    ax.set_title("Magnitude Heatmap (log scale)")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Observations")
    
    # --- layout에 따라 구분선과 라벨 추가 ---
    labels = []
    positions = []
    for ax in axs:
        for name, col_slice in layout.items():
            if name == 'total': continue
            start = col_slice.start
            end = col_slice.stop -1
            
            # 파라미터 블록의 중간 위치에 라벨 추가
            positions.append(start + (end - start) / 2)
            labels.append(name)
            
            # 파라미터 블록 끝에 구분선 추가
            if start > 0:
                ax.axvline(x=start - 0.5, color='r', linestyle='--', linewidth=0.8)

    for ax in axs:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==============================================================================
# 헬퍼 함수 (Helper Functions)
# - 메인 함수가 의존하는 헬퍼 함수들을 먼저 정의합니다.
# ==============================================================================

def _project_dataset_flat_revised(
    X1_EC: np.ndarray,
    T_B1B2: np.ndarray,
    T_E2B: np.ndarray,
    T_E1B1_list: list,
    T_B2E2_list: list,
    obj_pts_list: list,
    intr: dict,
    # (수정) 유틸리티 함수를 인자로 받음
    _project_point_division=None
) -> np.ndarray:
    """
    (수정) 새로운 변환 체인에 맞춰 모든 3D 객체 포인트를 투영하고 1D 벡터로 반환합니다.
    내부 투영 로직 대신 _project_point_division 유틸리티 함수를 사용합니다.
    """
    assert _project_point_division is not None, "_project_point_division 함수를 전달해야 합니다."
    
    all_projected_pts = []
    
    c, kappa, sx, sy, cx, cy = intr['c'], intr['kappa'], intr['sx'], intr['sy'], intr['cx'], intr['cy']

    for i in range(len(obj_pts_list)):
        # 전체 변환 행렬 계산: Board -> Camera
        T_C1_B = X1_EC @ T_E1B1_list[i] @ T_B1B2 @ T_B2E2_list[i] @ T_E2B
        
        # 3D 포인트를 동차 좌표로 변환
        pts_b_hom = np.hstack([obj_pts_list[i], np.ones((obj_pts_list[i].shape[0], 1))])
        
        # 카메라 좌표계로 변환
        pts_c = (T_C1_B @ pts_b_hom.T).T[:, :3]

        # (수정) 각 포인트에 대해 유틸리티 함수를 사용하여 투영
        projected_pts_for_pose = np.zeros((len(pts_c), 2), dtype=float)
        for pt_idx, pc in enumerate(pts_c):
            u, v = _project_point_division(pc, c, kappa, sx, sy, cx, cy)
            projected_pts_for_pose[pt_idx] = [u, v]
            
        all_projected_pts.append(projected_pts_for_pose)
        
    return np.concatenate([pts.reshape(-1) for pts in all_projected_pts])

# ==============================================================================
# 메인 최적화 함수 (Main Optimization Function)
# - 최종 버전: 이름 변경 및 전체 코드 구현 완료
# ==============================================================================

def run_optimization_with_vce_axbycz(
    # 모델/데이터
    model_type: str,
    X1_EC_init: np.ndarray,
    T_B1B2_init: np.ndarray,
    T_E2B_init: np.ndarray,
    T_E1B1_list_init: list,
    T_B2E2_list_init: list,
    img_pts_list: list,
    obj_pts_list: list,
    # 포즈 관측(옵션)
    T_E1B1_list_obs: list = None,
    T_B2E2_list_obs: list = None,
    # intrinsics
    intrinsics_init: dict = None,
    # 노이즈 (초기 분산)
    sigma_image_px: float = 0.1,
    sigma_angle_deg: float = 0.1,
    sigma_trans_mm: float = 1.0,
    # 반복/LM
    max_vce_iter: int = 5,
    max_param_iter: int = 10,
    term_thresh: float = 1e-6,
    # 추정 플래그
    estimate_x1ec: bool = True,
    estimate_b1b2: bool = True,
    estimate_e2b:  bool = True,
    estimate_e1b1: bool = True,
    estimate_b2e2: bool = True,
    estimate_intrinsics: bool = False,
    include_sy: bool = False,
    is_scara_x1: bool = False,
):
    """
    (최종) VCE를 포함한 Levenberg-Marquardt 최적화.
    'Board -> E2 -> B2 -> B1 -> E1 -> C1' 변환 체인에 맞춰 작성되었습니다.
    """
    assert model_type == 'division', "현재 division 모델만 지원합니다."
    
    nr = len(obj_pts_list)
    # --- 상태 변수 초기화 ---
    X1_EC   = X1_EC_init.copy()
    T_B1B2  = T_B1B2_init.copy()
    T_E2B   = T_E2B_init.copy()
    T_E1B1_list = [T.copy() for T in T_E1B1_list_init]
    T_B2E2_list = [T.copy() for T in T_B2E2_list_init]

    # --- intrinsics 상태 ---
    if intrinsics_init is None:
        intrinsics_init = dict(c=1.0, kappa=0.0, sx=1.0, sy=1.0, cx=0.0, cy=0.0, include_sy=include_sy)
    intr = {
        'c': intrinsics_init['c'], 'kappa': intrinsics_init['kappa'], 'sx': intrinsics_init['sx'],
        'sy': intrinsics_init['sy'],
        'cx': intrinsics_init['cx'], 'cy': intrinsics_init['cy'], 'include_sy': include_sy
    }

    # --- 관측 벡터 구성 ---
    l_obs_img = np.concatenate([pts.reshape(-1) for pts in img_pts_list])
    obs_blocks = []
    if estimate_e1b1 and (T_E1B1_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs]))
    if estimate_b2e2 and (T_B2E2_list_obs is not None):
        obs_blocks.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs]))

    l_obs = np.concatenate([l_obs_img, *obs_blocks]) if obs_blocks else l_obs_img
    ni_img = len(l_obs_img)

    # --- 초기 분산/LM 파라미터 ---
    VAR_I_FLOOR, VAR_A_FLOOR, VAR_T_FLOOR = 1e-12, (np.deg2rad(1e-9))**2, (1e-9)**2
    var_i = max(sigma_image_px**2, VAR_I_FLOOR)
    var_a = max(np.deg2rad(sigma_angle_deg)**2, VAR_A_FLOOR)
    var_t = max((sigma_trans_mm / 1000.0)**2, VAR_T_FLOOR)
    lam, lam_up, lam_down, lam_min, lam_max, max_ls_tries = 1e-2, 5.0, 0.25, 1e-12, 1e+8, 8

    # --- 레이아웃 헬퍼 함수 ---
    def compute_layout():
        layout = {}
        col = 0
        dim_x1 = 5 if is_scara_x1 else 6
        if estimate_x1ec: layout['x1ec'] = slice(col, col+dim_x1); col += dim_x1
        if estimate_b1b2: layout['b1b2'] = slice(col, col+6); col += 6
        if estimate_e2b:  layout['e2b']  = slice(col, col+6); col += 6
        if estimate_e1b1: layout['e1b1'] = slice(col, col+6*nr); col += 6*nr
        if estimate_b2e2: layout['b2e2'] = slice(col, col+6*nr); col += 6*nr
        if estimate_intrinsics:
            layout['intr'] = slice(col, col + (6 if include_sy else 5)); col += (6 if include_sy else 5)
        layout['total'] = col
        return layout

    # --- 상태 벡터 빌드 헬퍼 함수 ---
    def build_x_current(layout):
        vecs = []
        if estimate_x1ec:
            e6 = mat_to_vec6d(X1_EC); vecs.append(e6[:5] if is_scara_x1 else e6)
        if estimate_b1b2: vecs.append(mat_to_vec6d(T_B1B2))
        if estimate_e2b:  vecs.append(mat_to_vec6d(T_E2B))
        if estimate_e1b1: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list]))
        if estimate_b2e2: vecs.append(np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list]))
        if estimate_intrinsics:
            if include_sy: vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['sy'], intr['cx'], intr['cy']]))
            else:          vecs.append(np.array([intr['c'], intr['kappa'], intr['sx'], intr['cx'], intr['cy']]))
        return np.concatenate(vecs) if vecs else np.zeros(0)

    print("="*50); print("Revised Dual-Arm VCE Optimization Start"); print("="*50)

    for vce_iter in range(max_vce_iter):
        print(f"\n--- VCE Iteration {vce_iter+1}/{max_vce_iter} ---")
        print(f"Variances: σ_img²={var_i:.3e}, σ_ang²={var_a:.3e}, σ_trans²={var_t:.3e}")
        
        wi = 1.0/var_i
        Pll_diag_list = [np.full(ni_img, wi)]
        pose_weights = np.tile(np.concatenate([np.full(3, 1.0/var_a), np.full(3, 1.0/var_t)]), nr)
        if estimate_e1b1 and (T_E1B1_list_obs is not None): Pll_diag_list.append(pose_weights)
        if estimate_b2e2 and (T_B2E2_list_obs is not None): Pll_diag_list.append(pose_weights)
        Pll_diag = np.concatenate(Pll_diag_list)
        
        layout = compute_layout()
        phi_best = np.inf

        for param_iter in range(max_param_iter):
            x_k = build_x_current(layout)

            f_pix = _project_dataset_flat_revised(
                X1_EC, T_B1B2, T_E2B, T_E1B1_list, T_B2E2_list, obj_pts_list, intr,
                _project_point_division=_project_point_division
            )

            w_list = [l_obs_img - f_pix]
            if estimate_e1b1 and (T_E1B1_list_obs is not None):
                obs_e1b1 = np.concatenate([mat_to_vec6d(T) for T in T_E1B1_list_obs])
                w_list.append(obs_e1b1 - x_k[layout['e1b1']])
            if estimate_b2e2 and (T_B2E2_list_obs is not None):
                obs_b2e2 = np.concatenate([mat_to_vec6d(T) for T in T_B2E2_list_obs])
                w_list.append(obs_b2e2 - x_k[layout['b2e2']])
            w = np.concatenate(w_list)

            if np.isnan(w).any(): print(f"  [LM-{param_iter+1}] NaN residual detected. Stop."); break

            A_img = calculate_analytical_jacobian_division_model_axbycz(
                X1_EC=X1_EC, T_B1B2=T_B1B2, T_E2B=T_E2B,
                T_E1B1_list=T_E1B1_list, T_B2E2_list=T_B2E2_list,
                obj_pts_list=obj_pts_list,
                c=intr['c'], kappa=intr['kappa'], sx=intr['sx'], sy=intr['sy'], cx=intr['cx'], cy=intr['cy'],
                estimate_x1ec=estimate_x1ec, estimate_b1b2=estimate_b1b2, estimate_e2b=estimate_e2b,
                estimate_e1b1=estimate_e1b1, estimate_b2e2=estimate_b2e2,
                estimate_intrinsics=estimate_intrinsics,
                include_sy=include_sy, is_scara_x1=is_scara_x1,
                mat_to_vec6d=mat_to_vec6d
            )

            A = np.zeros((w.size, layout['total']), dtype=float)
            A[:ni_img, :] = A_img
            row_ptr = ni_img
            if estimate_e1b1 and (T_E1B1_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
            if estimate_b2e2 and (T_B2E2_list_obs is not None):
                A[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr

            phi0 = float(np.sum(Pll_diag * (w**2)))
            N = A.T @ (Pll_diag[:, None] * A)
            
            # # 조건수 계산
            # cond_num = np.linalg.cond(N)
            # print(f"  [DEBUG] Normal Matrix Condition Number: {cond_num:.2e}")

            # # 만약 cond_num이 매우 크다면, 여기서부터 문제가 있음을 알 수 있습니다.
            # if cond_num > 1e4:
            #     print("  [WARN] Problem is likely ill-conditioned!")
            
            # # 특이값 분해 수행
            # u, s, vh = np.linalg.svd(N)

            # print(f"  [DEBUG] Singular Values (s):")
            # print(f"    - Max: {s[0]:.2e}")
            # print(f"    - Min: {s[-1]:.2e}")
            # print(f"    - Ratio (Max/Min): {(s[0]/s[-1]):.2e}") # 이것이 조건수와 유사한 역할을 합니다.

            # # 시각화 (매우 유용)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.semilogy(s, 'o-')
            # plt.title('Singular Value Spectrum of N')
            # plt.ylabel('Singular Value (log scale)')
            # plt.xlabel('Index')
            # plt.grid(True)
            # plt.show()

            b = A.T @ (Pll_diag * w)
            
            success = False
            for ls_try in range(max_ls_tries):
                try:
                    N_aug = N + lam * np.diag(np.diag(N))
                    delta = np.linalg.solve(N_aug, b)
                except np.linalg.LinAlgError:
                    lam = min(lam * lam_up, lam_max); continue

                x_new = x_k + delta
                if not np.isfinite(x_new).all(): lam = min(lam * lam_up, lam_max); continue

                try:
                    X1_EC_new = X1_EC
                    if estimate_x1ec:
                        v = x_new[layout['x1ec']]
                        X1_EC_new = vec6d_to_mat(np.array([*v, mat_to_vec6d(X1_EC)[5]])) if is_scara_x1 else vec6d_to_mat(v)
                    T_B1B2_new = vec6d_to_mat(x_new[layout['b1b2']]) if estimate_b1b2 else T_B1B2
                    T_E2B_new = vec6d_to_mat(x_new[layout['e2b']]) if estimate_e2b else T_E2B
                    T_E1B1_list_new = [vec6d_to_mat(a) for a in x_new[layout['e1b1']].reshape(nr, 6)] if estimate_e1b1 else T_E1B1_list
                    T_B2E2_list_new = [vec6d_to_mat(a) for a in x_new[layout['b2e2']].reshape(nr, 6)] if estimate_b2e2 else T_B2E2_list
                    intr_new = intr
                    if estimate_intrinsics:
                        iv = x_new[layout['intr']]
                        if include_sy: intr_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':iv[3],'cx':iv[4],'cy':iv[5],'include_sy':True}
                        else:          intr_new = {'c':iv[0],'kappa':iv[1],'sx':iv[2],'sy':intr['sy'],'cx':iv[3],'cy':iv[4],'include_sy':False}
                except Exception:
                    lam = min(lam * lam_up, lam_max); continue

                f_pix_new = _project_dataset_flat_revised(
                    X1_EC_new, T_B1B2_new, T_E2B_new, T_E1B1_list_new, T_B2E2_list_new, obj_pts_list, intr_new,
                    _project_point_division=_project_point_division
                )
                w_list_new = [l_obs_img - f_pix_new]
                if estimate_e1b1 and (T_E1B1_list_obs is not None): w_list_new.append(obs_e1b1 - x_new[layout['e1b1']])
                if estimate_b2e2 and (T_B2E2_list_obs is not None): w_list_new.append(obs_b2e2 - x_new[layout['b2e2']])
                w_new = np.concatenate(w_list_new)
                
                if np.isnan(w_new).any(): lam = min(lam * lam_up, lam_max); continue
                
                phi_new = float(np.sum(Pll_diag * (w_new**2)))
                if phi_new < phi0:
                    X1_EC, T_B1B2, T_E2B = X1_EC_new, T_B1B2_new, T_E2B_new
                    T_E1B1_list, T_B2E2_list = T_E1B1_list_new, T_B2E2_list_new
                    intr = intr_new
                    lam = max(lam * lam_down, lam_min)
                    phi_best = phi_new
                    success = True
                    print(f"    - LM {param_iter+1} / try {ls_try+1}: φ {phi0:.6f} → {phi_new:.6f}, λ={lam:.2e}")
                    break
                else:
                    lam = min(lam * lam_up, lam_max)

            if not success: print(f"  [LM-{param_iter+1}] No improvement after {max_ls_tries} tries."); break
            if abs(phi0 - phi_best) < term_thresh: print("  [LM] Converged. Stop parameter search."); break

        # --- (F) VCE 업데이트 (전체 코드 구현) ---
        print("  Updating variance components...")
        f_pix_final = _project_dataset_flat_revised(
            X1_EC, T_B1B2, T_E2B, T_E1B1_list, T_B2E2_list, obj_pts_list, intr,
            _project_point_division=_project_point_division
        )
        v_list = [l_obs_img - f_pix_final]
        x_final = build_x_current(layout)
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            v_list.append(obs_e1b1 - x_final[layout['e1b1']])
        if estimate_b2e2 and (T_B2E2_list_obs is not None):
            v_list.append(obs_b2e2 - x_final[layout['b2e2']])
        v_hat = np.concatenate(v_list)

        A_final = np.zeros((v_hat.size, layout['total']), dtype=float)
        A_final[:ni_img, :] = A_img # LM 마지막 스텝의 자코비안 재사용
        row_ptr = ni_img
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            A_final[row_ptr:row_ptr+6*nr, layout['e1b1']] = np.eye(6*nr); row_ptr += 6*nr
        if estimate_b2e2 and (T_B2E2_list_obs is not None):
            A_final[row_ptr:row_ptr+6*nr, layout['b2e2']] = np.eye(6*nr); row_ptr += 6*nr
        
        N_mat = A_final.T @ (Pll_diag[:,None]*A_final)
        try:
            # 더 빠르고 안정적인 Cholesky 분해 사용 시도
            from scipy.linalg import cho_factor, cho_solve
            cF, lower = cho_factor(N_mat, check_finite=False)
            diag_Qlhat = np.array([float(A_final[i,:] @ cho_solve((cF, lower), A_final[i,:], check_finite=False)) for i in range(v_hat.size)])
        except Exception:
            # 실패 시 일반적인 pseudo-inverse 사용
            Q_xx = np.linalg.pinv(N_mat)
            diag_Qlhat = np.einsum('ij,jk,ik->i', A_final, Q_xx, A_final)

        diag_R = 1.0 - Pll_diag * diag_Qlhat

        R_img = float(np.sum(diag_R[:ni_img]))
        v_img = v_hat[:ni_img]
        sigma0_sq_img = float(v_img.T @ (Pll_diag[:ni_img] * v_img) / max(R_img, 1e-9))

        idx_ang_all, idx_trans_all = [], []
        offset = ni_img
        if estimate_e1b1 and (T_E1B1_list_obs is not None):
            for p in range(nr):
                b = offset + 6*p
                idx_ang_all.extend([b, b+1, b+2]); idx_trans_all.extend([b+3, b+4, b+5])
            offset += 6*nr
        if estimate_b2e2 and (T_B2E2_list_obs is not None):
            for p in range(nr):
                b = offset + 6*p
                idx_ang_all.extend([b, b+1, b+2]); idx_trans_all.extend([b+3, b+4, b+5])
            offset += 6*nr

        idx_ang, idx_trans = np.array(idx_ang_all, dtype=int), np.array(idx_trans_all, dtype=int)
        sigma0_sq_ang, sigma0_sq_trn = None, None

        if idx_ang.size:
            R_ang = float(np.sum(diag_R[idx_ang]))
            v_ang = v_hat[idx_ang]
            sigma0_sq_ang = float(v_ang.T @ (Pll_diag[idx_ang] * v_ang) / max(R_ang, 1e-9))
        if idx_trans.size:
            R_trans = float(np.sum(diag_R[idx_trans]))
            v_trn = v_hat[idx_trans]
            sigma0_sq_trn = float(v_trn.T @ (Pll_diag[idx_trans] * v_trn) / max(R_trans, 1e-9))

        print("  σ̂0² (should → 1): "
            f"img={sigma0_sq_img:.4f}"
            + (f", ang={sigma0_sq_ang:.4f}" if sigma0_sq_ang is not None else "")
            + (f", trans={sigma0_sq_trn:.4f}" if sigma0_sq_trn is not None else ""))

        alpha = 1.0
        old_var_i, old_var_a, old_var_t = var_i, var_a, var_t
        var_i = max((1-alpha)*var_i + alpha*var_i*sigma0_sq_img, VAR_I_FLOOR)
        if sigma0_sq_ang is not None: var_a = max((1-alpha)*var_a + alpha*var_a*sigma0_sq_ang, VAR_A_FLOOR)
        if sigma0_sq_trn is not None: var_t = max((1-alpha)*var_t + alpha*var_t*sigma0_sq_trn, VAR_T_FLOOR)

        print(f"  σ (interpretable): "
            f"img={np.sqrt(var_i):.3f}px"
            + (f", ang={np.rad2deg(np.sqrt(var_a)):.3f}°" if sigma0_sq_ang is not None else "")
            + (f", trans={1e3*np.sqrt(var_t):.3f}mm" if sigma0_sq_trn is not None else ""))

    final_intrinsics = {
        'c': intr['c'], 'kappa': intr['kappa'], 'sx': intr['sx'], 'sy': intr['sy'], 'cx': intr['cx'], 'cy': intr['cy'],
        'include_sy': include_sy
    }
    print("\nRevised Dual-Arm VCE Optimization Finished.")
    return X1_EC, T_B1B2, T_E2B, T_E1B1_list, T_B2E2_list, final_intrinsics