# -*- coding: utf-8 -*-
# "Uncertainty-Aware Hand-Eye Calibration" 논문의 부록 B를 기반으로 한
# 해석적 자코비안(Analytic Jacobian) 계산 (타겟 기반)
from __future__ import annotations
import numpy as np

def get_rotation_derivatives(alpha, beta, gamma):
    """
    논문의 식 (35-37)에 따라 R = Rx(α)Ry(β)Rz(γ)의 오일러 각에 대한
    편미분 행렬 dR/dα, dR/dβ, dR/dγ를 계산합니다.
    scipy의 ZYX 순서는 논문의 Rx(α)Ry(β)Rz(γ)와 같습니다.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    # dRx/dα
    dRx_da = np.array([[0, 0, 0], [0, -sa, -ca], [0, ca, -sa]])
    # dRy/dβ
    dRy_db = np.array([[-sb, 0, cb], [0, 0, 0], [-cb, 0, -sb]])
    # dRz/dγ
    dRz_dg = np.array([[-sg, -cg, 0], [cg, -sg, 0], [0, 0, 0]])
    
    dR_da = dRx_da @ Ry @ Rz
    dR_db = Rx @ dRy_db @ Rz
    dR_dg = Rx @ Ry @ dRz_dg

    # R_yx = Ry @ Rx
    # R_zy = Rz @ Ry
    
    # dR_da = R_zy @ dRx_da
    # dR_db = Rz @ dRy_db @ Rx
    # dR_dg = dRz_dg @ R_yx
    
    return dR_da, dR_db, dR_dg

def get_distortion_jacobian_division(pu: np.ndarray, kappa: float):
    """
    Division 모델을 기반으로 왜곡(distortion) 함수의 자코비안을 계산합니다.

    이 함수는 왜곡된 좌표(xd, yd)를 왜곡되지 않은 좌표(xu, yu)와
    왜곡 계수 kappa에 대해 편미분한 자코비안 행렬을 반환합니다.
    사용자가 제공한 사진의 수식을 기반으로 구현되었습니다.

    Args:
      pu: 2x1 크기의 numpy 배열. 왜곡되지 않은 좌표 (xu, yu)를 나타냅니다.
      kappa: 방사 왜곡 계수.

    Returns:
      J_u: 2x2 크기의 자코비안 행렬 ∂(xd,yd)/∂(xu,yu).
      J_k: 2x1 크기의 자코비안 벡터 ∂(xd,yd)/∂kappa.

    Model (Distortion):
      ru^2 = xu^2 + yu^2
      rd = (1 - sqrt(1 - 4*kappa*ru^2)) / (2*kappa*ru)
      xd = xu * (rd / ru)
      yd = yu * (rd / ru)
    """
    # ---------------------------------------------------------------------
    # 1. 초기값 및 특수 케이스 처리
    # ---------------------------------------------------------------------
    
    # kappa가 0이면 왜곡이 없으므로 변환은 항등 함수(identity)가 됩니다.
    # J_u는 단위 행렬, J_k는 영벡터가 됩니다.
    if kappa == 0.0:
        J_u = np.identity(2)
        J_k = np.zeros((2, 1))
        return J_u, J_k

    xu = float(pu[0])
    yu = float(pu[1])
    ru2 = xu * xu + yu * yu

    # 왜곡의 중심(0,0)에서는 미분값이 항등 함수와 동일합니다.
    if ru2 == 0.0:
        J_u = np.identity(2)
        J_k = np.zeros((2, 1))
        return J_u, J_k
        
    # 물리적으로 불가능한 왜곡(제곱근 안이 음수)에 대한 예외 처리
    sqrt_arg = 1.0 - 4.0 * kappa * ru2
    if sqrt_arg < 0:
        raise ValueError("왜곡이 물리적으로 불가능합니다. (kappa * ru^2 값이 너무 큽니다.)")

    # ---------------------------------------------------------------------
    # 2. 사진의 수식을 이용한 자코비안 계산
    # ---------------------------------------------------------------------

    # 수식에 반복적으로 사용되는 중간 변수들을 계산합니다.
    sqrt_term = np.sqrt(sqrt_arg)
    
    # 사진에 있는 분모 'D'
    # D = (1 + sqrt_term)^2 * sqrt_term
    D = (1.0 + sqrt_term)**2 * sqrt_term
    
    # 분모가 0이 되는 특이점(singularity) 처리
    if D == 0:
        # 이 지점에서는 미분값이 무한대로 발산합니다.
        J_u = np.full((2, 2), np.nan)
        J_k = np.full((2, 1), np.nan)
        return J_u, J_k

    # kappa에 대한 자코비안 J_k 계산
    # J_k = [ 4*xu*ru^2/D ; 4*yu*ru^2/D ]
    factor_k = (4.0 * ru2) / D
    dxd_dk = xu * factor_k
    dyd_dk = yu * factor_k
    J_k = np.array([[dxd_dk], 
                    [dyd_dk]], dtype=float)

    # (xu, yu)에 대한 자코비안 J_u 계산
    # J_u_00 = 2/(1+sqrt_term) + 8*kappa*xu^2/D
    # J_u_01 = 8*kappa*xu*yu/D
    term1 = 2.0 / (1.0 + sqrt_term)
    factor_u = (8.0 * kappa) / D
    
    dxd_dxu = term1 + factor_u * xu * xu
    dxd_dyu = factor_u * xu * yu
    dyd_dxu = dxd_dyu  # 자코비안의 비대각(off-diagonal) 성분은 동일합니다.
    dyd_dyu = term1 + factor_u * yu * yu
    
    J_u = np.array([[dxd_dxu, dxd_dyu],
                    [dyd_dxu, dyd_dyu]], dtype=float)

    return J_u, J_k


def get_distortion_jacobian_polynomial(pu: np.ndarray, dist_coeffs: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    표준 다항식 왜곡 모델(Brown-Conrady)의 자코비안을 계산합니다.
    k1, k2, p1, p2, k3 순서의 왜곡 계수를 가정합니다.

    Args:
        pu: 왜곡 전 정규 이미지 좌표 (xu, yu).
        dist_coeffs: OpenCV 형태의 왜곡 계수 배열 (k1, k2, p1, p2, k3).

    Returns:
        J_dist_u (2x2): d(pd)/d(pu) - 왜곡된 좌표를 왜곡 전 좌표로 미분한 행렬.
        J_dist_coeffs (2x5): d(pd)/d(coeffs) - 왜곡된 좌표를 왜곡 계수로 미분한 행렬.
    """
    xu, yu = pu[0], pu[1]
    
    # 왜곡 계수가 5개 미만일 경우 0으로 패딩하여 계산 안정성 확보
    coeffs = np.zeros(5)
    coeffs[:len(dist_coeffs)] = dist_coeffs.flatten()
    k1, k2, p1, p2, k3 = coeffs

    r2 = xu**2 + yu**2
    r4 = r2**2
    r6 = r2**3

    # 방사 왜곡 항
    radial_dist = (1 + k1 * r2 + k2 * r4 + k3 * r6)
    
    # --- J_dist_u: d(pd)/d(pu) 계산 ---
    d_radial_dxu = 2 * xu * (k1 + 2 * k2 * r2 + 3 * k3 * r4)
    d_radial_dyu = 2 * yu * (k1 + 2 * k2 * r2 + 3 * k3 * r4)
    
    # Tangential distortion terms' derivatives
    # d(xd)/d(xu)
    d_xd_dxu = radial_dist + xu * d_radial_dxu + 2 * p1 * yu + 6 * p2 * xu
    # d(xd)/d(yu)
    d_xd_dyu = xu * d_radial_dyu + 2 * p1 * xu + 2 * p2 * yu
    # d(yd)/d(xu)
    d_yd_dxu = yu * d_radial_dxu + 2 * p1 * yu + 2 * p2 * xu # Note the symmetry with d_xd_dyu
    # d(yd)/d(yu)
    d_yd_dyu = radial_dist + yu * d_radial_dyu + 6 * p1 * yu + 2 * p2 * xu
    
    J_dist_u = np.array([
        [d_xd_dxu, d_xd_dyu],
        [d_yd_dxu, d_yd_dyu]
    ])

    # --- J_dist_coeffs: d(pd)/d(coeffs) 계산 ---
    # 각 왜곡 계수에 대한 편미분
    d_pd_dk1 = np.array([xu * r2, yu * r2])
    d_pd_dk2 = np.array([xu * r4, yu * r4])
    d_pd_dp1 = np.array([2 * xu * yu, r2 + 2 * yu**2])
    d_pd_dp2 = np.array([r2 + 2 * xu**2, 2 * xu * yu])
    d_pd_dk3 = np.array([xu * r6, yu * r6])

    # (2, 5) 형태로 결합
    J_dist_coeffs = np.column_stack([d_pd_dk1, d_pd_dk2, d_pd_dp1, d_pd_dp2, d_pd_dk3])

    return J_dist_u, J_dist_coeffs

# --- 5 parameter polynomial ---
def distort_polynomial(pu: np.ndarray, D_poly: np.ndarray) -> np.ndarray:
    """xd, yd for polynomial model (K1,K2,K3,P1,P2)."""
    K1, K2, K3, P1, P2 = [float(x) for x in np.asarray(D_poly).ravel()[:5]]
    xu, yu = float(pu[0]), float(pu[1])
    xu2, yu2 = xu*xu, yu*yu
    r2 = xu2 + yu2
    r4, r6 = r2*r2, r2*r2*r2
    radial = 1.0 + K1*r2 + K2*r4 + K3*r6
    xuyu = xu*yu
    xd = xu*radial + 2.0*P1*xu*yu + P2*(r2 + 2.0*xu2)
    yd = yu*radial + P1*(r2 + 2.0*yu2) + 2.0*P2*xu*yu
    return np.array([xd, yd], dtype=float)


def get_distortion_param_jacobian_polynomial_numeric(
    pu: np.ndarray, D_poly: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """
    ∂(xd,yd)/∂(K1,K2,K3,P1,P2)  (2 x 5), central difference.
    """
    D_poly = np.asarray(D_poly, dtype=float).ravel()
    m = D_poly.size
    J = np.zeros((2, m), dtype=float)
    for j in range(m):
        Dp = D_poly.copy(); Dm = D_poly.copy()
        Dp[j] += eps; Dm[j] -= eps
        fp = distort_polynomial(pu, Dp)
        fm = distort_polynomial(pu, Dm)
        J[:, j] = (fp - fm) / (2.0*eps)
    return J


def get_intrinsic_jacobian(sx: float, sy: float, xd: float, yd: float) -> np.ndarray:
    """
    Table XIV (Step 6) Jacobian:
      rows:   [∂xi/·; ∂yi/·]
      cols:   [sx, sy, cx, cy, xd, yd]
    xi = xd/sx + cx,   yi = yd/sy + cy
    """
    inv_sx = 1.0 / float(sx)
    inv_sy = 1.0 / float(sy)
    inv_sx2 = inv_sx * inv_sx
    inv_sy2 = inv_sy * inv_sy

    return np.array([
        [-(float(xd)) * inv_sx2,  0.0, 1.0, 0.0,  inv_sx, 0.0],   # ∂xi/∂(...)
        [0.0, -(float(yd)) * inv_sy2, 0.0, 1.0,  0.0,    inv_sy]  # ∂yi/∂(...)
    ], dtype=float)


def calculate_analytical_jacobian_division_model(
    T_ee_cam: np.ndarray, T_base_board: np.ndarray, T_be_list: list, obj_pts_list: list,
    c: float, kappa: float, sx: float, sy: float, cx: float, cy: float,
    *,
    is_target_based: bool,
    is_scara: bool,
    # [수정] 최적화 대상을 명시적으로 제어하는 플래그
    estimate_ec: bool,
    estimate_wb: bool,
    estimate_be: bool,
    estimate_intrinsics: bool,
    include_sy: bool,
    mat_to_vec6d: None
) -> np.ndarray:
    """
    논문의 정의를 따르는 통합 자코비안 함수 (c는 고정).
    플래그에 따라 동적으로 구조가 결정되는 자코비안 행렬 A를 계산합니다.
    """
    num_poses = len(T_be_list)
    num_total_points = sum(len(pts) for pts in obj_pts_list)
    num_rows = num_total_points * 2

    # if not include_sy:
    #     sy = sx

    # --- 최종 자코비안 A의 열(column) 레이아웃 동적 계산 ---
    layout = {}
    current_col = 0
    dim_ec = 5 if is_scara else 6
    layout['ec'] = slice(current_col, current_col + dim_ec); current_col += dim_ec
    if is_target_based:
        layout['wb'] = slice(current_col, current_col + 6); current_col += 6
    if estimate_be:
        layout['be'] = slice(current_col, current_col + 6 * num_poses)
        current_col += 6 * num_poses
    
    if estimate_intrinsics:
        num_intr_params = 6 if include_sy else 5
        layout['intr'] = slice(current_col, current_col + num_intr_params)
        current_col += num_intr_params
    num_cols = current_col
    J = np.zeros((num_rows, num_cols), dtype=float)

    # --- 고정 변환 분해 ---
    R_WB, t_WB = T_base_board[:3, :3], T_base_board[:3, 3]
    e_WB = mat_to_vec6d(T_base_board)
    dR_da_WB, dR_db_WB, dR_dg_WB = get_rotation_derivatives(*e_WB[:3])

    R_EC, t_EC = T_ee_cam[:3, :3], T_ee_cam[:3, 3]
    e_EC_full = mat_to_vec6d(T_ee_cam)
    dR_da_EC, dR_db_EC, dR_dg_EC = get_rotation_derivatives(*e_EC_full[:3])

    current_row = 0
    for i in range(num_poses):
        T_BE = T_be_list[i]
        R_BE, t_BE = T_BE[:3, :3], T_BE[:3, 3]
        e_BE = mat_to_vec6d(T_BE)
        dR_da_BE, dR_db_BE, dR_dg_BE = get_rotation_derivatives(*e_BE[:3])
        
        for pw in obj_pts_list[i]:
            row_slice = slice(current_row, current_row + 2)
            
            # --- 1. 3D 좌표 변환 (Forward Kinematics) ---
            pb = R_WB @ pw + t_WB
            pt = R_BE @ pb + t_BE
            # [버그 수정] t_BE가 아닌 t_CE를 사용해야 합니다.
            pc = R_EC @ pt + t_EC
            
            if pc[2] <= 1e-9:
                J[row_slice, :] = np.nan
                current_row += 2
                print("pc error")
                continue

            # --- 2. 외부 파라미터에 대한 미분 (dpc/de*) ---
            dpc_drot_EC = np.stack([dR_da_EC @ pt, dR_db_EC @ pt, dR_dg_EC @ pt], axis=1)
            dpc_dt_EC = np.eye(3)
            dpc_de_EC_full = np.hstack((dpc_drot_EC, dpc_dt_EC))
            dpc_de_EC = dpc_de_EC_full[:,:5] if is_scara else dpc_de_EC_full

            if is_target_based:
                dpc_drot_WB = R_EC @ R_BE @ np.stack([dR_da_WB @ pw, dR_db_WB @ pw, dR_dg_WB @ pw], axis=1)
                dpc_dt_WB   = R_EC @ R_BE
                dpc_de_WB   = np.hstack((dpc_drot_WB, dpc_dt_WB))

            dpc_drot_BE = R_EC @ np.stack([dR_da_BE @ pb, dR_db_BE @ pb, dR_dg_BE @ pb], axis=1)
            dpc_dt_BE   = R_EC
            dpc_de_BE   = np.hstack((dpc_drot_BE, dpc_dt_BE))
            
            # --- 3. 카메라 모델 자코비안 (연쇄 법칙) ---
            # Step 4: Projection (pc -> pu)
            inv_Zc, inv_Zc2 = 1.0 / pc[2], 1.0 / (pc[2]**2)
            pu = c * pc[:2] * inv_Zc
            J_proj_pc = c * np.array([[inv_Zc, 0, -pc[0] * inv_Zc2],
                                      [0, inv_Zc, -pc[1] * inv_Zc2]])
            J_proj_c = pc[:2] * inv_Zc

            # Step 5: Distortion (pu -> pd)
            J_dist_u, J_dist_k = get_distortion_jacobian_division(pu, kappa)
            
            # Step 6을 위해 xd, yd 계산
            ru2 = pu[0]**2 + pu[1]**2
            xd, yd = 0.0, 0.0
            if kappa == 0.0: xd, yd = pu[0], pu[1]
            else:
                sqrt_arg = 1.0 - 4.0 * kappa * ru2
                if sqrt_arg >= 0:
                    ru = np.sqrt(ru2)
                    if ru > 1e-9:
                        rd = (1.0 - np.sqrt(sqrt_arg)) / (2.0 * kappa * ru)
                        scale = rd / ru
                        xd, yd = pu[0] * scale, pu[1] * scale
                else: xd, yd = np.nan, np.nan

            J_map_full = get_intrinsic_jacobian(sx, sy, xd, yd)
            J_map_d = J_map_full[:, 4:]

            # --- 4. 최종 자코비안 채우기 ---
            J_cam = J_map_d @ J_dist_u @ J_proj_pc

            J[row_slice, layout['ec']] = J_cam @ dpc_de_EC
            if is_target_based:
                J[row_slice, layout['wb']] = J_cam @ dpc_de_WB
            
            if estimate_be:
                # 현재 포즈 i에 해당하는 열 블록에만 값을 할당
                col_be_i = slice(layout['be'].start + 6*i, layout['be'].start + 6*(i+1))
                J[row_slice, col_be_i] = J_cam @ dpc_de_BE

            # col_be = slice(layout['be'].start + 6*i, layout['be'].start + 6*(i+1))
            # J[row_slice, col_be] = J_cam @ dpc_de_BE

            if estimate_intrinsics:
                intr_slice = layout['intr']
                J_c_chain = J_map_d @ J_dist_u @ J_proj_c
                J_kappa_chain = J_map_d @ J_dist_k
                J_map_intr = J_map_full[:, :4]
                
                if include_sy:
                    # 상태 벡터 순서: c, κ, sx, sy, cx, cy
                    J[row_slice, intr_slice] = np.hstack([J_c_chain[:,None], J_kappa_chain, J_map_intr[:,[0,1,2,3]]])
                else:
                    # 상태 벡터 순서: c, κ, sx, cx, cy
                    J[row_slice, intr_slice] = np.hstack([J_c_chain[:,None], J_kappa_chain, J_map_intr[:,[0,2,3]]])

            current_row += 2
    return J


def get_distortion_jacobian_rational(pu: np.ndarray, dist_coeffs: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Azure Kinect / OpenCV의 Rational Function 왜곡 모델의 자코비안을 계산합니다.
    k1, k2, p1, p2, k3, k4, k5, k6 순서의 8개 왜곡 계수를 가정합니다.

    Args:
        pu: 왜곡 전 정규 이미지 좌표 (xu, yu).
        dist_coeffs: OpenCV 형태의 8-계수 배열.

    Returns:
        J_dist_u (2x2): d(pd)/d(pu) - 왜곡된 좌표를 왜곡 전 좌표로 미분한 행렬.
        J_dist_coeffs (2x8): d(pd)/d(coeffs) - 왜곡된 좌표를 왜곡 계수로 미분한 행렬.
    """
    xu, yu = pu[0], pu[1]
    
    # 왜곡 계수가 8개 미만일 경우 0으로 패딩
    coeffs = np.zeros(8)
    coeffs[:len(dist_coeffs)] = dist_coeffs.flatten()
    k1, k2, p1, p2, k3, k4, k5, k6 = coeffs

    r2 = xu**2 + yu**2
    r4 = r2**2
    r6 = r4 * r2

    # Rational 모델의 분자(a)와 분모(b)
    a = 1 + k1 * r2 + k2 * r4 + k3 * r6
    b = 1 + k4 * r2 + k5 * r4 + k6 * r6
    
    # 0으로 나누는 것을 방지
    b_inv = 1.0 / b if abs(b) > 1e-9 else 1.0
    d = a * b_inv

    # --- J_dist_u: d(pd)/d(pu) 계산 ---
    # d(a)/d(r^2), d(b)/d(r^2)
    da_dr2 = k1 + 2 * k2 * r2 + 3 * k3 * r4
    db_dr2 = k4 + 2 * k5 * r2 + 3 * k6 * r4
    
    # d(d)/d(r^2) using quotient rule: (f'g - fg') / g^2
    dd_dr2 = (da_dr2 * b - a * db_dr2) * (b_inv**2)
    
    # d(d)/d(xu), d(d)/d(yu) using chain rule
    dd_dxu = dd_dr2 * (2 * xu)
    dd_dyu = dd_dr2 * (2 * yu)
    
    # 최종 편미분 계산
    # d(xd)/d(xu)
    d_xd_dxu = d + xu * dd_dxu + 2 * p1 * yu + 6 * p2 * xu
    # d(xd)/d(yu)
    d_xd_dyu = xu * dd_dyu + 2 * p1 * xu + 2 * p2 * yu
    # d(yd)/d(xu)
    d_yd_dxu = yu * dd_dxu + 2 * p1 * xu + 2 * p2 * yu
    # d(yd)/d(yu)
    d_yd_dyu = d + yu * dd_dyu + 6 * p1 * yu + 2 * p2 * xu
    
    J_dist_u = np.array([
        [d_xd_dxu, d_xd_dyu],
        [d_yd_dxu, d_yd_dyu]
    ])

    # --- J_dist_coeffs: d(pd)/d(coeffs) 계산 ---
    # 각 왜곡 계수에 대한 편미분
    d_pd_dk1 = np.array([xu * r2 * b_inv,   yu * r2 * b_inv])
    d_pd_dk2 = np.array([xu * r4 * b_inv,   yu * r4 * b_inv])
    d_pd_dk3 = np.array([xu * r6 * b_inv,   yu * r6 * b_inv])
    
    # 분모 계수에 대한 미분 (체인룰: d/dk4 = (d/db) * (db/dk4))
    d_d_db = -a * (b_inv**2)
    d_pd_dk4 = np.array([xu * d_d_db * r2,  yu * d_d_db * r2])
    d_pd_dk5 = np.array([xu * d_d_db * r4,  yu * d_d_db * r4])
    d_pd_dk6 = np.array([xu * d_d_db * r6,  yu * d_d_db * r6])
    
    d_pd_dp1 = np.array([2 * xu * yu,     r2 + 2 * yu**2])
    d_pd_dp2 = np.array([r2 + 2 * xu**2,  2 * xu * yu])

    # (2, 8) 형태로 결합
    J_dist_coeffs = np.column_stack([
        d_pd_dk1, d_pd_dk2, d_pd_dp1, d_pd_dp2, 
        d_pd_dk3, d_pd_dk4, d_pd_dk5, d_pd_dk6
    ])

    return J_dist_u, J_dist_coeffs


def calculate_analytical_jacobian_polynomial_model(
    T_ee_cam: np.ndarray, T_base_board: np.ndarray, T_be_list: list, obj_pts_list: list,
    # --- 표준 카메라 파라미터를 직접 입력받음 ---
    fx: float, fy: float, cx: float, cy: float,
    dist_coeffs: np.ndarray,
    # -----------------------------------------
    *,
    is_target_based: bool,
    is_scara: bool,
    estimate_ec: bool,
    estimate_wb: bool,
    estimate_be: bool,
    estimate_intrinsics: bool,
    mat_to_vec6d=None,
) -> np.ndarray:
    """
    [최종 버전] 표준 다항식(Brown-Conrad) 모델과 fx, fy 파라미터를
    사용하는 해석적 자코비안 함수입니다.
    """
    num_poses = len(T_be_list)
    num_total_points = sum(len(pts) for pts in obj_pts_list)
    num_rows = num_total_points * 2

    # --- 1. 자코비안 행렬 레이아웃 계산 ---
    layout = {}
    current_col = 0
    dim_ec = 5 if is_scara else 6
    if estimate_ec:
        layout['ec'] = slice(current_col, current_col + dim_ec); current_col += dim_ec
    if estimate_wb:
        layout['wb'] = slice(current_col, current_col + 6); current_col += 6
    if estimate_be:
        layout['be'] = slice(current_col, current_col + 6 * num_poses); current_col += 6 * num_poses
    
    if estimate_intrinsics:
        # 상태 벡터: fx, fy, cx, cy + 왜곡 계수들
        num_intr_params = 4 + len(dist_coeffs.flatten())
        layout['intr'] = slice(current_col, current_col + num_intr_params)
        current_col += num_intr_params
    num_cols = current_col
    J = np.zeros((num_rows, num_cols), dtype=float)

    # --- 2. 외부 파라미터 분해 ---
    R_WB, t_WB = T_base_board[:3, :3], T_base_board[:3, 3]
    e_WB = mat_to_vec6d(T_base_board)
    dR_da_WB, dR_db_WB, dR_dg_WB = get_rotation_derivatives(*e_WB[:3])
    R_EC, t_EC = T_ee_cam[:3, :3], T_ee_cam[:3, 3]
    e_EC_full = mat_to_vec6d(T_ee_cam)
    dR_da_EC, dR_db_EC, dR_dg_EC = get_rotation_derivatives(*e_EC_full[:3])

    current_row = 0
    for i in range(num_poses):
        T_BE = T_be_list[i]
        R_BE, t_BE = T_BE[:3, :3], T_BE[:3, 3]
        e_BE = mat_to_vec6d(T_BE)
        dR_da_BE, dR_db_BE, dR_dg_BE = get_rotation_derivatives(*e_BE[:3])
        
        for pw in obj_pts_list[i]:
            row_slice = slice(current_row, current_row + 2)
            
            # --- 3. 3D 좌표 변환 (모든 모델 공통) ---
            pb = R_WB @ pw + t_WB
            pt = R_BE @ pb + t_BE
            pc = R_EC @ pt + t_EC
            
            if pc[2] <= 1e-9:
                J[row_slice, :] = np.nan; current_row += 2; continue

            # --- 4. 외부 파라미터에 대한 미분 (모든 모델 공통) ---
            dpc_drot_EC = np.stack([dR_da_EC @ pt, dR_db_EC @ pt, dR_dg_EC @ pt], axis=1)
            dpc_dt_EC = np.eye(3)
            dpc_de_EC_full = np.hstack((dpc_drot_EC, dpc_dt_EC))
            dpc_de_EC = dpc_de_EC_full[:,:5] if is_scara else dpc_de_EC_full
            
            if is_target_based and estimate_wb:
                dpc_drot_WB = R_EC @ R_BE @ np.stack([dR_da_WB @ pw, dR_db_WB @ pw, dR_dg_WB @ pw], axis=1)
                dpc_dt_WB   = R_EC @ R_BE
                dpc_de_WB   = np.hstack((dpc_drot_WB, dpc_dt_WB))
            
            dpc_drot_BE = R_EC @ np.stack([dR_da_BE @ pb, dR_db_BE @ pb, dR_dg_BE @ pb], axis=1)
            dpc_dt_BE   = R_EC
            dpc_de_BE   = np.hstack((dpc_drot_BE, dpc_dt_BE))
            
            # --- 5. 표준 카메라 모델 자코비안 계산 ---
            inv_Zc, inv_Zc2 = 1.0 / pc[2], 1.0 / (pc[2]**2)
            pu = pc[:2] * inv_Zc # 정규 이미지 좌표 (xu, yu)
            J_proj_pc = np.array([[inv_Zc, 0, -pc[0] * inv_Zc2],
                                  [0, inv_Zc, -pc[1] * inv_Zc2]])

            J_dist_u, J_dist_coeffs = get_distortion_jacobian_rational(pu, dist_coeffs)
            
            # # Forward pass to get distorted point `pd`
            # r2 = pu[0]**2 + pu[1]**2; r4 = r2**2; r6 = r2**3
            # coeffs = np.zeros(5); coeffs[:len(dist_coeffs)] = dist_coeffs.flatten()
            # k1,k2,p1,p2,k3 = coeffs
            # radial = (1 + k1*r2 + k2*r4 + k3*r6)
            # xd = pu[0]*radial + (2*p1*pu[0]*pu[1] + p2*(r2 + 2*pu[0]**2))
            # yd = pu[1]*radial + (p1*(r2 + 2*pu[1]**2) + 2*p2*pu[0]*pu[1])

            # ***** 핵심 수정 사항: 순방향 계산(Forward pass)을 Rational 모델로 교체 *****
            
            # 8개 왜곡 계수 언패킹
            coeffs = np.zeros(8)
            coeffs[:len(dist_coeffs)] = dist_coeffs.flatten()
            k1, k2, p1, p2, k3, k4, k5, k6 = coeffs

            # 필요한 r의 거듭제곱 계산
            r2 = pu[0]**2 + pu[1]**2
            r4 = r2**2
            r6 = r4 * r2
            
            # Rational 모델의 분자(a)와 분모(b)
            a = 1 + k1 * r2 + k2 * r4 + k3 * r6
            b = 1 + k4 * r2 + k5 * r4 + k6 * r6
            
            # 방사 왜곡 인자 d 계산
            d = a / b if abs(b) > 1e-9 else a
            
            # 왜곡된 정규 좌표 pd = [xd, yd] 계산
            xd = pu[0]*d + (2*p1*pu[0]*pu[1] + p2*(r2 + 2*pu[0]**2))
            yd = pu[1]*d + (p1*(r2 + 2*pu[1]**2) + 2*p2*pu[0]*pu[1])
            
            # *******************************************************************

            # Mapping Jacobian
            J_map_d = np.array([[fx, 0], [0, fy]])
            # Jacobian wrt intrinsics: d(uv)/d(fx, fy, cx, cy)
            J_map_intr = np.array([[xd, 0, 1, 0], [0, yd, 0, 1]])

            # --- 6. 최종 자코비안 조립 (연쇄 법칙) ---
            J_cam_pc = J_map_d @ J_dist_u @ J_proj_pc
            
            if estimate_ec:
                J[row_slice, layout['ec']] = J_cam_pc @ dpc_de_EC
            if is_target_based and estimate_wb:
                J[row_slice, layout['wb']] = J_cam_pc @ dpc_de_WB
            if estimate_be:
                col_be = slice(layout['be'].start + 6*i, layout['be'].start + 6*(i+1))
                J[row_slice, col_be] = J_cam_pc @ dpc_de_BE

            if estimate_intrinsics:
                # 상태 벡터 순서: fx, fy, cx, cy, k1, k2, ...
                J_dist_chain = J_map_d @ J_dist_coeffs
                J[row_slice, layout['intr']] = np.hstack([J_map_intr, J_dist_chain])

            current_row += 2
            
    return J


def calculate_analytical_jacobian_division_model_dual(
    # Global (estimate flags로 on/off)
    X1_EC: np.ndarray,        # ^C1 T_E1
    T_B1B2: np.ndarray,       # ^B1 T_B2
    E2_C2: np.ndarray,        # ^E2 T_C2 (= inv(^C2 T_E2))
    *,
    # Per-pose (모두 상태/관측 동일 프레임)
    T_E1B1_list: list,        # [ ^E1 T_B1 ]_i
    T_E2B2_list: list,        # [ ^E2 T_B2 ]_i   # ← 입력/상태는 E2B2
    T_C2B_list: list,         # [ ^C2 T_B  ]_i
    obj_pts_list: list,       # [ (Ni,3) ]_i
    # Division intrinsics
    c: float, kappa: float, sx: float, sy: float, cx: float, cy: float,
    # Flags
    estimate_x1ec: bool = True,
    estimate_e1b1: bool = True,
    estimate_b1b2: bool = True,
    estimate_b2e2: bool = True,    # ← 키 이름은 호환을 위해 유지(내용은 E2B2)
    estimate_e2c2: bool = True,
    estimate_c2b: bool = False,
    estimate_intrinsics: bool = True,
    include_sy: bool = False,
    is_scara_x1: bool = False,
    mat_to_vec6d=None
) -> np.ndarray:
    """
    체인(단일 카메라 잔차):
      B --(C2B_i)--> C2 --(E2C2)--> E2 --(B2E2)--> B2 --(B1B2)--> B1 --(E1B1_i)--> E1 --(X1EC)--> C1
    단, 상태/입력은 ^E2T_B2 (E2B2)이며, 체인에서는 B2E2 = inv(E2B2)를 사용.
    열 순서: [X1_EC, B1B2, E2_C2, (E1B1_i), (B2E2_i ← E2B2의 inverse rule), (C2B_i), intr]
    """
    assert mat_to_vec6d is not None, "mat_to_vec6d 등을 넘겨주세요."

    if not include_sy:
        sy = sx  # sy= sx로 묶음

    num_poses = len(obj_pts_list)
    assert len(T_E1B1_list) == len(T_E2B2_list) == len(T_C2B_list) == num_poses

    # ---- 열 레이아웃 ----
    layout = {}
    col = 0
    dim_x1 = 5 if is_scara_x1 else 6
    if estimate_x1ec: layout['x1ec'] = slice(col, col + dim_x1); col += dim_x1
    if estimate_b1b2: layout['b1b2'] = slice(col, col + 6); col += 6
    if estimate_e2c2: layout['e2c2'] = slice(col, col + 6); col += 6
    if estimate_e1b1: layout['e1b1'] = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_b2e2: layout['b2e2'] = slice(col, col + 6 * num_poses); col += 6 * num_poses  # ← 키 유지
    if estimate_c2b:  layout['c2b']  = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_intrinsics:
        n_intr = 6 if include_sy else 5
        layout['intr'] = slice(col, col + n_intr); col += n_intr

    num_cols = col
    num_rows = 2 * sum(len(pts) for pts in obj_pts_list)
    J = np.zeros((num_rows, num_cols), dtype=float)

    # ---- 회전 미분 유틸 ----
    def rot_derivs_from_T(T):
        a, b, g, *_ = mat_to_vec6d(T)
        return get_rotation_derivatives(a, b, g)

    # 글로벌 파트
    dRa_x1, dRb_x1, dRg_x1         = rot_derivs_from_T(X1_EC)
    dRa_b12, dRb_b12, dRg_b12      = rot_derivs_from_T(T_B1B2)
    dRa_e2c2, dRb_e2c2, dRg_e2c2   = rot_derivs_from_T(E2_C2)
    R_x1,  t_x1  = X1_EC[:3,:3],  X1_EC[:3,3]
    R_b12, t_b12 = T_B1B2[:3,:3], T_B1B2[:3,3]
    R_e2c2, t_e2c2 = E2_C2[:3,:3], E2_C2[:3,3]

    # ---- 투영/왜곡/내부 자코비안 블록 ----
    def _proj_blocks(pc, c, kappa, sx, sy, cx, cy):
        if pc[2] <= 1e-12: return None
        invZ  = 1.0 / pc[2]; invZ2 = invZ * invZ
        pu = c * pc[:2] * invZ
        J_proj_pc = c * np.array([[invZ, 0.0, -pc[0]*invZ2],
                                  [0.0, invZ, -pc[1]*invZ2]], float)
        J_proj_c  = (pc[:2] * invZ).reshape(2,1)
        J_dist_u, J_dist_k = get_distortion_jacobian_division(pu, kappa)

        ux, uy = float(pu[0]), float(pu[1]); ru2 = ux*ux + uy*uy
        if abs(kappa) < 1e-15 or ru2 < 1e-24:
            xd, yd = ux, uy
        else:
            g = 1.0 - 4.0*kappa*ru2
            if g <= 0.0: return None
            Delta = np.sqrt(g); ru = np.sqrt(ru2)
            rd = (1.0 - Delta) / (2.0*kappa*ru)
            s = rd / (ru + 1e-12)
            xd, yd = s*ux, s*uy

        J_map_full = get_intrinsic_jacobian(sx, sy, xd, yd)
        J_map_intr = J_map_full[:, :4]
        J_map_d    = J_map_full[:, 4:]
        J_cam = J_map_d @ J_dist_u @ J_proj_pc
        J_c   = J_map_d @ J_dist_u @ J_proj_c
        J_k   = J_map_d @ J_dist_k
        return (J_cam, J_c, J_k, J_map_intr)

    row = 0
    for i in range(num_poses):
        T_c2b  = T_C2B_list[i];  R_c2b,  t_c2b  = T_c2b[:3,:3],  T_c2b[:3,3]
        T_e2b2 = T_E2B2_list[i]; R_e2b2, t_e2b2 = T_e2b2[:3,:3], T_e2b2[:3,3]
        T_e1b1 = T_E1B1_list[i]; R_e1b1, t_e1b1 = T_e1b1[:3,:3], T_e1b1[:3,3]

        # B2E2 = inv(E2B2)
        R_b2e2 = R_e2b2.T
        t_b2e2 = -R_e2b2.T @ t_e2b2

        # per-pose 회전 미분 (E2B2에 대해 직접 미분 → inverse 규칙에 들어감)
        dRa_c2b,  dRb_c2b,  dRg_c2b  = rot_derivs_from_T(T_c2b)
        dRa_e1b1, dRb_e1b1, dRg_e1b1 = rot_derivs_from_T(T_e1b1)
        dRa_e2b2, dRb_e2b2, dRg_e2b2 = rot_derivs_from_T(T_e2b2)

        for pw in obj_pts_list[i]:
            s0 = pw.astype(float)
            s1 = R_c2b  @ s0 + t_c2b
            s2 = R_e2c2 @ s1 + t_e2c2
            s3 = R_b2e2 @ s2 + t_b2e2     # ← inv(E2B2)
            s4 = R_b12  @ s3 + t_b12
            s5 = R_e1b1 @ s4 + t_e1b1
            pc = R_x1   @ s5 + t_x1

            blocks = _proj_blocks(pc, c, kappa, sx, sy, cx, cy)
            if blocks is None:
                J[row:row+2, :] = np.nan
                row += 2
                continue
            J_cam, J_c, J_k, J_map_intr = blocks

            # helper: 특정 변환 블록의 6D(abc,tx,ty,tz) 기여 채우기
            def _fill_rot_block(dRa, dRb, dRg, s_k, Rprefix, col_slice):
                dpc_drot = np.column_stack((Rprefix @ (dRa @ s_k),
                                            Rprefix @ (dRb @ s_k),
                                            Rprefix @ (dRg @ s_k)))
                dpc = np.hstack((dpc_drot, Rprefix))
                J[row:row+2, col_slice] = J_cam @ dpc

            # X1_EC
            if estimate_x1ec:
                if is_scara_x1:
                    dpc_drot = np.column_stack((dRa_x1 @ s5, dRb_x1 @ s5, dRg_x1 @ s5))
                    dpc_full = np.hstack((dpc_drot, np.eye(3)))
                    J[row:row+2, layout['x1ec']] = (J_cam @ dpc_full)[:, :5]
                else:
                    _fill_rot_block(dRa_x1, dRb_x1, dRg_x1, s5, np.eye(3), layout['x1ec'])

            # E1B1_i
            if estimate_e1b1:
                cs = slice(layout['e1b1'].start + 6*i, layout['e1b1'].start + 6*(i+1))
                _fill_rot_block(dRa_e1b1, dRb_e1b1, dRg_e1b1, s4, R_x1, cs)

            # B1B2
            if estimate_b1b2:
                _fill_rot_block(dRa_b12, dRb_b12, dRg_b12, s3, R_x1 @ R_e1b1, layout['b1b2'])

            # E2B2 (체인에는 B2E2=inv(E2B2)가 들어가므로 inverse rule 적용)
            if estimate_b2e2:
                cs = slice(layout['b2e2'].start + 6*i, layout['b2e2'].start + 6*(i+1))
                Rprefix = R_x1 @ R_e1b1 @ R_b12
                # d pc / d (E2B2) :
                #   회전: Rprefix @ (dR_e2b2^T @ (s2 - t_e2b2))
                #   평행이동: Rprefix @ (-R_e2b2^T)
                dpc_drot = np.column_stack((
                    Rprefix @ (dRa_e2b2.T @ (s2 - t_e2b2)),
                    Rprefix @ (dRb_e2b2.T @ (s2 - t_e2b2)),
                    Rprefix @ (dRg_e2b2.T @ (s2 - t_e2b2)),
                ))
                dpc_dtrn = Rprefix @ (-R_e2b2.T)
                dpc = np.hstack((dpc_drot, dpc_dtrn))
                J[row:row+2, cs] = J_cam @ dpc

            # E2_C2
            if estimate_e2c2:
                _fill_rot_block(dRa_e2c2, dRb_e2c2, dRg_e2c2, s1, R_x1 @ R_e1b1 @ R_b12 @ R_b2e2, layout['e2c2'])

            # C2B_i
            if estimate_c2b:
                cs = slice(layout['c2b'].start + 6*i, layout['c2b'].start + 6*(i+1))
                _fill_rot_block(dRa_c2b, dRb_c2b, dRg_c2b, s0, R_x1 @ R_e1b1 @ R_b12 @ R_b2e2 @ R_e2c2, cs)

            # Intrinsics
            if estimate_intrinsics:
                if include_sy:
                    # [c, kappa, sx, sy, cx, cy]
                    J[row:row+2, layout['intr']] = np.hstack([J_c, J_k, J_map_intr[:, [0,1,2,3]]])
                else:
                    # sy==sx 묶음 → [c, kappa, sx, cx, cy]
                    J[row:row+2, layout['intr']] = np.hstack([J_c, J_k, J_map_intr[:, [0,2,3]]])

            row += 2

    return J

def calculate_analytical_jacobian_division_model_dual_bicamera(
    # Global
    X1_EC: np.ndarray,        # ^C1 T_E1
    T_B1B2: np.ndarray,       # ^B1 T_B2
    E2_C2: np.ndarray,        # ^E2 T_C2
    *,
    # Per-pose (공유)
    T_E1B1_list: list,        # [ ^E1 T_B1 ]_i
    T_E2B2_list: list,        # [ ^E2 T_B2 ]_i
    # 보드 관측(옵션)
    T_C2B_list: list,         # [ ^C2 T_B ]_i  (Cam1 블록에서 앵커)
    T_C1B_list: list,         # [ ^C1 T_B ]_i  (Cam2 블록에서 앵커)
    # 포인트
    obj_pts_list: list,       # [ (Ni,3) ]_i
    #
    # Cam1 intrinsics (division)
    c1: float, kappa1: float, sx1: float, sy1: float, cx1: float, cy1: float,
    # Cam2 intrinsics (division)
    c2: float, kappa2: float, sx2: float, sy2: float, cx2: float, cy2: float,
    #
    # Flags
    estimate_x1ec: bool = True,
    estimate_b1b2: bool = True,
    estimate_e2c2: bool = True,
    estimate_e1b1: bool = True,
    estimate_b2e2: bool = True,
    estimate_c2b:  bool = False,
    estimate_c1b:  bool = False,
    estimate_intr1: bool = True,
    estimate_intr2: bool = True,
    include_sy1: bool = False,
    include_sy2: bool = False,
    is_scara_x1: bool = False,
    mat_to_vec6d=None
) -> tuple[np.ndarray, dict]:
    """
    (Docstring은 기존과 동일)
    """
    assert mat_to_vec6d is not None
    num_poses = len(obj_pts_list)

    # --- 공통 유틸리티 및 레이아웃 (기존과 동일) ---
    def rot_derivs_from_T(T):
        a, b, g, *_ = mat_to_vec6d(T)
        return get_rotation_derivatives(a, b, g)

    layout = {}
    col = 0
    dim_x1 = 5 if is_scara_x1 else 6
    if estimate_x1ec: layout['x1ec'] = slice(col, col + dim_x1); col += dim_x1
    if estimate_b1b2: layout['b1b2'] = slice(col, col + 6); col += 6
    if estimate_e2c2: layout['e2c2'] = slice(col, col + 6); col += 6
    if estimate_e1b1: layout['e1b1'] = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_b2e2: layout['b2e2'] = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_c2b:  layout['c2b']  = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_c1b:  layout['c1b']  = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_intr1:
        n1 = 6 if include_sy1 else 5
        layout['intr1'] = slice(col, col + n1); col += n1
    if estimate_intr2:
        n2 = 6 if include_sy2 else 5
        layout['intr2'] = slice(col, col + n2); col += n2
    layout['total'] = col

    n_cam1_rows = 2 * sum(len(pts) for pts in obj_pts_list)
    n_cam2_rows = 2 * sum(len(pts) for pts in obj_pts_list)
    J = np.zeros((n_cam1_rows + n_cam2_rows, layout['total']), dtype=float)

    def _proj_blocks(pc, c, kappa, sx, sy, cx, cy):
        if pc[2] <= 1e-12: return None
        invZ = 1.0 / pc[2]; invZ2 = invZ * invZ
        pu = c * pc[:2] * invZ
        J_proj_pc = c * np.array([[invZ, 0.0, -pc[0] * invZ2], [0.0, invZ, -pc[1] * invZ2]], float)
        J_proj_c = (pc[:2] * invZ).reshape(2, 1)
        J_dist_u, J_dist_k = get_distortion_jacobian_division(pu, kappa)
        ux, uy = float(pu[0]), float(pu[1]); ru2 = ux * ux + uy * uy
        if abs(kappa) < 1e-15 or ru2 < 1e-24:
            xd, yd = ux, uy
        else:
            g = 1.0 - 4.0 * kappa * ru2
            if g <= 0.0: return None
            Delta = np.sqrt(g); ru = np.sqrt(ru2)
            rd = (1.0 - Delta) / (2.0 * kappa * ru); s = rd / (ru + 1e-12)
            xd, yd = s * ux, s * uy
        J_map_full = get_intrinsic_jacobian(sx, sy, xd, yd)
        J_map_intr = J_map_full[:, :4]; J_map_d = J_map_full[:, 4:]
        J_cam = J_map_d @ J_dist_u @ J_proj_pc
        J_c   = J_map_d @ J_dist_u @ J_proj_c; J_k = J_map_d @ J_dist_k
        return (J_cam, J_c, J_k, J_map_intr)

    # =========================================================
    # 1) Cam1 잔차 블록:  B → C2 → E2 → B2E2 → B1 → E1 → C1
    # =========================================================
    row = 0
    R_x1, t_x1 = X1_EC[:3,:3], X1_EC[:3,3]
    R_b12, t_b12 = T_B1B2[:3,:3], T_B1B2[:3,3]
    R_e2c2, t_e2c2 = E2_C2[:3,:3], E2_C2[:3,3]
    dRa_x1, dRb_x1, dRg_x1 = rot_derivs_from_T(X1_EC)
    dRa_b12, dRb_b12, dRg_b12 = rot_derivs_from_T(T_B1B2)
    dRa_e2c2, dRb_e2c2, dRg_e2c2 = rot_derivs_from_T(E2_C2)

    for i in range(num_poses):
        T_c2b = T_C2B_list[i]; R_c2b, t_c2b = T_c2b[:3,:3], T_c2b[:3,3]
        T_e1b1 = T_E1B1_list[i]; R_e1b1, t_e1b1 = T_e1b1[:3,:3], T_e1b1[:3,3]
        T_e2b2 = T_E2B2_list[i]; R_e2b2, t_e2b2 = T_e2b2[:3,:3], T_e2b2[:3,3]
        R_b2e2, t_b2e2 = R_e2b2.T, -R_e2b2.T @ t_e2b2 # B2E2 = inv(E2B2)
        dRa_c2b, dRb_c2b, dRg_c2b = rot_derivs_from_T(T_c2b)
        dRa_e1b1, dRb_e1b1, dRg_e1b1 = rot_derivs_from_T(T_e1b1)
        dRa_e2b2, dRb_e2b2, dRg_e2b2 = rot_derivs_from_T(T_e2b2)
        
        for pw in obj_pts_list[i]:
            s0 = pw.astype(float)
            s1 = R_c2b @ s0 + t_c2b
            s2 = R_e2c2 @ s1 + t_e2c2
            s3 = R_b2e2 @ s2 + t_b2e2
            s4 = R_b12 @ s3 + t_b12
            s5 = R_e1b1 @ s4 + t_e1b1
            pc = R_x1 @ s5 + t_x1

            blocks = _proj_blocks(pc, c1, kappa1, sx1, sy1, cx1, cy1)
            if blocks is None: J[row:row+2,:] = np.nan; row += 2; continue
            J_cam, J_c, J_k, J_map_intr = blocks

            if estimate_x1ec:
                dpc_drot = np.column_stack([dRa_x1 @ s5, dRb_x1 @ s5, dRg_x1 @ s5])
                dpc_dTx1 = np.hstack([dpc_drot, np.eye(3)])
                J[row:row+2, layout['x1ec']] = J_cam @ (dpc_dTx1[:,:5] if is_scara_x1 else dpc_dTx1)
            if estimate_e1b1:
                R_prefix = R_x1
                dpc_drot = np.column_stack([R_prefix @ dRa_e1b1 @ s4, R_prefix @ dRb_e1b1 @ s4, R_prefix @ dRg_e1b1 @ s4])
                dpc_dTe1b1 = np.hstack([dpc_drot, R_prefix])
                J[row:row+2, slice(layout['e1b1'].start + 6*i, layout['e1b1'].start + 6*(i+1))] = J_cam @ dpc_dTe1b1
            if estimate_b1b2:
                R_prefix = R_x1 @ R_e1b1
                dpc_drot = np.column_stack([R_prefix @ dRa_b12 @ s3, R_prefix @ dRb_b12 @ s3, R_prefix @ dRg_b12 @ s3])
                dpc_dTb12 = np.hstack([dpc_drot, R_prefix])
                J[row:row+2, layout['b1b2']] = J_cam @ dpc_dTb12
            if estimate_b2e2:
                R_prefix = R_x1 @ R_e1b1 @ R_b12
                dpc_drot = np.column_stack([ R_prefix @ (dRa_e2b2.T @ (s2 - t_e2b2)),
                                             R_prefix @ (dRb_e2b2.T @ (s2 - t_e2b2)),
                                             R_prefix @ (dRg_e2b2.T @ (s2 - t_e2b2)) ])
                dpc_dtrans = R_prefix @ (-R_e2b2.T)
                dpc_dTe2b2 = np.hstack([dpc_drot, dpc_dtrans])
                J[row:row+2, slice(layout['b2e2'].start + 6*i, layout['b2e2'].start + 6*(i+1))] = J_cam @ dpc_dTe2b2
            if estimate_e2c2:
                R_prefix = R_x1 @ R_e1b1 @ R_b12 @ R_b2e2
                dpc_drot = np.column_stack([R_prefix @ dRa_e2c2 @ s1, R_prefix @ dRb_e2c2 @ s1, R_prefix @ dRg_e2c2 @ s1])
                dpc_dTe2c2 = np.hstack([dpc_drot, R_prefix])
                J[row:row+2, layout['e2c2']] = J_cam @ dpc_dTe2c2
            if estimate_c2b:
                R_prefix = R_x1 @ R_e1b1 @ R_b12 @ R_b2e2 @ R_e2c2
                dpc_drot = np.column_stack([R_prefix @ dRa_c2b @ s0, R_prefix @ dRb_c2b @ s0, R_prefix @ dRg_c2b @ s0])
                dpc_dTc2b = np.hstack([dpc_drot, R_prefix])
                J[row:row+2, slice(layout['c2b'].start + 6*i, layout['c2b'].start + 6*(i+1))] = J_cam @ dpc_dTc2b
            if estimate_intr1:
                J[row:row+2, layout['intr1']] = np.hstack([J_c, J_k, J_map_intr[:,[0,1,2,3] if include_sy1 else [0,2,3]]])
            row += 2

    # =====================================================================
    # 2) Cam2 잔차 블록: B→C1→inv(X1)→inv(E1B1)→inv(B1B2)→E2B2→inv(E2C2)
    # =====================================================================
    for i in range(num_poses):
        T_c1b = T_C1B_list[i]; R_c1b, t_c1b = T_c1b[:3,:3], T_c1b[:3,3]
        T_e1b1 = T_E1B1_list[i]; R_e1b1, t_e1b1 = T_e1b1[:3,:3], T_e1b1[:3,3]
        T_e2b2 = T_E2B2_list[i]; R_e2b2, t_e2b2 = T_e2b2[:3,:3], T_e2b2[:3,3]
        dRa_c1b, dRb_c1b, dRg_c1b = rot_derivs_from_T(T_c1b)
        dRa_e1b1, dRb_e1b1, dRg_e1b1 = rot_derivs_from_T(T_e1b1)
        dRa_e2b2, dRb_e2b2, dRg_e2b2 = rot_derivs_from_T(T_e2b2)
        
        RT_x1, t_inv_x1 = R_x1.T, -R_x1.T @ t_x1
        RT_e1b1, t_inv_e1b1 = R_e1b1.T, -R_e1b1.T @ t_e1b1
        RT_b12, t_inv_b12 = R_b12.T, -R_b12.T @ t_b12
        RT_e2c2, t_inv_e2c2 = R_e2c2.T, -R_e2c2.T @ t_e2c2

        for pw in obj_pts_list[i]:
            q0 = pw.astype(float)
            q1 = R_c1b @ q0 + t_c1b
            q2 = RT_x1 @ q1 + t_inv_x1
            q3 = RT_e1b1 @ q2 + t_inv_e1b1
            q4 = RT_b12 @ q3 + t_inv_b12
            q5 = R_e2b2 @ q4 + t_e2b2
            pc2 = RT_e2c2 @ q5 + t_inv_e2c2
            
            blocks = _proj_blocks(pc2, c2, kappa2, sx2, sy2, cx2, cy2)
            if blocks is None: J[row:row+2,:] = np.nan; row += 2; continue
            J_cam, J_c, J_k, J_map_intr = blocks

            if estimate_c1b:
                R_prefix = RT_e2c2 @ R_e2b2 @ RT_b12 @ RT_e1b1 @ RT_x1
                dpc_drot = np.column_stack([R_prefix @ dRa_c1b @ q0, R_prefix @ dRb_c1b @ q0, R_prefix @ dRg_c1b @ q0])
                dpc_dTc1b = np.hstack([dpc_drot, R_prefix])
                J[row:row+2, slice(layout['c1b'].start + 6*i, layout['c1b'].start + 6*(i+1))] = J_cam @ dpc_dTc1b
            if estimate_x1ec and (not is_scara_x1):
                R_prefix = RT_e2c2 @ R_e2b2 @ RT_b12 @ RT_e1b1
                dpc_drot = np.column_stack([R_prefix @ (dRa_x1.T @ (q1 - t_x1)),
                                            R_prefix @ (dRb_x1.T @ (q1 - t_x1)),
                                            R_prefix @ (dRg_x1.T @ (q1 - t_x1))])
                dpc_dtrans = R_prefix @ (-RT_x1)
                dpc_dTx1 = np.hstack([dpc_drot, dpc_dtrans])
                J[row:row+2, layout['x1ec']] = J_cam @ dpc_dTx1
            if estimate_e1b1:
                R_prefix = RT_e2c2 @ R_e2b2 @ RT_b12
                dpc_drot = np.column_stack([R_prefix @ (dRa_e1b1.T @ (q2 - t_e1b1)),
                                            R_prefix @ (dRb_e1b1.T @ (q2 - t_e1b1)),
                                            R_prefix @ (dRg_e1b1.T @ (q2 - t_e1b1))])
                dpc_dtrans = R_prefix @ (-RT_e1b1)
                dpc_dTe1b1 = np.hstack([dpc_drot, dpc_dtrans])
                J[row:row+2, slice(layout['e1b1'].start + 6*i, layout['e1b1'].start + 6*(i+1))] = J_cam @ dpc_dTe1b1
            if estimate_b1b2:
                R_prefix = RT_e2c2 @ R_e2b2
                dpc_drot = np.column_stack([R_prefix @ (dRa_b12.T @ (q3 - t_b12)),
                                            R_prefix @ (dRb_b12.T @ (q3 - t_b12)),
                                            R_prefix @ (dRg_b12.T @ (q3 - t_b12))])
                dpc_dtrans = R_prefix @ (-RT_b12)
                dpc_dTb12 = np.hstack([dpc_drot, dpc_dtrans])
                J[row:row+2, layout['b1b2']] = J_cam @ dpc_dTb12
            if estimate_b2e2:
                R_prefix = RT_e2c2
                dpc_drot = np.column_stack([R_prefix @ dRa_e2b2 @ q4, R_prefix @ dRb_e2b2 @ q4, R_prefix @ dRg_e2b2 @ q4])
                dpc_dTe2b2 = np.hstack([dpc_drot, R_prefix])
                J[row:row+2, slice(layout['b2e2'].start + 6*i, layout['b2e2'].start + 6*(i+1))] = J_cam @ dpc_dTe2b2
            if estimate_e2c2:
                R_prefix = np.eye(3)
                dpc_drot = np.column_stack([R_prefix @ (dRa_e2c2.T @ (q5 - t_e2c2)),
                                            R_prefix @ (dRb_e2c2.T @ (q5 - t_e2c2)),
                                            R_prefix @ (dRg_e2c2.T @ (q5 - t_e2c2))])
                dpc_dtrans = R_prefix @ (-RT_e2c2)
                dpc_dTe2c2 = np.hstack([dpc_drot, dpc_dtrans])
                J[row:row+2, layout['e2c2']] = J_cam @ dpc_dTe2c2
            if estimate_intr2:
                J[row:row+2, layout['intr2']] = np.hstack([J_c, J_k, J_map_intr[:,[0,1,2,3] if include_sy2 else [0,2,3]]])

            row += 2
            
    return J, layout


def calculate_analytical_jacobian_shared_target_v2(
    # --- 함수 인자는 이전과 동일 ---
    T_B1_Board: np.ndarray, T_C1E1: np.ndarray, T_B2B1: np.ndarray, T_C2E2: np.ndarray,
    T_E1B1_list: list, T_B2E2_list: list, obj_pts_list: list,
    c1: float, kappa1: float, sx1: float, sy1: float, cx1: float, cy1: float,
    c2: float, kappa2: float, sx2: float, sy2: float, cx2: float, cy2: float,
    estimate_b1board: bool, estimate_c1e1: bool, estimate_b2b1: bool, estimate_c2e2: bool,
    estimate_e1b1: bool, estimate_b2e2: bool, estimate_intr1: bool, estimate_intr2: bool,
    include_sy1: bool, include_sy2: bool, is_scara_c1e1: bool,
    mat_to_vec6d=None # 호환성을 위한 인자
):
    """
    [최종 수정 3.0] '공유 T' 결합형 V2 모델을 위한 통합 자코비안 함수.
    'division_model' 함수의 구조를 충실히 따라서 확장한 최종 버전.
    """
    num_poses = len(T_E1B1_list)
    num_points_cam1 = sum(len(pts) for pts in obj_pts_list) * 2
    num_rows = num_points_cam1 * 2

    # --- 레이아웃 계산 (기존과 동일) ---
    layout = {}; col = 0
    dim_c1e1 = 5 if is_scara_c1e1 else 6
    if estimate_b1board: layout['b1board'] = slice(col, col+6); col += 6
    if estimate_c1e1:  layout['c1e1']  = slice(col, col+dim_c1e1); col += dim_c1e1
    if estimate_b2b1:  layout['b2b1']  = slice(col, col+6); col += 6
    if estimate_c2e2:  layout['c2e2']  = slice(col, col+6); col += 6
    if estimate_e1b1:  layout['e1b1']  = slice(col, col+6*num_poses); col += 6*num_poses
    if estimate_b2e2:  layout['b2e2']  = slice(col, col+6*num_poses); col += 6*num_poses
    if estimate_intr1: layout['intr1'] = slice(col, col+(6 if include_sy1 else 5)); col += (6 if include_sy1 else 5)
    if estimate_intr2: layout['intr2'] = slice(col, col+(6 if include_sy2 else 5)); col += (6 if include_sy2 else 5)
    num_cols = col
    
    A_img = np.zeros((num_rows, num_cols), dtype=float)

    # --- 전역 변환 분해 및 미분 미리 계산 ---
    R_B1_Board, t_B1_Board = T_B1_Board[:3,:3], T_B1_Board[:3,3]
    dR_da_B1B, dR_db_B1B, dR_dg_B1B = get_rotation_derivatives(*mat_to_vec6d(T_B1_Board)[:3])

    R_C1E1, t_C1E1 = T_C1E1[:3,:3], T_C1E1[:3,3]
    dR_da_C1E1, dR_db_C1E1, dR_dg_C1E1 = get_rotation_derivatives(*mat_to_vec6d(T_C1E1)[:3])
    
    R_B2B1, t_B2B1 = T_B2B1[:3,:3], T_B2B1[:3,3]
    dR_da_B2B1, dR_db_B2B1, dR_dg_B2B1 = get_rotation_derivatives(*mat_to_vec6d(T_B2B1)[:3])
    
    R_C2E2, t_C2E2 = T_C2E2[:3,:3], T_C2E2[:3,3]
    dR_da_C2E2, dR_db_C2E2, dR_dg_C2E2 = get_rotation_derivatives(*mat_to_vec6d(T_C2E2)[:3])

    row_ptr1, row_ptr2 = 0, num_points_cam1
    for i in range(num_poses):
        R_E1B1, t_E1B1 = T_E1B1_list[i][:3,:3], T_E1B1_list[i][:3,3]
        dR_da_E1B1, dR_db_E1B1, dR_dg_E1B1 = get_rotation_derivatives(*mat_to_vec6d(T_E1B1_list[i])[:3])
        
        R_B2E2, t_B2E2 = T_B2E2_list[i][:3,:3], T_B2E2_list[i][:3,3]
        dR_da_B2E2, dR_db_B2E2, dR_dg_B2E2 = get_rotation_derivatives(*mat_to_vec6d(T_B2E2_list[i])[:3])
        
        for pw in obj_pts_list[i]:
            # --- 중간 포인트 계산 ---
            p_b1 = R_B1_Board @ pw + t_B1_Board
            p_e1 = R_E1B1 @ p_b1 + t_E1B1
            pc1  = R_C1E1 @ p_e1 + t_C1E1
            
            p_b2 = R_B2B1 @ p_b1 + t_B2B1
            p_e2 = R_B2E2 @ p_b2 + t_B2E2
            pc2  = R_C2E2 @ p_e2 + t_C2E2
            
            # --- 1. 카메라 1 자코비안 ---
            row_slice1 = slice(row_ptr1, row_ptr1 + 2)
            J_cam1 = calculate_camera_model_jacobian(pc1, c1, kappa1, sx1, sy1, cx1, cy1, include_sy1)
            if J_cam1 is None:
                A_img[row_slice1, :] = np.nan; row_ptr1 += 2; continue
            
            if estimate_b1board:
                dpc1_drot = R_C1E1 @ R_E1B1 @ np.stack([dR_da_B1B@pw, dR_db_B1B@pw, dR_dg_B1B@pw], axis=1)
                dpc1_dt   = R_C1E1 @ R_E1B1
                A_img[row_slice1, layout['b1board']] = J_cam1['pix_pc'] @ np.hstack((dpc1_drot, dpc1_dt))
            if estimate_c1e1:
                dpc1_drot = np.stack([dR_da_C1E1@p_e1, dR_db_C1E1@p_e1, dR_dg_C1E1@p_e1], axis=1)
                dpc1_dt   = np.eye(3)
                A_img[row_slice1, layout['c1e1']] = J_cam1['pix_pc'] @ np.hstack((dpc1_drot, dpc1_dt))[:,:dim_c1e1]
            if estimate_e1b1:
                col = slice(layout['e1b1'].start + 6*i, layout['e1b1'].start + 6*(i+1))
                dpc1_drot = R_C1E1 @ np.stack([dR_da_E1B1@p_b1, dR_db_E1B1@p_b1, dR_dg_E1B1@p_b1], axis=1)
                dpc1_dt   = R_C1E1
                A_img[row_slice1, col] = J_cam1['pix_pc'] @ np.hstack((dpc1_drot, dpc1_dt))
            if estimate_intr1: A_img[row_slice1, layout['intr1']] = J_cam1['pix_intr']
            
            # --- 2. 카메라 2 자코비안 ---
            row_slice2 = slice(row_ptr2, row_ptr2 + 2)
            J_cam2 = calculate_camera_model_jacobian(pc2, c2, kappa2, sx2, sy2, cx2, cy2, include_sy2)
            if J_cam2 is None:
                A_img[row_slice2, :] = np.nan; row_ptr2 += 2; continue
            
            if estimate_b1board:
                dpc2_drot = R_C2E2 @ R_B2E2 @ R_B2B1 @ np.stack([dR_da_B1B@pw, dR_db_B1B@pw, dR_dg_B1B@pw], axis=1)
                dpc2_dt   = R_C2E2 @ R_B2E2 @ R_B2B1
                A_img[row_slice2, layout['b1board']] = J_cam2['pix_pc'] @ np.hstack((dpc2_drot, dpc2_dt))
            if estimate_b2b1:
                dpc2_drot = R_C2E2 @ R_B2E2 @ np.stack([dR_da_B2B1@p_b1, dR_db_B2B1@p_b1, dR_dg_B2B1@p_b1], axis=1)
                dpc2_dt   = R_C2E2 @ R_B2E2
                A_img[row_slice2, layout['b2b1']] = J_cam2['pix_pc'] @ np.hstack((dpc2_drot, dpc2_dt))
            if estimate_c2e2:
                dpc2_drot = np.stack([dR_da_C2E2@p_e2, dR_db_C2E2@p_e2, dR_dg_C2E2@p_e2], axis=1)
                dpc2_dt   = np.eye(3)
                A_img[row_slice2, layout['c2e2']] = J_cam2['pix_pc'] @ np.hstack((dpc2_drot, dpc2_dt))
            if estimate_b2e2:
                col = slice(layout['b2e2'].start + 6*i, layout['b2e2'].start + 6*(i+1))
                dpc2_drot = R_C2E2 @ np.stack([dR_da_B2E2@p_b2, dR_db_B2E2@p_b2, dR_dg_B2E2@p_b2], axis=1)
                dpc2_dt   = R_C2E2
                A_img[row_slice2, col] = J_cam2['pix_pc'] @ np.hstack((dpc2_drot, dpc2_dt))
            if estimate_intr2: A_img[row_slice2, layout['intr2']] = J_cam2['pix_intr']

            row_ptr1 += 2
            row_ptr2 += 2
            
    return A_img, layout

def calculate_camera_model_jacobian(pc, c, kappa, sx, sy, cx, cy, include_sy):
    """
    주어진 3D 포인트 pc에 대해 카메라 모델(투영, 왜곡, 내재 파라미터 매핑)의
    자코비안을 계산하는 헬퍼 함수.
    """
    if pc[2] <= 1e-9: return None # 투영 불가

    # 1. Projection (pc -> pu): 3D -> 정규화 이미지 평면
    inv_Zc, inv_Zc2 = 1.0 / pc[2], 1.0 / (pc[2]**2)
    pu = c * pc[:2] * inv_Zc
    J_proj_pc = c * np.array([[inv_Zc, 0, -pc[0] * inv_Zc2],
                              [0, inv_Zc, -pc[1] * inv_Zc2]])
    J_proj_c = pc[:2] * inv_Zc

    # 2. Distortion (pu -> pd): Division Model
    J_dist_u, J_dist_k = get_distortion_jacobian_division(pu, kappa)
    
    ru2 = pu[0]**2 + pu[1]**2
    sqrt_arg = 1.0 - 4.0 * kappa * ru2
    if sqrt_arg < 0: return None # 왜곡 불가
    
    ru = np.sqrt(ru2)
    if ru < 1e-9:
        xd, yd = pu[0], pu[1]
    else:
        rd = (1.0 - np.sqrt(sqrt_arg)) / (2.0 * kappa * ru)
        xd, yd = pu * (rd / ru)

    # 3. Intrinsic Mapping (pd -> pix)
    J_map_full = get_intrinsic_jacobian(sx, sy, xd, yd)
    J_map_d = J_map_full[:, 4:] # d_pix / d_pd
    
    # 4. Chain Rule 결합
    J_pix_pc = J_map_d @ J_dist_u @ J_proj_pc
    
    J_c_chain = J_map_d @ J_dist_u @ J_proj_c
    J_kappa_chain = J_map_d @ J_dist_k
    J_map_intr = J_map_full[:, :4]
    
    if include_sy:
        # 상태 벡터 순서: c, κ, sx, sy, cx, cy
        J_pix_intr = np.hstack([J_c_chain[:,None], J_kappa_chain, J_map_intr[:,[0,1,2,3]]])
    else:
        # 상태 벡터 순서: c, κ, sx, cx, cy
        J_pix_intr = np.hstack([J_c_chain[:,None], J_kappa_chain, J_map_intr[:,[0,2,3]]])

    return {'pix_pc': J_pix_pc, 'pix_intr': J_pix_intr}


def calculate_analytical_jacobian_division_model_axbycz(
    # Global (estimate flags로 on/off)
    X1_EC: np.ndarray,        # ^C1 T_E1 (Hand-Eye)
    T_B1B2: np.ndarray,       # ^B1 T_B2 (Robot-Robot)
    T_E2B: np.ndarray,        # ^E2 T_B  (Tool) <-- (수정) 새로운 글로벌 파라미터
    # Per-pose
    T_E1B1_list: list,        # [ ^E1 T_B1 ]_i (Robot1 FK)
    T_B2E2_list: list,        # [ ^B2 T_E2 ]_i (Robot2 FK)
    obj_pts_list: list,       # [ (Ni,3) ]_i  (board 좌표계)
    # Division intrinsics
    c: float, kappa: float, sx: float, sy: float, cx: float, cy: float,
    # Flags
    estimate_x1ec: bool = True,
    estimate_b1b2: bool = True,
    estimate_e2b: bool = True, # <-- (수정)
    estimate_e1b1: bool = True,
    estimate_b2e2: bool = True,
    estimate_intrinsics: bool = True,
    include_sy: bool = False,
    is_scara_x1: bool = False,
    # 유틸리티 함수 포인터 (반드시 제공)
    mat_to_vec6d=None,
) -> np.ndarray:
    """
    수정된 체인: Board -> E2 -> B2 -> B1 -> E1 -> C1
    p_cam = (X1_EC) * (E1B1_i) * (B1B2) * (B2E2_i) * (E2B) * p_board
    자코비안 열 순서: [글로벌(X1_EC, B1B2, E2B), per-pose(E1B1_i, B2E2_i), intrinsics]
    """

    if not include_sy:
        sy = sx

    num_poses = len(obj_pts_list)
    assert len(T_E1B1_list) == len(T_B2E2_list) == num_poses

    # ---- (수정) 열 레이아웃 구성 ----
    layout = {}
    col = 0
    dim_x1 = 5 if is_scara_x1 else 6
    if estimate_x1ec:
        layout['x1ec'] = slice(col, col + dim_x1); col += dim_x1
    if estimate_b1b2:
        layout['b1b2'] = slice(col, col + 6); col += 6
    if estimate_e2b:
        layout['e2b'] = slice(col, col + 6); col += 6
    if estimate_e1b1:
        layout['e1b1'] = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_b2e2:
        layout['b2e2'] = slice(col, col + 6 * num_poses); col += 6 * num_poses
    if estimate_intrinsics:
        n_intr = 6 if include_sy else 5
        layout['intr'] = slice(col, col + n_intr); col += n_intr

    num_cols = col
    num_rows = sum(len(pts) for pts in obj_pts_list) * 2
    J = np.zeros((num_rows, num_cols), dtype=float)

    # ---- (수정) 글로벌 T에서 회전 자코비안 미리 준비 ----
    def rot_derivs_from_T(T):
        a, b, g, *_ = mat_to_vec6d(T)
        return get_rotation_derivatives(a, b, g)

    dR_da_x1, dR_db_x1, dR_dg_x1   = rot_derivs_from_T(X1_EC)
    dR_da_b12, dR_db_b12, dR_dg_b12 = rot_derivs_from_T(T_B1B2)
    dR_da_e2b, dR_db_e2b, dR_dg_e2b   = rot_derivs_from_T(T_E2B)

    R_x1,  t_x1  = X1_EC[:3,:3],  X1_EC[:3,3]
    R_b12, t_b12 = T_B1B2[:3,:3], T_B1B2[:3,3]
    R_e2b, t_e2b = T_E2B[:3,:3], T_E2B[:3,3]

    row_offset = 0
    for i in range(num_poses):
        T_e1b1 = T_E1B1_list[i];  R_e1b1, t_e1b1 = T_e1b1[:3,:3], T_e1b1[:3,3]
        T_b2e2 = T_B2E2_list[i];  R_b2e2, t_b2e2 = T_b2e2[:3,:3], T_b2e2[:3,3]

        dR_da_e1b1, dR_db_e1b1, dR_dg_e1b1 = rot_derivs_from_T(T_e1b1)
        dR_da_b2e2, dR_db_b2e2, dR_dg_b2e2 = rot_derivs_from_T(T_b2e2)

        # ---- (수정) 체인 순서 ----
        # k=0: E2B,  k=1: B2E2_i,  k=2: B1B2,  k=3: E1B1_i,  k=4: X1EC
        R_list = [R_e2b, R_b2e2, R_b12, R_e1b1, R_x1]
        t_list = [t_e2b, t_b2e2, t_b12, t_e1b1, t_x1]

        # R_post[k] = R_{k+1} ... R_4
        R_post = [np.eye(3) for _ in range(5)]
        for k in range(3, -1, -1):
            R_post[k] = R_list[k+1] @ R_post[k+1]

        for pt_idx, pw in enumerate(obj_pts_list[i]):
            row = row_offset + pt_idx * 2
            
            # (수정) prefix points: s0=pw, s_{k+1}=R_k s_k + t_k
            s = [None] * 6
            s[0] = pw.astype(float)
            for k in range(5):
                s[k+1] = R_list[k] @ s[k] + t_list[k]
            pc = s[5] # 최종 카메라 좌표계 포인트

            if pc[2] <= 1e-9:
                continue

            # ---- 투영 및 왜곡 자코비안 (이하 로직은 대부분 동일) ----
            invZ  = 1.0 / pc[2]; invZ2 = invZ*invZ
            pu = c * pc[:2] * invZ
            J_proj_pc = c * np.array([[invZ, 0.0, -pc[0]*invZ2],
                                      [0.0, invZ, -pc[1]*invZ2]], dtype=float)
            J_proj_c  = (pc[:2] * invZ).reshape(2,1)
            J_dist_u, J_dist_k = get_distortion_jacobian_division(pu, kappa)
            ux, uy = float(pu[0]), float(pu[1])
            ru2 = ux*ux + uy*uy
            if abs(kappa) < 1e-15 or ru2 < 1e-24:
                xd, yd = ux, uy
            else:
                ru = np.sqrt(ru2); g = 1.0 - 4.0*kappa*ru2
                if g < 0.0: continue
                Delta = np.sqrt(g); rd = (1.0 - Delta) / (2.0*kappa*ru)
                s_scale = rd / (ru + 1e-12); xd, yd = s_scale * ux, s_scale * uy
            J_map_full = get_intrinsic_jacobian(sx, sy, xd, yd)
            J_map_intr = J_map_full[:, :4]; J_map_d = J_map_full[:, 4:]
            J_cam = J_map_d @ J_dist_u @ J_proj_pc

            # ---- (수정) 변환 블록별 열 채우기 (체인 룰) ----
            def fill_jacobian_block(dR_a, dR_b, dR_g, s_k, R_post_k, col_slice):
                dpc_drot = R_post_k @ np.column_stack((dR_a @ s_k, dR_b @ s_k, dR_g @ s_k))
                dpc_dt = R_post_k
                J[row:row+2, col_slice] = J_cam @ np.hstack((dpc_drot, dpc_dt))

            # X1_EC (k=4)
            if estimate_x1ec:
                if is_scara_x1:
                    dpc_drot = np.column_stack((dR_da_x1 @ s[4], dR_db_x1 @ s[4], dR_dg_x1 @ s[4]))
                    dpc_de_full = np.hstack((dpc_drot, np.eye(3)))
                    J[row:row+2, layout['x1ec']] = J_cam @ dpc_de_full[:, :5]
                else:
                    fill_jacobian_block(dR_da_x1, dR_db_x1, dR_dg_x1, s[4], np.eye(3), layout['x1ec'])

            # B1B2 (k=2)
            if estimate_b1b2:
                fill_jacobian_block(dR_da_b12, dR_db_b12, dR_dg_b12, s[2], R_post[2], layout['b1b2'])
            
            # E2B (k=0)
            if estimate_e2b:
                fill_jacobian_block(dR_da_e2b, dR_db_e2b, dR_dg_e2b, s[0], R_post[0], layout['e2b'])

            # E1B1_i (k=3)
            if estimate_e1b1:
                cs = slice(layout['e1b1'].start + 6*i, layout['e1b1'].start + 6*(i+1))
                fill_jacobian_block(dR_da_e1b1, dR_db_e1b1, dR_dg_e1b1, s[3], R_post[3], cs)

            # B2E2_i (k=1)
            if estimate_b2e2:
                cs = slice(layout['b2e2'].start + 6*i, layout['b2e2'].start + 6*(i+1))
                fill_jacobian_block(dR_da_b2e2, dR_db_b2e2, dR_dg_b2e2, s[1], R_post[1], cs)

            # Intrinsics
            if estimate_intrinsics:
                intr_slice = layout['intr']
                J_c_chain = J_map_d @ J_dist_u @ J_proj_c
                J_kappa_chain = J_map_d @ J_dist_k
                if include_sy:
                    J_intr_block = np.hstack([J_c_chain, J_kappa_chain, J_map_intr[:,0:1], J_map_intr[:,1:2], J_map_intr[:,2:3], J_map_intr[:,3:4]])
                    J[row:row+2, intr_slice] = J_intr_block
                else:
                    J_sx_combined = (J_map_intr[:, 0] + J_map_intr[:, 1]).reshape(2,1)
                    J_intr_block = np.hstack([J_c_chain, J_kappa_chain, J_sx_combined, J_map_intr[:,2:3], J_map_intr[:,3:4]])
                    J[row:row+2, intr_slice] = J_intr_block
        
        row_offset += len(obj_pts_list[i]) * 2

    return J