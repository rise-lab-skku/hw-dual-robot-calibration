# metric.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

# 핵심: projection.py의 공용 인터페이스(Protocol)만 의존
from utils.projection import BaseProjector  # DivisionProjector / PolynomialProjector 둘 다 이 인터페이스를 만족

# -------- se(3) 유틸 --------
def _inv4(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def log_se3_from_matrix(T: np.ndarray) -> np.ndarray:
    """
    SE(3) -> se(3) (rho, w). rho[m], w[rad].
    """
    R = T[:3, :3]; t = T[:3, 3]
    tr = np.trace(R)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)

    if abs(theta) < 1e-9:
        w = np.array([R[2,1] - R[1,2],
                      R[0,2] - R[2,0],
                      R[1,0] - R[0,1]]) / 2.0
        w_hat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
        V_inv = np.eye(3) - 0.5 * w_hat
        rho = V_inv @ t
    else:
        w_hat = (theta / (2.0 * np.sin(theta))) * (R - R.T)
        w = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])
        A = np.sin(theta); B = 1.0 - np.cos(theta)
        V_inv = (np.eye(3)
                 - 0.5 * w_hat
                 + (1.0 / (theta**2)) * (1 - (A / (2*B/theta))) * (w_hat @ w_hat))
        rho = V_inv @ t
    return np.hstack([rho, w])

@dataclass
class PoseError:
    trans_mm: float
    rot_deg: float

@dataclass
class AXZBReport:
    per_pose: List[PoseError]
    mean_trans_mm: float
    mean_rot_deg: float
    max_trans_mm: float
    max_rot_deg: float

class Metrics:
    """
    모델-불문(division / polynomial / …) 공용 메트릭 계산기.
    - projector: projection.py의 Projector (DivisionProjector / PolynomialProjector 등)
    """
    def __init__(self, nan_policy: str = "omit"):
        assert nan_policy in ("omit", "raise")
        self.nan_policy = nan_policy

    # -----------------------
    # Reprojection Residuals
    # -----------------------
    def reproj_residuals(
        self,
        projector: BaseProjector,
        X_EC: np.ndarray,
        X_WB: np.ndarray,
        T_BE_list: List[np.ndarray],
        obj_pts_list: List[np.ndarray],
        img_pts_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        전체 데이터셋에 대한 (u_obs - u_pred, v_obs - v_pred) 잔차 벡터와
        pose별 잔차 배열 리스트를 반환.
        """
        assert len(T_BE_list) == len(obj_pts_list) == len(img_pts_list)

        residuals_all = []
        residuals_per_pose = []

        for T_BE, Pw, uv_obs in zip(T_BE_list, obj_pts_list, img_pts_list):
            uv_pred = self._project_pose(projector, X_EC, X_WB, T_BE, Pw)  # (N, 2) with NaN allowed
            obs = uv_obs.reshape(-1, 2).astype(np.float32)

            valid_mask = ~np.isnan(uv_pred).any(axis=1)
            if not np.any(valid_mask):
                if self.nan_policy == "raise":
                    raise ValueError("No valid projections for a pose.")
                # omit
                continue

            r = obs[valid_mask] - uv_pred[valid_mask].astype(np.float32)  # (M, 2)
            residuals_per_pose.append(r)
            residuals_all.append(r.reshape(-1))

        if len(residuals_all) == 0:
            return np.array([]), []

        return np.concatenate(residuals_all), residuals_per_pose

    def _project_pose(
        self,
        projector: BaseProjector,
        X_EC: np.ndarray,
        X_WB: np.ndarray,
        T_BE: np.ndarray,
        Pw: np.ndarray,
    ) -> np.ndarray:
        """
        단일 포즈용 2D 예측. projector.project_single을 벡터화해서 NaN-safe하게 반환.
        """
        # 구현체가 project_single을 제공한다고 가정(Projection의 Protocol)
        uv_list = []
        for p in Pw:
            uv = projector.project_single(X_EC, X_WB, T_BE, p)
            if uv is None or not np.all(np.isfinite(uv)):
                uv_list.append([np.nan, np.nan])
            else:
                uv_list.append(uv)
        return np.asarray(uv_list, dtype=float)

    # -----------------------
    # RMSE (px)
    # -----------------------
    def reproj_rmse(
        self,
        projector: BaseProjector,
        X_EC: np.ndarray,
        X_WB: np.ndarray,
        T_BE_list: List[np.ndarray],
        obj_pts_list: List[np.ndarray],
        img_pts_list: List[np.ndarray],
        return_per_pose: bool = False
    ) -> Tuple[float, Optional[List[float]]]:
        """
        Global RMSE(px) 및 (옵션) 포즈별 RMSE(px) 리스트 반환.
        """
        r_all, r_per_pose = self.reproj_residuals(projector, X_EC, X_WB, T_BE_list, obj_pts_list, img_pts_list)
        if r_all.size == 0:
            return float("inf"), None if not return_per_pose else []

        # global RMSE
        # r_all = [du1, dv1, du2, dv2, ...]
        rmse_global = float(np.sqrt(np.mean(r_all**2)))

        if not return_per_pose:
            return rmse_global, None

        rmse_each = [float(np.sqrt(np.mean(r**2))) for r in r_per_pose]
        return rmse_global, rmse_each

    # -----------------------
    # 비교 유틸 (모델 A vs B)
    # -----------------------
    def compare_models_rmse(
        self,
        proj_a: BaseProjector,
        proj_b: BaseProjector,
        X_EC: np.ndarray,
        X_WB: np.ndarray,
        T_BE_list: List[np.ndarray],
        obj_pts_list: List[np.ndarray],
        img_pts_list: List[np.ndarray],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        같은 데이터로 두 프로젝트의 RMSE를 비교해 준다.
        """
        rmse_a, _ = self.reproj_rmse(proj_a, X_EC, X_WB, T_BE_list, obj_pts_list, img_pts_list, return_per_pose=False)
        rmse_b, _ = self.reproj_rmse(proj_b, X_EC, X_WB, T_BE_list, obj_pts_list, img_pts_list, return_per_pose=False)
        if verbose:
            print("\n==================================================")
            print("Reprojection Error Comparison")
            print("==================================================")
            print(f"  Model A RMSE: {rmse_a:.6f} px")
            print(f"  Model B RMSE: {rmse_b:.6f} px")
            print("==================================================\n")
        return {"rmse_a_px": rmse_a, "rmse_b_px": rmse_b}

    # -----------------------
    # AX = ZB 검증 및 RAE
    # -----------------------
    def ax_zb_report(
        self,
        A_list: List[np.ndarray],  # 보드->캠 (또는 캠->보드) 일관된 정의로
        B_list: List[np.ndarray],  # 베이스->EE (또는 EE->베이스)
        X: np.ndarray,             # 미지 해
        Z: np.ndarray              # 미지 해
    ) -> AXZBReport:
        """
        AX=ZB 검증: 각 포즈에 대해 T_err = (A_i X)^-1 (Z B_i), log(SE3)로
        이동/회전 오차를 집계.
        """
        per_pose = []
        max_t, max_r = -1.0, -1.0
        sum_t, sum_r = 0.0, 0.0

        for A, B in zip(A_list, B_list):
            LHS = A @ X
            RHS = Z @ B
            T_err = _inv4(LHS) @ RHS
            xi = log_se3_from_matrix(T_err)  # [rho(m), w(rad)]
            trans_mm = float(np.linalg.norm(xi[:3]) * 1000.0)
            rot_deg  = float(np.linalg.norm(xi[3:]) * 180.0/np.pi)

            per_pose.append(PoseError(trans_mm, rot_deg))
            sum_t += trans_mm; sum_r += rot_deg
            max_t = max(max_t, trans_mm); max_r = max(max_r, rot_deg)

        n = max(1, len(per_pose))
        return AXZBReport(
            per_pose=per_pose,
            mean_trans_mm=sum_t / n,
            mean_rot_deg=sum_r / n,
            max_trans_mm=max_t,
            max_rot_deg=max_r
        )

    def rae_m2_from_axzb(
        self,
        A_list: List[np.ndarray],
        B_list: List[np.ndarray],
        X: np.ndarray,
        Z: np.ndarray
    ) -> float:
        """
        RAE(m^2): AX=ZB의 se(3) 로그에서 translation(rho)의 제곱 평균.
        (데이터셋/정의에 따라 다른 RAE가 필요하면 이 함수를 교체하세요.)
        """
        if len(A_list) == 0:
            return float("inf")
        sq = []
        for A, B in zip(A_list, B_list):
            T_err = _inv4(A @ X) @ (Z @ B)
            rho = log_se3_from_matrix(T_err)[:3]  # [m]
            sq.append(float(rho @ rho))
        return float(np.mean(sq))
