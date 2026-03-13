# projection.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Protocol, runtime_checkable
import numpy as np

_EPS = 1e-9

# ---------------------------
# Intrinsics dataclasses
# ---------------------------

@dataclass
class DivisionIntrinsics:
    c: float
    kappa: float
    sx: float
    sy: float
    cx: float
    cy: float
    include_sy: bool = False  # False면 sy는 무시하고 sx를 사용

@dataclass
class PolyIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # (5,) or (8,)

# ---------------------------
# 공통 체인/유틸
# ---------------------------

def _to_array3xN(Pw_list: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Pw_list: (N,3) 또는 list-like(N,3)
    return: Pw(3,N), N
    """
    if isinstance(Pw_list, np.ndarray):
        assert Pw_list.ndim == 2 and Pw_list.shape[1] == 3, "Pw_list must be (N,3)"
        N = Pw_list.shape[0]
        return Pw_list.T.copy(), N
    arr = np.asarray(Pw_list, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 3, "Pw_list must be (N,3)"
    return arr.T.copy(), arr.shape[0]

def _T_cam_from_board(X_EC: np.ndarray, X_WB: np.ndarray, T_BE: np.ndarray) -> np.ndarray:
    """
    체인 계약:  T_CB = X_EC @ T_BE @ X_WB
    ( ^C T_B = ^C T_E * ^E T_B * ^B T_W ), 여기서 W는 보드 좌표계.
    """
    return X_EC @ T_BE @ X_WB

def _apply_T(T: np.ndarray, Pw_3xN: np.ndarray) -> np.ndarray:
    """Pw(3,N) → Pc(3,N) with 4x4 T."""
    R, t = T[:3, :3], T[:3, 3:4]
    return R @ Pw_3xN + t

def _uv_flatten(uv_Nx2: np.ndarray) -> np.ndarray:
    """(N,2) → (2N,) [u0,v0,u1,v1,...]"""
    N = uv_Nx2.shape[0]
    out = np.empty(2 * N, dtype=float)
    out[0::2] = uv_Nx2[:, 0]
    out[1::2] = uv_Nx2[:, 1]
    return out

# ---------------------------
# Projector 인터페이스
# ---------------------------

@runtime_checkable
class BaseProjector(Protocol):
    """
    Metrics가 기대하는 최소 인터페이스.
    """
    def project_dataset_flat(self,
                             X_EC: np.ndarray,
                             X_WB: np.ndarray,
                             T_BE_list: Sequence[np.ndarray],
                             obj_pts_list: Sequence[np.ndarray]) -> np.ndarray: ...
    def project_single(self,
                       X_EC: np.ndarray,
                       X_WB: np.ndarray,
                       T_BE: np.ndarray,
                       Pw: np.ndarray) -> np.ndarray: ...

# ---------------------------
# Division Projector
# ---------------------------

class DivisionProjector:
    """
    Scaramuzza 스타일(루트식) division 모델.
    픽셀 매핑: u = xd/sx + cx, v = yd/sy + cy  (include_sy=False면 sy=sx로 동작)
    """

    def __init__(self, intr: DivisionIntrinsics):
        self.intr = intr

    def _project_pose_uv(self, X_EC: np.ndarray, X_WB: np.ndarray, T_BE: np.ndarray,
                         Pw_list: np.ndarray) -> np.ndarray:
        T_CB = _T_cam_from_board(X_EC, X_WB, T_BE)
        Pw, N = _to_array3xN(Pw_list)
        Pc = _apply_T(T_CB, Pw)

        uv = np.full((N, 2), np.nan, dtype=float)
        mask = Pc[2, :] > _EPS
        if not np.any(mask):
            return uv

        X = Pc[0, mask]; Y = Pc[1, mask]; Z = Pc[2, mask]

        # undistorted plane using c
        xu = self.intr.c * X / Z
        yu = self.intr.c * Y / Z

        if abs(self.intr.kappa) < 1e-16:
            xd, yd = xu, yu
        else:
            ru2 = xu * xu + yu * yu
            sqrt_arg = 1.0 - 4.0 * self.intr.kappa * ru2

            xd = np.full_like(xu, np.nan)
            yd = np.full_like(yu, np.nan)
            valid = sqrt_arg >= 0.0
            if np.any(valid):
                ru = np.sqrt(ru2[valid])
                scale = np.ones_like(ru)
                nz = ru > _EPS
                if np.any(nz):
                    delta = np.sqrt(sqrt_arg[valid])
                    scale = 2.0 / (1.0 + delta)
                    # rd = (1.0 - np.sqrt(sqrt_arg[valid][nz])) / (2.0 * self.intr.kappa * ru[nz])
                    # scale[nz] = rd / ru[nz]
                xd[valid] = xu[valid] * scale
                yd[valid] = yu[valid] * scale

        u = xd / self.intr.sx + self.intr.cx
        v = yd / self.intr.sy + self.intr.cy

        uv[mask, 0] = u
        uv[mask, 1] = v
        return uv

    # ---- 인터페이스 구현 ----
    def project_single(self, X_EC, X_WB, T_BE, Pw) -> np.ndarray:
        """단일 3D점 Pw(3,) → (u,v)"""
        uv = self._project_pose_uv(X_EC, X_WB, T_BE, Pw.reshape(1, 3))
        return uv[0]

    def project_dataset_flat(self, X_EC, X_WB, T_BE_list: Sequence[np.ndarray],
                             obj_pts_list: Sequence[np.ndarray]) -> np.ndarray:
        cols = []
        for T_BE, Pw in zip(T_BE_list, obj_pts_list):
            cols.append(_uv_flatten(self._project_pose_uv(X_EC, X_WB, T_BE, Pw)))
        return np.concatenate(cols, axis=0)

    # 편의: UV 형태로 받고 싶을 때
    def project_dataset_uv(self, X_EC, X_WB, T_BE_list, obj_pts_list) -> List[np.ndarray]:
        return [self._project_pose_uv(X_EC, X_WB, T_BE, Pw) for T_BE, Pw in zip(T_BE_list, obj_pts_list)]

# ---------------------------
# Polynomial / Rational Projector
# ---------------------------

class PolynomialProjector:
    """
    OpenCV Brown-Conrady(5계수) / Azure Kinect Rational(8계수)을 자동 지원.
    dist_coeffs 길이로 분기: len>5 → Rational(8), else → 5계수(OpenCV)
    """

    def __init__(self, intr: PolyIntrinsics):
        self.intr = intr
        dc = np.asarray(intr.dist_coeffs, dtype=float).reshape(-1)
        self._is_rational = (dc.size > 5)
        if self._is_rational:
            pad = np.zeros(8); pad[:dc.size] = dc
            self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6 = pad
        else:
            pad = np.zeros(5); pad[:dc.size] = dc
            self.k1, self.k2, self.p1, self.p2, self.k3 = pad

    def _project_pose_uv(self, X_EC: np.ndarray, X_WB: np.ndarray, T_BE: np.ndarray,
                         Pw_list: np.ndarray) -> np.ndarray:
        T_CB = _T_cam_from_board(X_EC, X_WB, T_BE)
        Pw, N = _to_array3xN(Pw_list)
        Pc = _apply_T(T_CB, Pw)

        uv = np.full((N, 2), np.nan, dtype=float)
        mask = Pc[2, :] > _EPS
        if not np.any(mask):
            return uv

        X = Pc[0, mask]; Y = Pc[1, mask]; Z = Pc[2, mask]
        xu = X / Z; yu = Y / Z
        r2 = xu * xu + yu * yu

        if self._is_rational:
            r4 = r2 * r2
            r6 = r4 * r2
            a = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
            b = 1.0 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6
            d = a / np.where(np.abs(b) > _EPS, b, 1.0)
        else:
            r4 = r2 * r2
            r6 = r2 * r4
            d = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

        xd = xu * d + (2 * self.p1 * xu * yu + self.p2 * (r2 + 2 * xu * xu))
        yd = yu * d + (self.p1 * (r2 + 2 * yu * yu) + 2 * self.p2 * xu * yu)

        u = self.intr.fx * xd + self.intr.cx
        v = self.intr.fy * yd + self.intr.cy

        uv[mask, 0] = u
        uv[mask, 1] = v
        return uv

    # ---- 인터페이스 구현 ----
    def project_single(self, X_EC, X_WB, T_BE, Pw) -> np.ndarray:
        uv = self._project_pose_uv(X_EC, X_WB, T_BE, Pw.reshape(1, 3))
        return uv[0]

    def project_dataset_flat(self, X_EC, X_WB, T_BE_list: Sequence[np.ndarray],
                             obj_pts_list: Sequence[np.ndarray]) -> np.ndarray:
        cols = []
        for T_BE, Pw in zip(T_BE_list, obj_pts_list):
            cols.append(_uv_flatten(self._project_pose_uv(X_EC, X_WB, T_BE, Pw)))
        return np.concatenate(cols, axis=0)

    def project_dataset_uv(self, X_EC, X_WB, T_BE_list, obj_pts_list) -> List[np.ndarray]:
        return [self._project_pose_uv(X_EC, X_WB, T_BE, Pw) for T_BE, Pw in zip(T_BE_list, obj_pts_list)]

# ---------------------------
# 팩토리 (선택)
# ---------------------------

def make_projector(model: str, intr) -> BaseProjector:
    """
    문자열로 간단히 생성:
      - model="division", intr: DivisionIntrinsics
      - model="polynomial"|"rational"|"opencv", intr: PolyIntrinsics
    """
    m = model.lower()
    if m == "division":
        if not isinstance(intr, DivisionIntrinsics):
            raise TypeError("intr must be DivisionIntrinsics for 'division'")
        return DivisionProjector(intr)
    elif m in ("polynomial", "rational", "opencv"):
        if not isinstance(intr, PolyIntrinsics):
            raise TypeError("intr must be PolyIntrinsics for 'polynomial'/'rational'/'opencv'")
        return PolynomialProjector(intr)
    else:
        raise ValueError(f"Unknown model: {model}")
