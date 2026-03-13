# dual_cam_datagen_realistic.py
from __future__ import annotations
import shutil
from dataclasses import dataclass, field, asdict
from typing import Tuple
import numpy as np
from pathlib import Path
import yaml
import cv2
import matplotlib.pyplot as plt

# =============================== SE(3) 유틸리티 ===============================
def euler_xyz_to_R(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rx = np.array([[1, 0, 0],
                   [0, ca, -sa],
                   [0, sa,  ca]])
    Ry = np.array([[ cb, 0, sb],
                   [  0, 1,  0],
                   [-sb, 0, cb]])
    Rz = np.array([[ cg, -sg, 0],
                   [ sg,  cg, 0],
                   [  0,   0, 1]])
    return Rx @ Ry @ Rz  # XYZ 순서

def R_to_euler_xyz(R):
    # R = Rx(α) Ry(β) Rz(γ) 가정하에서의 안정적 분해
    # 유도식: 
    #   beta = asin(R[0,2])
    #   alpha = atan2(-R[1,2], R[2,2])
    #   gamma = atan2(-R[0,1], R[0,0])
    beta = np.arcsin(np.clip(R[0,2], -1.0, 1.0))
    cb = np.cos(beta)
    # 기민락 방지: cos(beta) ~ 0이면 보정
    if abs(cb) < 1e-8:
        # 근사 처리: gamma=0으로 두고 alpha만 사용
        alpha = np.arctan2(R[2,1], R[1,1])
        gamma = 0.0
    else:
        alpha = np.arctan2(-R[1,2], R[2,2])
        gamma = np.arctan2(-R[0,1], R[0,0])
    return alpha, beta, gamma

def hat(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=float)

def so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    if th < 1e-12: return np.eye(3)
    a = w / th; K = hat(a)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = np.asarray(t).flatten(); return T

def inv4(T: np.ndarray) -> np.ndarray:
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

def log_so3(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-12: return np.zeros(3)
    return (th / (2.0 * np.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def make_checkerboard_corners(rows: int, cols: int, square_m: float) -> np.ndarray:
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    pts = np.stack([x.flatten()*square_m, y.flatten()*square_m, np.zeros(rows*cols)], axis=-1)
    pts[:,0] -= (cols-1)*square_m*0.5
    pts[:,1] -= (rows-1)*square_m*0.5
    return pts

# ========================== Plot 유틸리티 ===============================
def _triad(ax, T, length=0.15, lw=2.0, label=None, alpha=1.0):
    """좌표축 x(빨), y(초), z(파) 그리기. T: ^World T_Frame (여기선 B1이 World)."""
    o = T[:3, 3]
    x = o + T[:3, :3] @ (np.array([1,0,0]) * length)
    y = o + T[:3, :3] @ (np.array([0,1,0]) * length)
    z = o + T[:3, :3] @ (np.array([0,0,1]) * length)
    ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], '-', color='C3', lw=lw, alpha=alpha)
    ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], '-', color='C2', lw=lw, alpha=alpha)
    ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], '-', color='C0', lw=lw, alpha=alpha)
    if label:
        ax.text(o[0], o[1], o[2], f' {label}', fontsize=10)

def _draw_checkerboard(ax, T_W_Board, rows, cols, square, color='k', alpha=0.15):
    """보드 평면(사각형 테두리 + 코너 점들) 표시. rows x cols, 한 변 square(m)."""
    # 보드 좌표계에서 코너들 생성 (원점은 보드 모서리라고 가정)
    xs = np.arange(cols) * square
    ys = np.arange(rows) * square
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X).ravel(), np.ones_like(X).ravel()], axis=0)  # 4xN
    W = (T_W_Board @ P)[:3, :]  # 3xN
    ax.scatter(W[0,:], W[1,:], W[2,:], s=8, c=color, alpha=0.6)

    # 외곽 테두리(직사각형) 그리기
    poly_B = np.array([
        [0,          0,           0, 1],
        [cols*square,0,           0, 1],
        [cols*square,rows*square, 0, 1],
        [0,          rows*square, 0, 1],
        [0,          0,           0, 1],
    ]).T  # 4x5
    poly_W = (T_W_Board @ poly_B)[:3,:]
    ax.plot(poly_W[0,:], poly_W[1,:], poly_W[2,:], '-', color=color, alpha=0.8)

def _draw_camera_frustum(ax, T_W_C, intr, depth=0.2, alpha=0.15):
    """
    간단한 카메라 프러스텀 (폭/높이는 intr의 센서 스케일을 기반으로 근사).
    - depth: 프러스텀 길이(m)
    """
    # 센서 스케일 근사: f = c, 픽셀 스케일 sx, sy (m/px) → 시야각 근사
    # 화면 반폭/반높이를 픽셀 기준으로 가정하고, focal=c / sx, c / sy로 근사
    # 여기선 '표현' 목적이라 간단히 FoV 60° (반각 30°) 정도로 그릴 수도 있음.
    # 보다 정확히 하려면 intr.width/height가 있어야 함.
    fov_half = np.deg2rad(30.0)
    w = depth * np.tan(fov_half)
    h = depth * np.tan(fov_half)

    # 카메라 좌표계에서 프러스텀 4 코너(앞면 z=+depth)
    corners_C = np.array([
        [ 0,   0,    0],          # origin
        [-w, -h,  depth],
        [ w, -h,  depth],
        [ w,  h,  depth],
        [-w,  h,  depth],
    ]).T  # 3x5
    corners_C = np.vstack([corners_C, np.ones((1, corners_C.shape[1]))])  # 4x5
    corners_W = (T_W_C @ corners_C)[:3, :]

    o = corners_W[:,0]
    a,b,c,d = corners_W[:,1], corners_W[:,2], corners_W[:,3], corners_W[:,4]

    # 면과 엣지
    ax.plot([o[0], a[0]], [o[1], a[1]], [o[2], a[2]], color='k', alpha=alpha)
    ax.plot([o[0], b[0]], [o[1], b[1]], [o[2], b[2]], color='k', alpha=alpha)
    ax.plot([o[0], c[0]], [o[1], c[1]], [o[2], c[2]], color='k', alpha=alpha)
    ax.plot([o[0], d[0]], [o[1], d[1]], [o[2], d[2]], color='k', alpha=alpha)
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='k', alpha=alpha)
    ax.plot([b[0], c[0]], [b[1], c[1]], [b[2], c[2]], color='k', alpha=alpha)
    ax.plot([c[0], d[0]], [c[1], d[1]], [c[2], d[2]], color='k', alpha=alpha)
    ax.plot([d[0], a[0]], [d[1], a[1]], [d[2], a[2]], color='k', alpha=alpha)

def _draw_box(ax, min_xyz, max_xyz, color='gray', alpha=0.08, lw=1.0, label=None):
    """작업공간 박스 시각화(월드=B1 좌표). min/max는 B1좌표 기준."""
    x0,y0,z0 = min_xyz; x1,y1,z1 = max_xyz
    # 8 corners
    pts = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]
    ])
    # 12 edges (by index)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for (i,j) in edges:
        ax.plot([pts[i,0], pts[j,0]],[pts[i,1], pts[j,1]],[pts[i,2], pts[j,2]], color=color, lw=lw, alpha=0.7)
    if label:
        cx, cy, cz = (pts[0]+pts[6])/2
        ax.text(cx, cy, cz, f' {label}', color=color)

def plot_scene_overview(
    *,
    X_gt, Z_gt, Y_gt, T_Board_B1,
    ws1=None, ws2=None,
    intr1=None, intr2=None,
    # 한 번 샘플링된 포즈 (없으면 내부에서 샘플링)
    A_sample=None, C_sample=None, rng=None,
    cb_rows=7, cb_cols=5, cb_square_m=0.05,
    title="Scene Overview (World = B1)"
):
    """
    월드 프레임=B1. B2/Board/Cam1/Cam2, ws1/ws2를 함께 그림.
    - X_gt = ^C1 T_E1
    - Z_gt = ^C2 T_E2
    - Y_gt = ^B2 T_B1
    - T_Board_B1 = ^B1 T_B
    - A_sample = ^E1 T_B1
    - C_sample = ^E2 T_B2
    """
    rng = rng or np.random.default_rng(0)

    # 샘플 포즈 1회 생성(필요시)
    if A_sample is None and ws1 is not None:
        A_sample = ws1.sample_pose(rng)  # ^E1 T_B1
    if C_sample is None and ws2 is not None:
        C_sample = ws2.sample_pose(rng)  # ^E2 T_B2

    # 좌표 변환들을 B1(World) 기준으로 정리
    T_W_B1   = np.eye(4)                # ^B1 T_B1
    T_W_B2   = inv4(Y_gt)               # ^B1 T_B2
    T_W_B    = T_Board_B1               # ^B1 T_B
    T_W_C1   = inv4(X_gt @ A_sample)    # ^B1 T_C1,  (cam1)
    T_W_C2   = inv4(Z_gt @ C_sample @ Y_gt)  # ^B1 T_C2, (cam2)

    # 3D Figure
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # triads
    _triad(ax, T_W_B1, length=0.15, label='B1', alpha=1.0)
    _triad(ax, T_W_B2, length=0.15, label='B2', alpha=0.9)
    _triad(ax, T_W_B,  length=0.12, label='Board', alpha=0.9)
    _triad(ax, T_W_C1, length=0.10, label='C1', alpha=0.9)
    _triad(ax, T_W_C2, length=0.10, label='C2', alpha=0.9)

    # board mesh
    _draw_checkerboard(ax, T_W_B, cb_rows, cb_cols, cb_square_m, color='k', alpha=0.15)

    # camera frustum (간단 표현)
    if intr1 is not None:
        _draw_camera_frustum(ax, T_W_C1, intr1, depth=0.25, alpha=0.25)
    if intr2 is not None:
        _draw_camera_frustum(ax, T_W_C2, intr2, depth=0.25, alpha=0.25)

    # workspaces (B1 좌표계 기준 min/max가 들어온다고 가정)
    # - ws 객체에 get_bbox_in_B1(min,max) 같은 메서드가 있으면 그걸 사용
    # - 없다면 ws1/ws2가 B1 기준 박스(min,max)를 이미 갖고 있다고 가정
    try:
        if ws1 is not None and hasattr(ws1, "bbox_min") and hasattr(ws1, "bbox_max"):
            _draw_box(ax, ws1.bbox_min, ws1.bbox_max, color='tab:blue', label='ws1')
        if ws2 is not None and hasattr(ws2, "bbox_min") and hasattr(ws2, "bbox_max"):
            _draw_box(ax, ws2.bbox_min, ws2.bbox_max, color='tab:orange', label='ws2')
    except Exception:
        pass

    # 보기 영역 세팅
    all_pts = []
    for T in [T_W_B1, T_W_B2, T_W_B, T_W_C1, T_W_C2]:
        all_pts.append(T[:3,3])
    P = np.stack(all_pts, axis=0)
    c = P.mean(axis=0); r = np.max(np.linalg.norm(P - c, axis=1)) + 0.5
    ax.set_xlim([c[0]-r, c[0]+r])
    ax.set_ylim([c[1]-r, c[1]+r])
    ax.set_zlim([c[2]-r*0.5, c[2]+r*0.8])

    ax.set_xlabel('X (B1)'); ax.set_ylabel('Y (B1)'); ax.set_zlabel('Z (B1)')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()

# ========================== Division 카메라 모델 ===============================
@dataclass
class DivisionIntrinsics:
    width: int = 1280
    height: int = 1024
    c: float = 8.43          # focal [mm]
    kappa: float = 0.001     # [mm^-2]
    sx: float = 5.21e-3      # [mm/px]
    sy: float = 5.20e-3      # [mm/px]
    cx: float = 660.0        # [px]
    cy: float = 482.0        # [px]

def _project_point_division(pc, c, k, sx, sy, cx, cy):
    X, Y, Z = pc
    if Z <= 1e-12:
        return np.nan, np.nan
    ux, uy = c*(X/Z), c*(Y/Z)  # [mm]
    ru2 = ux*ux + uy*uy
    if abs(k) < 1e-15 or ru2 < 1e-24:
        xd, yd = ux, uy
    else:
        g = 1.0 - 4.0*k*ru2
        if g <= 0.0:
            # 물리적으로 해가 없는 구간(ru > 1/(2√k)) → NaN 처리 또는 클램프
            return np.nan, np.nan
        Delta = np.sqrt(g)
        s = 2.0 / (1.0 + Delta)      # == rd/ru
        xd, yd = s*ux, s*uy
    return xd/sx + cx, yd/sy + cy

def project_division_model(pts_3d, T_Board_Cam, intr: DivisionIntrinsics):
    pts_cam = (T_Board_Cam[:3,:3] @ pts_3d.T + T_Board_Cam[:3,3,None]).T
    uv = np.array([_project_point_division(pc, intr.c, intr.kappa, intr.sx, intr.sy, intr.cx, intr.cy)
                   for pc in pts_cam])
    mask = ~np.isnan(uv).any(axis=1)
    return uv, mask

def undistort_points_division(distorted_pts: np.ndarray, intr: DivisionIntrinsics):
    """
    Division 역왜곡(정확식):
      ru = rd / (1 + k * rd^2)
      (xu, yu) = (xd, yd) * (ru/rd) = (xd, yd) / (1 + k * rd^2)
    """
    xd = (distorted_pts[:,0] - intr.cx) * intr.sx
    yd = (distorted_pts[:,1] - intr.cy) * intr.sy
    rd2 = xd**2 + yd**2
    den = 1.0 + intr.kappa * rd2
    den = np.where(np.abs(den) < 1e-12, np.sign(den)*1e-12, den)
    scale = 1.0 / den
    xu = xd * scale
    yu = yd * scale
    undist_px = np.stack([xu/intr.sx + intr.cx, yu/intr.sy + intr.cy], axis=1)
    return undist_px

# ============================ 표의 GT / EST 예시 ==============================
def m_inv2_to_mm_inv2(k_m_inv2: float) -> float:
    return k_m_inv2 / (1000.0**2)

TABLE_GT_INTR = DivisionIntrinsics(
    width=1280, height=1024,
    c=8.4300, kappa=m_inv2_to_mm_inv2(1000.00),
    sx=5.21000e-3, sy=5.20000e-3, cx=660.00, cy=482.00
)
TABLE_EST_INTR = DivisionIntrinsics(
    width=1280, height=1024,
    c=8.4303, kappa=m_inv2_to_mm_inv2(999.92),
    sx=5.20997e-3, sy=5.20000e-3, cx=659.99, cy=481.96
)

TABLE_GT_INTR_CAM2 = DivisionIntrinsics(
    width=1280, height=1024,
    c=8.4350, kappa=m_inv2_to_mm_inv2(1050.00),
    sx=5.21100e-3, sy=5.20200e-3, cx=662.50, cy=480.50
)

TABLE_EST_INTR_CAM2 = DivisionIntrinsics(
    width=1280, height=1024,
    c=8.4352, kappa=m_inv2_to_mm_inv2(1049.90),
    sx=5.21098e-3, sy=5.20199e-3, cx=662.48, cy=480.53
)

# ======================= 노이즈/작업공간/뷰 제약 ==============================
@dataclass
class NoiseCfg:
    t_m: float = 1e-3
    rot_deg: float = 0.1
    pixel_std: float = 0.1

@dataclass
class WorkspaceBox:
    center_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.5])
    size_m:   list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    def sample_pose(self, rng):
        u = rng.uniform(0,1,3)
        q = np.array([
            np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]),
            np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
            np.sqrt(u[0])   * np.sin(2*np.pi*u[2]),
            np.sqrt(u[0])   * np.cos(2*np.pi*u[2]),
        ])
        x,y,z,w = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
        ], dtype=float)
        t = np.array(self.center_m) + (rng.random(3)-0.5)*np.array(self.size_m)
        return se3(R, t)

@dataclass
class ViewConstraint:
    min_visible_ratio: float = 0.9
    max_tilt_deg: float = 50.0
    def check(self, T_Board_Cam, obj_pts, intr: DivisionIntrinsics,
            in_ratio_thresh: float | None = None,
            margin_px: int = 0,
            min_hull_area_px: float = 0.0) -> bool:
        """
        - in_ratio_thresh: 프레임 내부 비율(기본값: self.min_visible_ratio)
        - margin_px: 프레임 가장자리 여유(0이면 경계까지 허용)
        - min_hull_area_px: 보드 픽셀 면적 최소치(너무 멀거나 평행해 PnP가 불안정한 경우 컷)
        """
        # 1) 틸트 제한
        T_Cam_Board = inv4(T_Board_Cam)
        tilt_deg = np.rad2deg(np.arccos(np.clip(T_Cam_Board[2,2], -1.0, 1.0)))
        if tilt_deg > self.max_tilt_deg:
            return False

        # 2) 투영 가능(양의 깊이 & division 해 유효)
        uv, mask = project_division_model(obj_pts, T_Board_Cam, intr)
        if not mask.any():
            return False

        # 3) 프레임 내부(in-bounds) 검사
        h, w = intr.height, intr.width
        m = mask.copy()
        # 이미지 경계 + 여유 margin
        inb = (
            (uv[:,0] >= margin_px) & (uv[:,0] <= w - 1 - margin_px) &
            (uv[:,1] >= margin_px) & (uv[:,1] <= h - 1 - margin_px)
        )
        valid = m & inb

        # 비율 기준 (기본은 self.min_visible_ratio 사용)
        thr = self.min_visible_ratio if in_ratio_thresh is None else in_ratio_thresh
        if valid.mean() < thr:
            return False

        # 4) 보드의 "픽셀 크기"가 너무 작지 않은지(조건수 방지)
        if min_hull_area_px > 0:
            pts = uv[valid]
            if len(pts) >= 3:
                # 간단한 컨벡스 헐 면적
                hull = cv2.convexHull(pts.astype(np.float32))
                area = cv2.contourArea(hull)
                if area < min_hull_area_px:
                    return False

        return True

def _as_3vec(x, angle=False):
    """
    x: scalar or length-3 iterable.
    angle=True면 deg -> rad 변환.
    """
    if np.isscalar(x):
        v = np.array([x, x, x], dtype=float)
    else:
        v = np.asarray(x, dtype=float)
        assert v.shape == (3,), "must be scalar or length-3"
    if angle:
        v = np.deg2rad(v)
    return v

def perturb_robot_pose_gmf(
    T_gt,
    rng,
    sigma_angle_deg=0.1,     # float or (σ_α, σ_β, σ_γ) [deg]
    sigma_trans_m=0.001,      # float or (σ_tx, σ_ty, σ_tz) [m]
    # 선택: 상관 포함하고 싶으면 공분산 행렬을 직접 넣음(라디안/미터 단위)
    cov_angle_rad=None,       # 3x3 or None
    cov_trans_m=None          # 3x3 or None
):
    """
    논문 가정: 관측된 Euler XYZ 각과 병진 성분 자체가 잡음.
    축별 표준편차 또는 공분산을 지원.
    """
    R_gt, t_gt = T_gt[:3,:3], T_gt[:3,3]
    a, b, g = R_to_euler_xyz(R_gt)

    if cov_angle_rad is not None:
        d_ang = rng.multivariate_normal(mean=np.zeros(3), cov=np.asarray(cov_angle_rad))
    else:
        sig_ang = _as_3vec(sigma_angle_deg, angle=True)     # rad
        d_ang = rng.normal(0.0, sig_ang)

    if cov_trans_m is not None:
        d_t = rng.multivariate_normal(mean=np.zeros(3), cov=np.asarray(cov_trans_m))
    else:
        sig_t = _as_3vec(sigma_trans_m, angle=False)        # m
        d_t = rng.normal(0.0, sig_t)

    a_m, b_m, g_m = a + d_ang[0], b + d_ang[1], g + d_ang[2]
    t_m = t_gt + d_t
    R_m = euler_xyz_to_R(a_m, b_m, g_m)
    return se3(R_m, t_m)

# ============================= 보조 함수들 =====================================
def pose_error_deg_mm(T_gt: np.ndarray, T_est: np.ndarray) -> tuple[float, float]:
    T = inv4(T_gt) @ T_est
    r_deg = np.rad2deg(np.linalg.norm(log_so3(T[:3,:3])))
    t_mm = np.linalg.norm(T[:3,3]) * 1000.0
    return r_deg, t_mm

def vis_points(save_path: Path,
               img_size: tuple[int, int],
               meas_px: np.ndarray,
               proj_px: np.ndarray,
               title: str,
               verbose: bool = True):
    # img_size: (height, width) 로 전달한다고 가정
    h, w = int(img_size[0]), int(img_size[1])

    def _finite_inbounds(px):
        px = np.asarray(px, dtype=float)
        finite = np.isfinite(px).all(axis=1)
        px = px[finite]
        if px.size == 0:
            return px, 0, 0
        inb = (px[:,0] >= 0) & (px[:,0] <= w) & (px[:,1] >= 0) & (px[:,1] <= h)
        return px, int(inb.sum()), int(px.shape[0] - inb.sum())

    meas_px_f, m_in, m_out = _finite_inbounds(meas_px)
    proj_px_f, p_in, p_out = _finite_inbounds(proj_px)

    if verbose:
        print(f"[vis] {title}")
        print(f"      meas: total={len(meas_px)}, finite={len(meas_px_f)}, in={m_in}, out={m_out}")
        print(f"      proj: total={len(proj_px)}, finite={len(proj_px_f)}, in={p_in}, out={p_out}")

        if len(meas_px_f):
            print(f"      meas x[min,max]=[{meas_px_f[:,0].min():.1f},{meas_px_f[:,0].max():.1f}], "
                  f"y[min,max]=[{meas_px_f[:,1].min():.1f},{meas_px_f[:,1].max():.1f}]")
        if len(proj_px_f):
            print(f"      proj x[min,max]=[{proj_px_f[:,0].min():.1f},{proj_px_f[:,0].max():.1f}], "
                  f"y[min,max]=[{proj_px_f[:,1].min():.1f},{proj_px_f[:,1].max():.1f}]")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))

    # 점이 0개라도 에러 없이 처리
    if len(meas_px_f):
        plt.scatter(meas_px_f[:, 0], meas_px_f[:, 1], s=20, label='measured (GT-distorted)',
                    alpha=0.9, linewidths=0, marker='o')
    if len(proj_px_f):
        plt.scatter(proj_px_f[:, 0], proj_px_f[:, 1], s=20, label='reprojected (EST intrinsics)',
                    alpha=0.9, linewidths=0, marker='x')

    ax = plt.gca()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)   # y축만 한 번 뒤집기(이 줄만 사용; invert_yaxis()는 사용하지 않음)

    # 프레임 경계선 표시(시각적으로 도움)
    ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, linestyle='--', linewidth=1.0))

    plt.legend(loc='upper right'); plt.title(title); plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

# =================== IPPE 기반 후보 선택 유틸리티 ============================
def pick_best_pose_from_candidates(
    obj_pts: np.ndarray,
    meas_px_distorted: np.ndarray,
    K: np.ndarray,
    intr_est: DivisionIntrinsics,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    후보 해들 중에서 (1) cheirality 만족, (2) division(EST) 재투영 RMSE 최소인 해 선택
    반환: (R, t) 또는 None
    """
    best = None; best_rmse = np.inf
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv); T_Cam_B = se3(R, tv)
        # cheirality: 대부분 Z>0
        P = (T_Cam_B[:3,:3] @ obj_pts.T + T_Cam_B[:3,3,None]).T
        if np.mean(P[:,2] > 0) < 0.9:
            continue
        # 왜곡공간(division, EST)에서 재투영 RMSE
        T_Board_Cam = T_Cam_B
        proj_px, m = project_division_model(obj_pts, T_Board_Cam, intr_est)
        if not m.any():
            continue
        e = meas_px_distorted[m] - proj_px[m]
        rmse = float(np.sqrt(np.mean(np.sum(e*e, axis=1))))
        if rmse < best_rmse:
            best_rmse = rmse
            best = (R, tv)
    return best

# ======================= 시뮬레이션 & 데이터 저장 =============================
def simulate_and_save_realistic_data(
    out_root,
    num_samples=50,
    *,
    X_gt=None, Z_gt=None, Y_gt=None, T_Board_B1=None,
    workspace1=None, workspace2=None, view_constraint=None,
    # ✅ 측정 생성용(ground-truth) intrinsics
    intr1_gt: DivisionIntrinsics = TABLE_GT_INTR,
    intr2_gt: DivisionIntrinsics = TABLE_GT_INTR,
    # ✅ PnP/재투영/시각화에 사용할 추정 intrinsics
    intr1_est: DivisionIntrinsics = TABLE_EST_INTR,
    intr2_est: DivisionIntrinsics = TABLE_EST_INTR,
    cb_rows=7, cb_cols=5, cb_square_m=0.05,
    noise: NoiseCfg = NoiseCfg(),
    rng: np.random.Generator | None = None,
    do_visualize: bool = True,
):
    """
    - 측정값(image_points) 생성: GT intrinsics 사용
    - PnP/언디스토션/재투영/시각화/ RMSE: Estimated intrinsics 사용
    """
    rng = rng or np.random.default_rng(0)
    ws1, ws2 = workspace1, workspace2
    vc = view_constraint
    obj_pts = make_checkerboard_corners(cb_rows, cb_cols, cb_square_m)

    try:
        A_demo = ws1.sample_pose(rng) if ws1 is not None else np.eye(4)  # ^E1 T_B1
        C_demo = ws2.sample_pose(rng) if ws2 is not None else np.eye(4)  # ^E2 T_B2
        plot_scene_overview(
            X_gt=X_gt, Z_gt=Z_gt, Y_gt=Y_gt, T_Board_B1=T_Board_B1,
            ws1=ws1, ws2=ws2,
            intr1=intr1_est, intr2=intr2_est,
            A_sample=A_demo, C_sample=C_demo, rng=rng,
            cb_rows=cb_rows, cb_cols=cb_cols, cb_square_m=cb_square_m,
            title="Scene Overview (first sampled pose)"
        )
    except Exception as e:
        print(f"[WARN] scene overview plot failed: {e}")

    print("\n[단계 1] GT 및 노이즈 포함 관측 데이터 생성 중 (GT intrinsics)...")
    frames = []
    while len(frames) < num_samples:
        # cam1
        A_gt, B_gt = None, None
        for _ in range(200):
            T_B1E1 = ws1.sample_pose(rng)
            A = T_B1E1                 # ^E1 T_B1
            B = X_gt @ A @ T_Board_B1   # ^B T_C1
            # _print_proj_debug("cam1", B, obj_pts, intr1_gt)
            if vc.check(B, obj_pts, intr1_gt):
                A_gt, B_gt = A, B; break
        if A_gt is None:
            # print("A NOT CREATED")
            continue

        # cam2
        C_gt, D_gt = None, None
        for _ in range(200):
            T_B2E2 = ws2.sample_pose(rng)
            C = T_B2E2                 # ^E2 T_B2
            D = Z_gt @ C @ Y_gt @ T_Board_B1   # ^B T_C2
            if vc.check(D, obj_pts, intr2_gt):
                C_gt, D_gt = C, D; break
        if C_gt is None:
            # print("C NOT CREATED")
            continue

        img1_gt, _ = project_division_model(obj_pts, B_gt, intr1_gt)
        img2_gt, _ = project_division_model(obj_pts, D_gt, intr2_gt)
        img1_meas = img1_gt + rng.normal(0, noise.pixel_std, img1_gt.shape)
        img2_meas = img2_gt + rng.normal(0, noise.pixel_std, img2_gt.shape)

        frames.append(dict(A_gt=A_gt, B_gt=B_gt, C_gt=C_gt, D_gt=D_gt,
                           img1_meas=img1_meas, img2_meas=img2_meas))
        print(f"\r  - GT 데이터 생성 완료: {len(frames)}/{num_samples}", end="")
    print("\n✅ GT 데이터 생성 완료.")

    # --- PnP는 Estimated intrinsics 사용 ---
    K1 = np.array([[intr1_est.c/intr1_est.sx, 0, intr1_est.cx],
                   [0, intr1_est.c/intr1_est.sy, intr1_est.cy],
                   [0, 0, 1]], dtype=np.float64)
    K2 = np.array([[intr2_est.c/intr2_est.sx, 0, intr2_est.cx],
                   [0, intr2_est.c/intr2_est.sy, intr2_est.cy],
                   [0, 0, 1]], dtype=np.float64)

    print("\n[단계 2] PnP(IPPE, Estimated intrinsics) + 저장/시각화...")
    out_root = Path(out_root)
    
    if out_root.exists():
        print(f"\n[INFO] 기존 데이터 폴더({out_root})를 삭제합니다...")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 데이터 폴더({out_root})를 새로 생성했습니다.")

    (out_root/"cam1"/"poses").mkdir(parents=True, exist_ok=True)
    (out_root/"cam2"/"poses").mkdir(parents=True, exist_ok=True)
    (out_root/"vis"/"cam1").mkdir(parents=True, exist_ok=True)
    (out_root/"vis"/"cam2").mkdir(parents=True, exist_ok=True)

    rot_err_deg_c1, trans_err_mm_c1, rmse_px_c1 = [], [], []
    rot_err_deg_c2, trans_err_mm_c2, rmse_px_c2 = [], [], []

    for i, f in enumerate(frames):
        # 로봇 관측 외란 (옵션)

        A_meas = perturb_robot_pose_gmf(
            f['A_gt'], rng,
            sigma_angle_deg=noise.rot_deg,     # 스칼라면 등방성 (각 축 동일)
            sigma_trans_m=noise.t_m            # 스칼라면 등방성 (각 축 동일)
        )
        C_meas = perturb_robot_pose_gmf(
            f['C_gt'], rng,
            sigma_angle_deg=noise.rot_deg,
            sigma_trans_m=noise.t_m
        )

        # ✅ 언디스토션: Estimated intrinsics로 수행
        img1_u = undistort_points_division(f['img1_meas'], intr1_est)
        img2_u = undistort_points_division(f['img2_meas'], intr2_est)

        # ====== IPPE 기반 후보 해 계산 ======
        ret1 = cv2.solvePnPGeneric(obj_pts.astype(np.float32),
                                   img1_u.astype(np.float32),
                                   K1, None,
                                   flags=cv2.SOLVEPNP_IPPE)
        # OpenCV 버전에 따라 반환 길이가 다를 수 있으므로 안전 해석
        ok1 = ret1[0] if isinstance(ret1[0], (bool, np.bool_)) else (len(ret1[1]) > 0)
        rvecs1 = ret1[1] if ok1 else []
        tvecs1 = ret1[2] if ok1 else []

        ret2 = cv2.solvePnPGeneric(obj_pts.astype(np.float32),
                                   img2_u.astype(np.float32),
                                   K2, None,
                                   flags=cv2.SOLVEPNP_IPPE)
        ok2 = ret2[0] if isinstance(ret2[0], (bool, np.bool_)) else (len(ret2[1]) > 0)
        rvecs2 = ret2[1] if ok2 else []
        tvecs2 = ret2[2] if ok2 else []

        # 후보 중 best pick (cheirality + division 재투영 RMSE 최소)
        best1 = pick_best_pose_from_candidates(obj_pts, f['img1_meas'], K1, intr1_est, rvecs1, tvecs1)
        best2 = pick_best_pose_from_candidates(obj_pts, f['img2_meas'], K2, intr2_est, rvecs2, tvecs2)

        # fallback: ITERATIVE
        if best1 is None:
            ok, r, t = cv2.solvePnP(obj_pts.astype(np.float32), img1_u.astype(np.float32), K1, None, flags=cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(r); best1 = (R, t)
        if best2 is None:
            ok, r, t = cv2.solvePnP(obj_pts.astype(np.float32), img2_u.astype(np.float32), K2, None, flags=cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(r); best2 = (R, t)

        R1, t1 = best1
        R2, t2 = best2
        T_C1B_pnp = se3(R1, t1)
        T_C2B_pnp = se3(R2, t2)

        # 포즈 에러(참조는 GT 포즈)
        rdeg1, tmm1 = pose_error_deg_mm(f['B_gt'], T_C1B_pnp)
        rdeg2, tmm2 = pose_error_deg_mm(f['D_gt'], T_C2B_pnp)
        rot_err_deg_c1.append(rdeg1); trans_err_mm_c1.append(tmm1)
        rot_err_deg_c2.append(rdeg2); trans_err_mm_c2.append(tmm2)

        # ✅ 재투영도 Estimated intrinsics 사용(현실과 동일하게)
        T_BC1_est = T_C1B_pnp
        T_BC2_est = T_C2B_pnp
        B_gt = X_gt @ f['A_gt'] @ T_Board_B1
        D_gt = Z_gt @ f['C_gt'] @ Y_gt @ T_Board_B1
        proj1gt_px, m1gt = project_division_model(obj_pts, B_gt, intr1_gt)
        proj2gt_px, m2gt = project_division_model(obj_pts, D_gt, intr2_gt)
        B_est = X_gt @ A_meas @ T_Board_B1
        proj1_px, m1 = project_division_model(obj_pts, T_BC1_est, intr1_est)
        proj2_px, m2 = project_division_model(obj_pts, T_BC2_est, intr2_est)

        e1gt = f['img1_meas'][m1gt] - proj1gt_px[m1gt]
        e2gt = f['img2_meas'][m2gt] - proj2gt_px[m2gt]
        e1 = f['img1_meas'][m1] - proj1_px[m1]
        e2 = f['img2_meas'][m2] - proj2_px[m2]
        rmsegt1 = float(np.sqrt(np.mean(np.sum(e1gt**2, axis=1)))) if m1gt.any() else np.nan
        rmsegt2 = float(np.sqrt(np.mean(np.sum(e2gt**2, axis=1)))) if m2gt.any() else np.nan
        rmse1 = float(np.sqrt(np.mean(np.sum(e1**2, axis=1)))) if m1.any() else np.nan
        rmse2 = float(np.sqrt(np.mean(np.sum(e2**2, axis=1)))) if m2.any() else np.nan
        rmse_px_c1.append(rmse1)
        rmse_px_c2.append(rmse2)

        print(f"[Frame {i:03d}] cam1: rot={rdeg1:6.3f} deg, trans={tmm1:7.3f} mm, RMSE={rmse1:7.3f} px, ,RMSE(gt)={rmsegt1:7.3f} px | "
              f"cam2: rot={rdeg2:6.3f} deg, trans={tmm2:7.3f} mm, RMSE={rmse2:7.3f} px, RMSE(gt)={rmsegt2:7.3f} px")

        # 🔎 GT 체인 잔차:  AXB ?= YCZD
        #  - 스코프에 X_gt, Y_gt, Z_gt가 없다면 f['X_gt'] 등으로 바꿔 써도 됩니다.
        T_L = inv4(A_meas) @ inv4(X_gt) @ f['B_gt']
        T_R = inv4(Y_gt) @ inv4(C_meas) @ inv4(Z_gt) @ f['D_gt']

        T_L = inv4(A_meas) @ inv4(X_gt) @ T_C1B_pnp
        T_R = inv4(Y_gt) @ inv4(C_meas) @ inv4(Z_gt) @ T_C2B_pnp
        # T_R = inv4(Y_gt) @ inv4(C_meas) @ inv4(Z_gt) @ T_C2B_pnp
        # T_L = inv4(A_meas) @ inv4(X_gt) @ f['B_gt']
        # T_R = inv4(Y_gt) @ inv4(C_meas) @ inv4(Z_gt) @ f['D_gt']
        chain_rdeg, chain_tmm = pose_error_deg_mm(T_L, T_R)

        print(
            f"[Frame {i:03d}] cam1: rot={rdeg1:6.3f} deg, trans={tmm1:7.3f} mm |"
            f"AXB vs YCZD: ΔR={chain_rdeg:6.3f} deg, Δt={chain_tmm:7.3f} mm"
        )

        # 시각화 저장
        if do_visualize:
            vis_points(out_root/"vis"/"cam1"/f"frame_{i:04d}.png",
                       (intr1_est.height, intr1_est.width),
                       f['img1_meas'], proj1_px,
                       f"cam1 frame {i} (EST reproject vs GT-measured)",
                       False)
            # vis_points(out_root/"vis"/"cam2"/f"frame_{i:04d}.png",
            #            (intr2_est.height, intr2_est.width),
            #            f['img2_meas'], proj2_px,
            #            f"cam2 frame {i} (EST reproject vs GT-measured)",
            #            False)

        # YAML 저장 (파이프라인 호환)
        gt_blob = {"GT": {"X": X_gt.tolist(), "Z": Z_gt.tolist(), "Y": Y_gt.tolist(),
                            "T_Base1_to_EE1": f['A_gt'].tolist(), "T_Cam1_to_Board": f['B_gt'].tolist(),
                            "T_Base2_to_EE2": f['C_gt'].tolist(), "T_Cam2_to_Board": f['D_gt'].tolist()}}
        data1 = {
            "T_Base_to_EE": inv4(A_meas).tolist(),
            # "T_Cam_to_Board": T_C1B_pnp.tolist(),
            "T_Cam_to_Board": f['B_gt'].tolist(),
            "object_points": obj_pts.tolist(),
            "image_points": f['img1_meas'].tolist(),
            **gt_blob
        }
        data2 = {
            "T_Base_to_EE": inv4(C_meas).tolist(),
            # "T_Cam_to_Board": T_C2B_pnp.tolist(),
            "T_Cam_to_Board": f['D_gt'].tolist(),
            "object_points": obj_pts.tolist(),
            "image_points": f['img2_meas'].tolist(),
            **gt_blob
        }
        with open(out_root/"cam1"/"poses"/f"frame_{i:04d}.yaml", "w") as fp: yaml.safe_dump(data1, fp, indent=2)
        with open(out_root/"cam2"/"poses"/f"frame_{i:04d}.yaml", "w") as fp: yaml.safe_dump(data2, fp, indent=2)

    # 요약
    def summarize(name, r_list, t_list, rmse_list):
        r = np.array(r_list); t = np.array(t_list); e = np.array(rmse_list)
        print(f"\n[{name}] Summary over {len(r)} frames")
        print(f"  Rotation error : mean={np.mean(r):.4f} deg | p95={np.percentile(r,95):.4f} | max={np.max(r):.4f}")
        print(f"  Translation err: mean={np.mean(t):.3f} mm  | p95={np.percentile(t,95):.3f} | max={np.max(t):.3f}")
        print(f"  RMSE (pixels)  : mean={np.mean(e):.4f} px  | p95={np.percentile(e,95):.4f} | max={np.max(e):.4f}")

    summarize("cam1 (B_gt vs PnP)", rot_err_deg_c1, trans_err_mm_c1, rmse_px_c1)
    # summarize("cam2 (D_gt vs PnP)", rot_err_deg_c2, trans_err_mm_c2, rmse_px_c2)
    print("\n✅ 최종 데이터 저장 & 시각화 완료.")

# ================================ 실행부 =======================================
if __name__ == "__main__":
    rng = np.random.default_rng(None)

    # GT kinematics
    X_gt = se3(
        so3_exp([ 0.1, 0.1, -np.pi/2.0]),   # ~ 1–2°,  -1.7°, 0.6°
        [0.06, 0.04, 0.10]                # 6cm right, 10cm forward (EE frame 기준)
    )

    Z_gt = se3(
        so3_exp([-0.1,  0.4, -np.pi/2.0]),   # ~ -0.6°,  2.3°, -1.1°
        [0.06, 0.00, 0.10]
    )

    # B1 -> B2 (우리가 쓰는 건 Y = ^B2 T_B1 임에 주의!)
    # 약 30cm baseline + 약 8.6° yaw 차이
    Y_gt = se3(
        so3_exp([0.0, 0.0, 0.0]),        # ~ 8.6° yaw
        [0.25, 0.00, 0.00]                # B2 프레임에서 본 B1의 위치가 +X 방향 0.3m
    )

    # 보드 위치: B1 기준 정면 0.8m
    T_Board_B1_gt = se3(
        so3_exp([0.0, 0.0, np.pi]),         # 보드 정면
        [0.0, 0.8, 0.5]                   # B1의 +Z가 전방이라고 가정 (전방축이 +Y라면 [0, 0.8, 0]로 바꿔)
    )

    # experiment flag
    # 1: same intrinsic
    # 2: different intrinsic
    experiment_flag = 1

    if experiment_flag == 1:
        print("[INFO] Generating Experiment 1 Data: Two cameras with the same intrinsics.")
        sim_params = {
            'workspace1': WorkspaceBox(),
            'workspace2': WorkspaceBox(),
            'view_constraint': ViewConstraint(min_visible_ratio=1.0, max_tilt_deg=25.0),
            # ✅ 측정 생성용 GT intrinsics
            'intr1_gt': TABLE_GT_INTR,
            'intr2_gt': TABLE_GT_INTR,
            # ✅ PnP/재투영/시각화용 Estimated intrinsics
            'intr1_est': TABLE_EST_INTR,
            'intr2_est': TABLE_EST_INTR,
            'cb_rows': 7, 'cb_cols': 6, 'cb_square_m': 0.08,
        }
    elif experiment_flag == 2:
        print("[INFO] Generating Experiment 2 Data: Two cameras with different intrinsics.")
        sim_params = {
            'workspace1': WorkspaceBox(),
            'workspace2': WorkspaceBox(),
            'view_constraint': ViewConstraint(min_visible_ratio=1.0, max_tilt_deg=25.0),
            # ✅ 측정 생성용 GT intrinsics
            'intr1_gt': TABLE_GT_INTR,
            'intr2_gt': TABLE_GT_INTR_CAM2,
            # ✅ PnP/재투영/시각화용 Estimated intrinsics
            'intr1_est': TABLE_EST_INTR,
            'intr2_est': TABLE_EST_INTR_CAM2,
            'cb_rows': 7, 'cb_cols': 6, 'cb_square_m': 0.08,
        }

    DATA_ROOT = "./data_dual_cam_realistic"

    simulate_and_save_realistic_data(
        out_root=DATA_ROOT,
        num_samples=20,
        X_gt=X_gt, Z_gt=Z_gt, Y_gt=Y_gt, T_Board_B1=T_Board_B1_gt,
        noise=NoiseCfg(t_m=0.001, rot_deg=0.1, pixel_std=0.1),
        rng=rng,
        **sim_params,
        do_visualize=True
    )
