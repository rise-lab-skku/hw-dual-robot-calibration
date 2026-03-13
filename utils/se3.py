import numpy as np
import cv2
from numpy.random import default_rng
from scipy.linalg import logm, expm

rng = default_rng(42)

# -- SE(3) 유틸리티 --
def hat3(w):
    wx, wy, wz = w
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]], float)


def so3_exp(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3)
    W = hat3(phi)
    a = np.sin(theta)/theta
    b = (1-np.cos(theta))/(theta**2)
    return np.eye(3) + a*W + b*(W@W)


def se3_exp(xi):
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi)
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        V = np.eye(3)
    else:
        W = hat3(phi)
        b = (1-np.cos(theta))/(theta**2)
        c = (theta-np.sin(theta))/(theta**3)
        V = np.eye(3) + b*W + c*(W@W)
    t = V @ rho
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T


def vee(S):
    """Maps a 4x4 se(3) matrix to a 6x1 twist vector."""
    return np.array([S[0,3], S[1,3], S[2,3], S[2,1], S[0,2], S[1,0]])


def hat(xi):
    """Maps a 6x1 twist vector to a 4x4 se(3) matrix."""
    v, w = xi[:3], xi[3:]
    w_hat = hat3(w) # Reuse existing hat3
    S = np.zeros((4,4))
    S[:3,:3] = w_hat
    S[:3,3] = v
    return S


def trlog(T):
    """Log map SE(3) -> se(3) (4x4 matrix). Uses scipy.linalg.logm."""
    return logm(T)


def trexp(xi):
    """Exp map se(3) -> SE(3) (4x4 matrix from 6x1 twist). Uses scipy.linalg.expm."""
    return expm(hat(xi))


def Ad(T):
    """Computes the 6x6 Adjoint matrix of a 4x4 transformation T."""
    R = T[:3,:3]
    p = T[:3,3]
    p_hat = hat3(p) # Use hat3 for the skew-symmetric part of p
    Ad_mat = np.zeros((6,6))
    Ad_mat[:3,:3] = R
    Ad_mat[3:,3:] = R
    Ad_mat[:3,3:] = p_hat @ R
    return Ad_mat


def ad(xi):
    """Computes the 6x6 adjoint (Lie bracket) of a 6x1 twist xi."""
    v, w = xi[:3], xi[3:]
    w_hat = hat3(w)
    v_hat = hat3(v)
    ad_mat = np.zeros((6,6))
    ad_mat[:3,:3] = w_hat
    ad_mat[3:,3:] = w_hat
    ad_mat[:3,3:] = v_hat
    return ad_mat


def left_jacobian_se3(xi):
    """Computes the exact 6x6 left Jacobian J_l(xi)."""
    w = xi[3:]
    phi = np.linalg.norm(w)
    A = ad(xi)

    if phi < 1e-8: # Use Taylor series for small angles
        A2 = A @ A; A3 = A2 @ A; A4 = A3 @ A
        return np.eye(6) + 0.5*A + (1/6)*A2 + (1/24)*A3 + (1/120)*A4

    s = np.sin(phi)
    c = np.cos(phi)
    phi2, phi3 = phi**2, phi**3

    # Coefficients from the formula
    a1 = (1 - c) / phi2
    a2 = (phi - s) / phi3
    
    Jl = np.eye(6) + a1 * A + a2 * (A @ A)
    return Jl


def inv_T(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti


def make_T(R,t):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T


def _vecF(M: np.ndarray) -> np.ndarray:
    """vec in column-major (Fortran) order, like MATLAB's reshape(:)."""
    return M.reshape(-1, order='F')


def _matF(v: np.ndarray, r: int, c: int) -> np.ndarray:
    """inverse of vecF: build matrix in column-major."""
    return v.reshape((r, c), order='F')


def _ortho_svd(R: np.ndarray) -> np.ndarray:
    """Project to SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def _rot_angle_deg(R):
    # 수치안정용 clamp
    c = (np.trace(R) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _pose_error(T_err):
    """T_err = T_rhs^{-1} @ T_lhs  에서 회전각(°), 평행이동노름(m)"""
    R = T_err[:3,:3]
    t = T_err[:3,3]
    return _rot_angle_deg(R), float(np.linalg.norm(t))


def _dual_rrmse_error(X, Y, Z, cfg):
    """Compute RRMSE and ground-truth errors"""
    # Example: implement evaluation logic here
    cam1_res, cam2_res = [], []
    for A, B, C, D, uv1, uv2 in zip(cfg["A"], cfg["B"], cfg["C"], cfg["D"], cfg["uv1"], cfg["uv2"]):
        # Cam1 ray
        T1 = inv_T(A @ X) @ (Y @ C @ Z @ D)
        uv1p = project(cfg['obj_pts'], T1, cfg['K1'], cfg['dist1'])
        cam1_res.append(uv1p - uv1)
        # Cam2 ray
        T2 = inv_T(Y @ C @ Z) @ (A @ X @ B)
        uv2p = project(cfg['obj_pts'], T2, cfg['K2'], cfg['dist2'])
        cam2_res.append(uv2p - uv2)
    # RMSE
    cam1_rmse = float(np.sqrt(np.mean(np.vstack(cam1_res)**2)))
    cam2_rmse = float(np.sqrt(np.mean(np.vstack(cam2_res)**2)))
    return {'cam1_rmse': cam1_rmse, 'cam2_rmse': cam2_rmse}


# -- 카메라 프로젝션 & 가시성 --
def project(pts3d, T, K, dist):
    rvec, _ = cv2.Rodrigues(T[:3,:3])
    tvec = T[:3,3].reshape(3,1)
    uv, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return uv.reshape(-1,2)


def board_visible(T, K, dist, obj_pts,
                  margin_px=8, z_range=(0.25,2.0), max_tilt_deg=80):
    cx = K[0, 2]
    cy = K[1, 2]
    W = int(2 * cx)
    H = int(2 * cy)
    R, t = T[:3,:3], T[:3,3]
    # 깊이 체크
    Pc = (R @ obj_pts.T + t[:,None]).T
    z = Pc[:,2]
    if not ((z > z_range[0]).all() and (z < z_range[1]).all()):
        return False
    # 기울기 체크
    n_cam = R @ np.array([0,0,1])
    cos_th = abs(n_cam[2])/(np.linalg.norm(n_cam)+1e-12)
    if cos_th < np.cos(np.deg2rad(max_tilt_deg)):
        return False
    # 이미지 내부 체크
    uv, _ = cv2.projectPoints(obj_pts, cv2.Rodrigues(R)[0], t.reshape(3,1), K, dist)
    uv = uv.reshape(-1,2)
    if not ((uv[:,0]>=margin_px).all() and (uv[:,0]<=W-margin_px).all() \
            and (uv[:,1]>=margin_px).all() and (uv[:,1]<=H-margin_px).all()):
        return False
    return True


# -- 랜덤 샘플링 유틸 --
def rpy_to_R(roll, pitch, yaw):
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return Rz @ Ry @ Rx


def rand_robot_pose(r_rng, z_rng, yaw_rng, pitch_rng, roll_rng):
    r = rng.uniform(*r_rng)
    yaw = rng.uniform(*yaw_rng)
    x, y = r*np.cos(yaw), r*np.sin(yaw)
    z = rng.uniform(*z_rng)
    roll = rng.uniform(*roll_rng)
    pitch = rng.uniform(*pitch_rng)
    yaw2 = rng.uniform(-np.pi, np.pi)
    # Rotation matrix
    R = rpy_to_R(roll, pitch, yaw2)
    return make_T(R, np.array([x,y,z]))


def rand_gt(dx, dy, dz, yaw_deg, pitch_deg, roll_deg):
    # Displacement
    tx = rng.uniform(-dx, dx)
    ty = rng.uniform(-dy, dy)
    tz = rng.uniform(-dz, dz)
    # Euler angles
    yaw = np.deg2rad(rng.uniform(-yaw_deg, yaw_deg))
    pitch = np.deg2rad(rng.uniform(-pitch_deg, pitch_deg))
    roll = np.deg2rad(rng.uniform(-roll_deg, roll_deg))
    R = rpy_to_R(roll, pitch, yaw)
    return make_T(R, np.array([tx,ty,tz]))


def rand_board_pose(dist_rng, x_rng, y_rng, tilt_deg):
    tz = rng.uniform(*dist_rng)
    tx = rng.uniform(*x_rng); ty = rng.uniform(*y_rng)
    ang = np.deg2rad(tilt_deg)
    phi = rng.uniform(-ang,ang,3)
    return se3_exp(np.hstack([[tx,ty,tz], phi]))


def build_obj_pts(board_cfg):
    """
    Generate 3D object points for a full chessboard grid
    (including outer edges), origin at top-left corner.

    board_cfg:
        squaresX: number of points along X (columns)  [default=6]
        squaresY: number of points along Y (rows)     [default=4]
        squareLength: size of each square in meters   [default=0.05]
    """
    NX = int(board_cfg.get('squaresX', 6))
    NY = int(board_cfg.get('squaresY', 4))
    square = float(board_cfg.get('squareLength', 0.05))

    # Full grid of points (including edges), origin at top-left
    obj_pts = np.stack(np.meshgrid(np.arange(NX), np.arange(NY)), -1).reshape(-1, 2)
    obj_pts = obj_pts[:, ::-1]  # swap to (x, y) ordering
    obj_pts = np.hstack([obj_pts * square, np.zeros((NX * NY, 1))]).astype(np.float64)

    return obj_pts


def calculateFOV(K, W, H, margin_px=0, safety_deg=2.0):
    """
    내부행렬 K와 이미지 크기에서 원뿔 반각(deg)을 보수적으로 산출.
    - margin_px: 가장자리 여백을 각도에서 제외
    - safety_deg: 살짝 보수 마진(수치불안정/왜곡 여유)
    수평/수직을 따로 계산해 더 작은(보수적) 반각을 반환.
    """
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    # 좌/우/상/하 픽셀 오프셋(주점 기준, margin 반영)
    dx_left   = max(cx - margin_px, 1e-6)
    dx_right  = max((W - 1 - margin_px) - cx, 1e-6)
    dy_top    = max(cy - margin_px, 1e-6)
    dy_bottom = max((H - 1 - margin_px) - cy, 1e-6)

    # 반각(rad): tan(theta) ~= offset/f
    theta_left   = np.arctan(dx_left  / fx)
    theta_right  = np.arctan(dx_right / fx)
    theta_top    = np.arctan(dy_top   / fy)
    theta_bottom = np.arctan(dy_bottom/ fy)

    # 수평/수직 반각(보수적으로 작은 쪽)
    half_h = min(theta_left, theta_right)
    half_v = min(theta_top,  theta_bottom)
    half   = min(half_h, half_v)

    deg = np.degrees(half) - safety_deg
    return max(5.0, float(deg))  # 너무 작아지지 않도록 하한

def trexp_rho_omega(rho, omega):
        """xi = [rho(3); omega(3)] → SE(3) using closed-form exp."""
        rho = np.asarray(rho, float).reshape(3)
        w   = np.asarray(omega, float).reshape(3)
        th  = np.linalg.norm(w)
        if th < 1e-12:
            R = np.eye(3)
            V = np.eye(3)
        else:
            k  = w / th
            Kx = np.array([[0, -k[2], k[1]],
                        [k[2], 0, -k[0]],
                        [-k[1], k[0], 0]])
            s, c = np.sin(th), np.cos(th)
            R = np.eye(3) + s*Kx + (1-c)*(Kx@Kx)
            V = (np.eye(3)
                + ((1-c)/th) * Kx
                + ((th - s)/(th)) * (Kx@Kx))
        t = V @ rho
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
        return T

def sample_se3_noise(theta_rad, l_mm):
    """
    회전: axis k ~ U(S^2), angle ~ U(0, theta_deg) [deg]
    병진: 각 성분 ~ U(-l_mm, +l_mm) [mm] → m
    반환: 4x4 T_noise (노이즈를 왼쪽 곱으로 사용: T_noisy = T_noise @ T_nom)
    """
    # axis-angle [rad]
    k = rng.normal(size=3)
    k /= (np.linalg.norm(k) + 1e-12)
    ang = rng.uniform(0.0, theta_rad)
    omega = ang * k
    # translation [m]
    rho = rng.uniform(-l_mm, l_mm, 3) * 1e-3
    return trexp_rho_omega(rho, omega)

def basis_from_z(z_axis):
    """
    z축으로부터 x축과 y축을 결정
    """
    z = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    helper = np.array([1., 0., 0.]) if abs(z[2]) > 0.9 else np.array([0., 0., 1.])
    x = np.cross(helper, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])