from __future__ import annotations

import numpy as np

# ---------- 기본 유틸 ----------
def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=float)

# Quaternion은 [x, y, z, w] (w=scalar) 형태로 사용합니다.

def quat_from_R(R: np.ndarray) -> np.ndarray:
    # robust 변환 (w가 마지막)
    tr = np.trace(R)
    q = np.zeros(4, dtype=float)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2,1] - R[1,2]) / s
        q[1] = (R[0,2] - R[2,0]) / s
        q[2] = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q[0] = 0.25 * s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = (R[0,2] + R[2,0]) / s
            q[3] = (R[2,1] - R[1,2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q[0] = (R[0,1] + R[1,0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = (R[0,2] - R[2,0]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q[0] = (R[0,2] + R[2,0]) / s
            q[1] = (R[1,2] + R[2,1]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,0] - R[0,1]) / s
    # 정규화 및 부호 정규화(w>=0)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q

def R_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)
    return R

def quat_conj(q: np.ndarray) -> np.ndarray:
    v, w = q[:3], q[3]
    return np.hstack([-v, w])

def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pv, ps = p[:3], p[3]
    qv, qs = q[:3], q[3]
    v = ps*qv + qs*pv + np.cross(pv, qv)
    s = ps*qs - np.dot(pv, qv)
    return np.hstack([v, s])

def quat_left(q: np.ndarray) -> np.ndarray:
    v, s = q[:3], q[3]
    L = np.zeros((4,4), dtype=float)
    L[:3,:3] = s*np.eye(3) + skew(v)
    L[:3,3]  = v
    L[3,:3]  = -v
    L[3,3]   = s
    return L

def quat_right(q: np.ndarray) -> np.ndarray:
    v, s = q[:3], q[3]
    R = np.zeros((4,4), dtype=float)
    R[:3,:3] = s*np.eye(3) - skew(v)
    R[:3,3]  = v
    R[3,:3]  = -v
    R[3,3]   = s
    return R

# ---------- Dual Quaternion ----------
# dq = qr + ε qd  (unit dual quaternion)
# qd = 0.5 * t_quat ⊗ qr,  where t_quat=[t,0]
def dq_from_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = T[:3,:3]; t = T[:3,3]
    qr = quat_from_R(R)
    t_quat = np.hstack([t, 0.0])
    qd = 0.5 * quat_mul(t_quat, qr)
    return qr, qd

def T_from_dq(qr: np.ndarray, qd: np.ndarray) -> np.ndarray:
    R = R_from_quat(qr)
    qr_conj = quat_conj(qr)
    tq = quat_mul( qd*2.0, qr_conj )  # = [t, 0]
    t = tq[:3]
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3,3]  = t
    return T
