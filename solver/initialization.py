# solver/initialization.py
# -*- coding: utf-8 -*-
# 이중 쿼터니언(DQ) 기반 AX=YB 솔버

import numpy as np
from scipy.linalg import svd
import time

# --- Helper Functions (이전과 동일) ---
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]])

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def hom_to_dq(H):
    R, t = H[:3, :3], H[:3, 3]
    q_real = rot_to_quat(R)
    q_dual = 0.5 * quat_mult(np.array([0, t[0], t[1], t[2]]), q_real)
    return np.concatenate((q_real, q_dual))

def dq_to_hom(dq):
    norm_real = np.linalg.norm(dq[0:4])
    dq = dq / norm_real
    q_real, q_dual = dq[0:4], dq[4:8]
    R = quat_to_rot(q_real)
    t_quat = 2 * quat_mult(q_dual, quat_conj(q_real))
    t = t_quat[1:4]
    H = np.identity(4)
    H[:3, :3], H[:3, 3] = R, t
    return H

def q2mat_Lmult(q):
    w, x, y, z = q
    return np.array([[w,-x,-y,-z],[x,w,-z,y],[y,z,w,-x],[z,-y,x,w]])

def q2mat_Rmult(q):
    w, x, y, z = q
    return np.array([[w,-x,-y,-z],[x,w,z,-y],[y,-z,w,x],[z,y,-x,w]])

# --- Main Solver Function ---
def solve_axyb_dq(A_list, B_list):
    """
    Fu et al (2020)의 알고리즘을 구현합니다.
    행렬 방정식 AX=YB에서 X, Y를 풉니다.
    """
    # 입력이 행렬의 리스트이므로, (4, 4, n) 형태의 3D numpy 배열로 변환합니다.
    A = np.stack(A_list, axis=-1)
    B = np.stack(B_list, axis=-1)
    
    nbr = A.shape[2]
    H_list = []
    for i in range(nbr):
        A_hm, B_hm = A[:, :, i], B[:, :, i]
        A_dq, B_dq = hom_to_dq(A_hm), hom_to_dq(B_hm)
        
        q_ra, q_da = A_dq[:4], A_dq[4:]
        q_rb, q_db = B_dq[:4], B_dq[4:]
        
        row1 = np.hstack([q2mat_Lmult(q_ra), np.zeros((4,4)), -q2mat_Rmult(q_rb), np.zeros((4,4))])
        row2 = np.hstack([q2mat_Lmult(q_da), q2mat_Lmult(q_ra), -q2mat_Rmult(q_db), -q2mat_Rmult(q_rb)])
        
        D = np.vstack([row1, row2])
        H_list.append(D)
        
    H = np.vstack(H_list)
    _, _, Vt = svd(H)
    
    v15, v16 = Vt[-2, :], Vt[-1, :]
    
    u1_r, u1_d = v15[0:4], v15[4:8]
    w1_r, w1_d = v15[8:12], v15[12:16]
    u2_r, u2_d = v16[0:4], v16[4:8]
    w2_r, w2_d = v16[8:12], v16[12:16]
    
    a = u1_r @ u1_d + w1_r @ w1_d
    b = u1_r @ u2_d + u2_r @ u1_d + w1_r @ w2_d + w2_r @ w1_d
    c = u2_r @ u2_d + w2_r @ w2_d
    
    roots = []
    if abs(a) < 1e-9:
        if abs(b) > 1e-9:
            roots.append(-c / b)
    else:
        delta = b**2 - 4*a*c
        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            roots.append((-b + sqrt_delta) / (2*a))
            roots.append((-b - sqrt_delta) / (2*a))

    if not roots:
        raise RuntimeError("Solver failed to find a real solution.")
        
    best_s, max_norm_sq = 0, -1
    for s_val in roots:
        norm_sq = (s_val**2*(u1_r@u1_r) + 2*s_val*(u1_r@u2_r) + (u2_r@u2_r)) + \
                  (s_val**2*(w1_r@w1_r) + 2*s_val*(w1_r@w2_r) + (w2_r@w2_r))
        if norm_sq > max_norm_sq:
            max_norm_sq = norm_sq
            best_s = s_val
    s = best_s
    
    val = (s**2*(u1_r@u1_r) + 2*s*(u1_r@u2_r) + u2_r@u2_r) + \
          (s**2*(w1_r@w1_r) + 2*s*(w1_r@w2_r) + w2_r@w2_r)
          
    mu2 = np.sqrt(1 / val) if val > 0 else 1
    mu1 = s * mu2
    
    deta = mu1*v15 + mu2*v16
    X_dq, Y_dq = deta[0:8], deta[8:16]

    X, Y = dq_to_hom(X_dq), dq_to_hom(Y_dq)
    return X, Y


def _inv4(T):
    """Fast SE(3) inverse."""
    R = T[:3, :3]; t = T[:3, 3]
    Ri = R.T
    ti = -Ri @ t
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = Ri
    Ti[:3, 3]  = ti
    return Ti


def _ortho_svd(R):
    """Closest rotation (polar decomposition via SVD) with det=+1."""
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0:
        U[:, -1] *= -1
        Ro = U @ Vt
    return Ro


def _stack_to_n44(arr_or_list):
    """
    Normalize inputs to shape (n, 4, 4).
    Accepts: list of (4,4), or np.ndarray with shape (n,4,4) or (4,4,n).
    """
    A = np.asarray(arr_or_list)
    if A.ndim == 3 and A.shape == (4, 4, A.shape[2]):
        # (4,4,n) -> (n,4,4)
        return np.transpose(A, (2, 0, 1))
    if A.ndim == 3 and A.shape[1:] == (4, 4):
        # (n,4,4)
        return A
    if isinstance(arr_or_list, (list, tuple)) and A.ndim == 3 and A.shape[1:] == (4, 4):
        return A
    if isinstance(arr_or_list, (list, tuple)) and len(arr_or_list) > 0 and np.asarray(arr_or_list[0]).shape == (4,4):
        return np.stack(arr_or_list, axis=0)
    raise ValueError("Expected list of (4,4) or array of shape (n,4,4) or (4,4,n).")


def _vec_F(M):
    """Column-major (Fortran-order) vec(M) to match MATLAB's reshape behavior."""
    return np.reshape(M, (-1, 1), order="F")


def solve_init_two_step_abc(A, B, C):
    """
    Two-step 초기 해법 (dual-robot): A_i X B_i = Y C_i Z 를 만족하는 X,Y,Z 초기값 추정.

    Args:
        A, B, C : 각기 (n,4,4) 또는 (4,4,n) 또는 list of (4,4).
            - B_i: Cam1->Board
            - A_i, C_i: 두 로봇/체인의 측정 변환 (식의 좌/우측에 대응)
    Returns:
        X, Y, Z : 4x4 homogeneous transforms (초기 추정)

    참고:
      MATLAB 원형의 크로네커/reshape는 column-major이므로,
      numpy에서는 order='F'로 맞춰줍니다.
    """
    A = _stack_to_n44(A)
    B = _stack_to_n44(B)
    C = _stack_to_n44(C)

    n = A.shape[0]
    if n < 2:
        raise ValueError("Need at least two poses for a stable initialization.")

    # ----------------------------------------------------------------------
    # 1) 회전/이동 분리
    # ----------------------------------------------------------------------
    RA = np.zeros((n, 3, 3), dtype=float)
    RB = np.zeros((n, 3, 3), dtype=float)  # from T_cam1_to_cam2
    RC = np.zeros((n, 3, 3), dtype=float)
    tA = np.zeros((n, 3), dtype=float)
    tB = np.zeros((n, 3), dtype=float)
    tC = np.zeros((n, 3), dtype=float)

    for i in range(n):
        RA[i] = A[i, :3, :3]
        RB[i] = B[i, :3, :3]
        RC[i] = C[i, :3, :3]
        tA[i] = A[i, :3, 3]
        tB[i] = B[i, :3, 3]
        tC[i] = C[i, :3, 3]

    # ----------------------------------------------------------------------
    # 2) M_ZX = (RZ ⊗ RX) 추정 (eigendecomposition)
    #     Utilde = [ ... kron(mB_j^T, kron(I, RAjk)) - kron(mB_k^T, kron(RCjk^T, I)) ... ]
    # ----------------------------------------------------------------------
    rows = []
    I3 = np.eye(3)
    for j in range(n - 1):
        k = j + 1
        RAjk = RA[k].T @ RA[j]
        RCjk = RC[k].T @ RC[j]
        Ljk = np.kron(I3, RAjk)
        Njk = np.kron(RCjk.T, I3)

        mBj = _vec_F(RB[j]).T  # (1,9)
        mBk = _vec_F(RB[k]).T  # (1,9)

        # Ui = kron(mBj, Ljk) - kron(mBk, Njk)
        # kron with (1,9) and (9,9) -> (9,81); we then stack rows
        Ui = np.kron(mBj, Ljk) - np.kron(mBk, Njk)  # shape (1*9, 9*9) = (9,81)
        rows.append(Ui)

    Utilde = np.vstack(rows)                       # (9*(n-1), 81)
    S = Utilde.T @ Utilde                          # (81,81)
    # Right singular vector of smallest singular value
    _, _, Vt = np.linalg.svd(S)
    mZX_vec = 3.0 * Vt[-1, :]                      # (81,)
    MZX = np.reshape(mZX_vec, (9, 9), order="F")   # (RZ ⊗ RX)

    # ----------------------------------------------------------------------
    # 3) RY 선형추정:  (RC_i^T ⊗ RA_i^T) vec(RY) = MZX vec(RB_i)
    # ----------------------------------------------------------------------
    MCA = []
    mZXB = []
    for i in range(n):
        MCA.append(np.kron(RC[i].T, RA[i].T))      # (9,9)
        mB = _vec_F(RB[i])                         # (9,1)
        mZXB.append(MZX @ mB)                      # (9,1)

    MCA = np.vstack(MCA)           # (9n, 9)
    mZXB = np.vstack(mZXB)         # (9n, 1)
    # Least squares
    mY, *_ = np.linalg.lstsq(MCA, mZXB, rcond=None)
    RY = np.reshape(mY, (3, 3), order="F")
    if np.linalg.det(RY) < 0:
        RY = -RY
    RY = _ortho_svd(RY)

    # ----------------------------------------------------------------------
    # 4) RX, RZ, tX, tY, tZ 해 구하기 (선형 시스템)
    #     Unknown vector: [vec(RX); vec(RZ); tX; tY; tZ]  (27x1)
    # ----------------------------------------------------------------------
    A_full_rows = []
    b_full_rows = []
    for i in range(n):
        RAi = RA[i]; RBi = RB[i]; RCi = RC[i]
        tAi = tA[i]; tBi = tB[i]; tCi = tC[i]

        # row block 1 (9 rows): rotation coupling
        # [ kron(RBi^T, I),  -kron(I, RAi^T RY RCi),  0, 0, 0 ] * [vec(RX); vec(RZ); tX; tY; tZ] = 0
        row1 = np.hstack([
            np.kron(RBi.T, np.eye(3)),
            -np.kron(np.eye(3), RAi.T @ RY @ RCi),
            np.zeros((9, 3)),
            np.zeros((9, 3)),
            np.zeros((9, 3)),
        ])

        # row block 2 (3 rows): translation coupling
        # [ kron(tBi^T, RY^T RAi),  0,  RY^T RAi,  -RY^T,  -RCi ] x = -RY^T tAi + tCi
        row2 = np.hstack([
            np.kron(tBi.reshape(1, 3), (RY.T @ RAi)),
            np.zeros((3, 9)),
            (RY.T @ RAi),
            -RY.T,
            -RCi
        ])

        A_full_rows.append(row1)
        A_full_rows.append(row2)

        b_full_rows.append(np.zeros((9, 1)))
        b_full_rows.append((-RY.T @ tAi.reshape(3, 1) + tCi.reshape(3, 1)))

    A_full = np.vstack(A_full_rows)     # ((12n) x 27)
    b_full = np.vstack(b_full_rows)     # ((12n) x 1)

    x_full, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
    x_full = x_full.reshape(-1)

    mX = x_full[0:9]
    mZ = x_full[9:18]
    tX = x_full[18:21]
    tY = x_full[21:24]
    tZ = x_full[24:27]

    RX = np.reshape(mX, (3, 3), order="F")
    RZ = np.reshape(mZ, (3, 3), order="F")

    RX = _ortho_svd(RX)
    RZ = _ortho_svd(RZ)

    # ----------------------------------------------------------------------
    # Assemble homogeneous transforms
    # ----------------------------------------------------------------------
    X = np.eye(4); X[:3, :3] = RX; X[:3, 3] = tX
    Y = np.eye(4); Y[:3, :3] = RY; Y[:3, 3] = tY
    Z = np.eye(4); Z[:3, :3] = RZ; Z[:3, 3] = tZ
    return X, Y, Z


def solve_init_two_step_abcd(A, B, C, D):
    """
    Two-step 초기 해법 (dual-robot): A_i X B_i = Y C_i Z 를 만족하는 X,Y,Z 초기값 추정.

    Args:
        A, B, C, D : 각기 (n,4,4) 또는 (4,4,n) 또는 list of (4,4).
            - B_i: Cam1->Board
            - D_i: Cam2->Board
            - A_i, C_i: 두 로봇/체인의 측정 변환 (식의 좌/우측에 대응)
    Returns:
        X, Y, Z : 4x4 homogeneous transforms (초기 추정)

    참고:
      MATLAB 원형의 크로네커/reshape는 column-major이므로,
      numpy에서는 order='F'로 맞춰줍니다.
    """
    A = _stack_to_n44(A)
    B = _stack_to_n44(B)
    C = _stack_to_n44(C)
    D = _stack_to_n44(D)

    n = A.shape[0]
    if n < 2:
        raise ValueError("Need at least two poses for a stable initialization.")

    # ----------------------------------------------------------------------
    # 0) Cam1 -> Cam2 만들기:  T_cam1_to_cam2 = B * inv(D)
    # ----------------------------------------------------------------------
    T12 = np.zeros((n, 4, 4), dtype=float)
    for i in range(n):
        Bi = B[i]
        Di_inv = _inv4(D[i])
        T12[i] = Bi @ Di_inv

    # ----------------------------------------------------------------------
    # 1) 회전/이동 분리
    # ----------------------------------------------------------------------
    RA = np.zeros((n, 3, 3), dtype=float)
    RB = np.zeros((n, 3, 3), dtype=float)  # from T_cam1_to_cam2
    RC = np.zeros((n, 3, 3), dtype=float)
    tA = np.zeros((n, 3), dtype=float)
    tB = np.zeros((n, 3), dtype=float)
    tC = np.zeros((n, 3), dtype=float)

    for i in range(n):
        RA[i] = A[i, :3, :3]
        RB[i] = T12[i, :3, :3]
        RC[i] = C[i, :3, :3]
        tA[i] = A[i, :3, 3]
        tB[i] = T12[i, :3, 3]
        tC[i] = C[i, :3, 3]

    # ----------------------------------------------------------------------
    # 2) M_ZX = (RZ ⊗ RX) 추정 (eigendecomposition)
    #     Utilde = [ ... kron(mB_j^T, kron(I, RAjk)) - kron(mB_k^T, kron(RCjk^T, I)) ... ]
    # ----------------------------------------------------------------------
    rows = []
    I3 = np.eye(3)
    for j in range(n - 1):
        k = j + 1
        RAjk = RA[k].T @ RA[j]
        RCjk = RC[k].T @ RC[j]
        Ljk = np.kron(I3, RAjk)
        Njk = np.kron(RCjk.T, I3)

        mBj = _vec_F(RB[j]).T  # (1,9)
        mBk = _vec_F(RB[k]).T  # (1,9)

        # Ui = kron(mBj, Ljk) - kron(mBk, Njk)
        # kron with (1,9) and (9,9) -> (9,81); we then stack rows
        Ui = np.kron(mBj, Ljk) - np.kron(mBk, Njk)  # shape (1*9, 9*9) = (9,81)
        rows.append(Ui)

    Utilde = np.vstack(rows)                       # (9*(n-1), 81)
    S = Utilde.T @ Utilde                          # (81,81)
    # Right singular vector of smallest singular value
    _, _, Vt = np.linalg.svd(S)
    mZX_vec = 3.0 * Vt[-1, :]                      # (81,)
    MZX = np.reshape(mZX_vec, (9, 9), order="F")   # (RZ ⊗ RX)

    # ----------------------------------------------------------------------
    # 3) RY 선형추정:  (RC_i^T ⊗ RA_i^T) vec(RY) = MZX vec(RB_i)
    # ----------------------------------------------------------------------
    MCA = []
    mZXB = []
    for i in range(n):
        MCA.append(np.kron(RC[i].T, RA[i].T))      # (9,9)
        mB = _vec_F(RB[i])                         # (9,1)
        mZXB.append(MZX @ mB)                      # (9,1)

    MCA = np.vstack(MCA)           # (9n, 9)
    mZXB = np.vstack(mZXB)         # (9n, 1)
    # Least squares
    mY, *_ = np.linalg.lstsq(MCA, mZXB, rcond=None)
    RY = np.reshape(mY, (3, 3), order="F")
    if np.linalg.det(RY) < 0:
        RY = -RY
    RY = _ortho_svd(RY)

    # ----------------------------------------------------------------------
    # 4) RX, RZ, tX, tY, tZ 해 구하기 (선형 시스템)
    #     Unknown vector: [vec(RX); vec(RZ); tX; tY; tZ]  (27x1)
    # ----------------------------------------------------------------------
    A_full_rows = []
    b_full_rows = []
    for i in range(n):
        RAi = RA[i]; RBi = RB[i]; RCi = RC[i]
        tAi = tA[i]; tBi = tB[i]; tCi = tC[i]

        # row block 1 (9 rows): rotation coupling
        # [ kron(RBi^T, I),  -kron(I, RAi^T RY RCi),  0, 0, 0 ] * [vec(RX); vec(RZ); tX; tY; tZ] = 0
        row1 = np.hstack([
            np.kron(RBi.T, np.eye(3)),
            -np.kron(np.eye(3), RAi.T @ RY @ RCi),
            np.zeros((9, 3)),
            np.zeros((9, 3)),
            np.zeros((9, 3)),
        ])

        # row block 2 (3 rows): translation coupling
        # [ kron(tBi^T, RY^T RAi),  0,  RY^T RAi,  -RY^T,  -RCi ] x = -RY^T tAi + tCi
        row2 = np.hstack([
            np.kron(tBi.reshape(1, 3), (RY.T @ RAi)),
            np.zeros((3, 9)),
            (RY.T @ RAi),
            -RY.T,
            -RCi
        ])

        A_full_rows.append(row1)
        A_full_rows.append(row2)

        b_full_rows.append(np.zeros((9, 1)))
        b_full_rows.append((-RY.T @ tAi.reshape(3, 1) + tCi.reshape(3, 1)))

    A_full = np.vstack(A_full_rows)     # ((12n) x 27)
    b_full = np.vstack(b_full_rows)     # ((12n) x 1)

    x_full, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
    x_full = x_full.reshape(-1)

    mX = x_full[0:9]
    mZ = x_full[9:18]
    tX = x_full[18:21]
    tY = x_full[21:24]
    tZ = x_full[24:27]

    RX = np.reshape(mX, (3, 3), order="F")
    RZ = np.reshape(mZ, (3, 3), order="F")

    RX = _ortho_svd(RX)
    RZ = _ortho_svd(RZ)

    # ----------------------------------------------------------------------
    # Assemble homogeneous transforms
    # ----------------------------------------------------------------------
    X = np.eye(4); X[:3, :3] = RX; X[:3, 3] = tX
    Y = np.eye(4); Y[:3, :3] = RY; Y[:3, 3] = tY
    Z = np.eye(4); Z[:3, :3] = RZ; Z[:3, 3] = tZ
    return X, Y, Z