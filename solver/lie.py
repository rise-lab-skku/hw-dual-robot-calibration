import numpy as np
from scipy.optimize import least_squares
from utils.se3 import (se3_exp, inv_T, project, _vecF, _matF, _ortho_svd,
                   vee, trlog, trexp, Ad, ad, left_jacobian_se3)

from scipy.spatial.transform import Rotation

def get_XYZ_extrinsic_to_se3_jacobian(roll, pitch):
    """
    'XYZ' extrinsic Euler angles -> se(3) 벡터 매핑에 대한 6x6 야코비안 계산
    """
    J = np.zeros((6, 6))
    J[:3, :3] = np.eye(3) # Translation 부분

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)

    if abs(abs(pitch) - np.pi / 2.0) < 1e-6: # 특이점 (Gimbal Lock)
        J[3:, 3:] = np.eye(3)
        return J

    J_rot = np.zeros((3, 3))
    J_rot[0, 0] = 1.0; J_rot[0, 1] = 0.0; J_rot[0, 2] = -sp
    J_rot[1, 0] = 0.0; J_rot[1, 1] = cr;  J_rot[1, 2] = sr * cp
    J_rot[2, 0] = 0.0; J_rot[2, 1] = -sr; J_rot[2, 2] = cr * cp

    J[3:, 3:] = J_rot
    return J

class LieOptimizationSolverAXBYCZ:
    """
    Dual-robot calibration via Lie-derivative Newton method.
    """
    def __init__(self, A, B, C, X0, Y0, Z0, **kwargs):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.X0, self.Y0, self.Z0 = X0, Y0, Z0
        self.n = self.A.shape[0]
        self.ls_kwargs = kwargs
        self.xiA = np.zeros((6, self.n))
        self.xiB = np.zeros((6, self.n))
        self.xiC = np.zeros((6, self.n))
        self.s = 1.0

    def _compute_residuals_and_jacobian(self, params):
        xiX, xiY, xiZ = params[0:6], params[6:12], params[12:18]
        e = np.zeros(6 * self.n)
        J = np.zeros((6 * self.n, 18))

        G_X = trexp(xiX)
        G_Y_inv = trexp(-xiY)
        G_Z_inv = trexp(-xiZ)

        JlX = left_jacobian_se3(xiX)
        JrY = left_jacobian_se3(-xiY)
        JrZ = left_jacobian_se3(-xiZ)

        for i in range(self.n):
            G_A = trexp(self.xiA[:, i])
            G_B = trexp(self.xiB[:, i])
            G_C_inv = trexp(-self.xiC[:, i])

            G_err = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv @ G_Y_inv
            e[6*i : 6*(i+1)] = vee(trlog(G_err))

            g1 = G_A
            g2 = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv
            g3 = G_A @ G_X @ G_B
            
            JX = Ad(g1) @ JlX
            JY = Ad(g2) @ JrY
            JZ = Ad(g3) @ JrZ
            
            J[6*i : 6*(i+1), :] = np.hstack([JX, -JY, -JZ])

        return e, J

    def solve(self):
        for i in range(self.n):
            self.xiA[:, i] = vee(trlog(self.A[i]))
            self.xiC[:, i] = vee(trlog(self.C[i]))
            self.xiB[:, i] = vee(trlog(self.B[i]))

        xiX = vee(trlog(self.X0))
        xiY = vee(trlog(self.Y0))
        xiZ = vee(trlog(self.Z0))

        w_phi = (np.linalg.norm(xiX[3:]) + np.linalg.norm(xiY[3:]) + np.linalg.norm(xiZ[3:])) / 3.0
        w_rho = (np.linalg.norm(xiX[:3]) + np.linalg.norm(xiY[:3]) + np.linalg.norm(xiZ[:3])) / 3.0
        self.s = w_phi / w_rho if w_rho > 1e-9 else 1.0

        self.xiA[:3, :] *= self.s
        self.xiB[:3, :] *= self.s
        self.xiC[:3, :] *= self.s
        xiX[:3] *= self.s
        xiY[:3] *= self.s
        xiZ[:3] *= self.s

        p0 = np.hstack([xiX, xiY, xiZ])

        # print("\n[LieOptimizationSolver] Starting optimization with verbose output...")
        # print("="*80)

        res = least_squares(
            lambda p: self._compute_residuals_and_jacobian(p)[0],
            p0,
            jac=lambda p: self._compute_residuals_and_jacobian(p)[1],
            method='lm',
            verbose=0,
            **self.ls_kwargs
        )

        # print("="*80)
        # print("[LieOptimizationSolver] Optimization finished.")

        xiX_opt, xiY_opt, xiZ_opt = res.x[0:6], res.x[6:12], res.x[12:18]
        xiX_opt[:3] /= self.s
        xiY_opt[:3] /= self.s
        xiZ_opt[:3] /= self.s

        X = trexp(xiX_opt)
        Y = trexp(xiY_opt)
        Z = trexp(xiZ_opt)

        return {'X': X, 'Y': Y, 'Z': Z, 'result': res}

# class LieOptimizationSolverAXBYCZD:
#     """
#     Dual-robot calibration via Lie-derivative Newton method.
#     """
#     def __init__(self, A, B, C, D, X0, Y0, Z0, **kwargs):
#         self.A = np.asarray(A, dtype=float)
#         self.B = np.asarray(B, dtype=float)
#         self.C = np.asarray(C, dtype=float)
#         self.D = np.asarray(D, dtype=float)
#         self.X0, self.Y0, self.Z0 = X0, Y0, Z0
#         self.n = self.A.shape[0]
#         self.ls_kwargs = kwargs
#         self.xiA = np.zeros((6, self.n))
#         self.xiB = np.zeros((6, self.n))
#         self.xiC = np.zeros((6, self.n))
#         self.s = 1.0

#     def _compute_residuals_and_jacobian(self, params):
#         xiX, xiY, xiZ = params[0:6], params[6:12], params[12:18]
#         e = np.zeros(6 * self.n)
#         J = np.zeros((6 * self.n, 18))

#         G_X = trexp(xiX)
#         G_Y_inv = trexp(-xiY)
#         G_Z_inv = trexp(-xiZ)

#         JlX = left_jacobian_se3(xiX)
#         JrY = left_jacobian_se3(-xiY)
#         JrZ = left_jacobian_se3(-xiZ)

#         for i in range(self.n):
#             G_A = trexp(self.xiA[:, i])
#             G_B = trexp(self.xiB[:, i])
#             G_C_inv = trexp(-self.xiC[:, i])

#             G_err = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv @ G_Y_inv
#             e[6*i : 6*(i+1)] = vee(trlog(G_err))

#             g1 = G_A
#             g2 = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv
#             g3 = G_A @ G_X @ G_B
            
#             JX = Ad(g1) @ JlX
#             JY = Ad(g2) @ JrY
#             JZ = Ad(g3) @ JrZ
            
#             J[6*i : 6*(i+1), :] = np.hstack([JX, -JY, -JZ])

#         return e, J

#     def solve(self):
#         B_mod = np.zeros_like(self.B)
#         for i in range(self.n):
#             self.xiA[:, i] = vee(trlog(self.A[i]))
#             self.xiC[:, i] = vee(trlog(self.C[i]))
#             B_mod[i] = self.B[i] @ inv_T(self.D[i])
#             self.xiB[:, i] = vee(trlog(B_mod[i]))

#         xiX = vee(trlog(self.X0))
#         xiY = vee(trlog(self.Y0))
#         xiZ = vee(trlog(self.Z0))

#         w_phi = (np.linalg.norm(xiX[3:]) + np.linalg.norm(xiY[3:]) + np.linalg.norm(xiZ[3:])) / 3.0
#         w_rho = (np.linalg.norm(xiX[:3]) + np.linalg.norm(xiY[:3]) + np.linalg.norm(xiZ[:3])) / 3.0
#         self.s = w_phi / w_rho if w_rho > 1e-9 else 1.0

#         self.xiA[:3, :] *= self.s
#         self.xiB[:3, :] *= self.s
#         self.xiC[:3, :] *= self.s
#         xiX[:3] *= self.s
#         xiY[:3] *= self.s
#         xiZ[:3] *= self.s

#         p0 = np.hstack([xiX, xiY, xiZ])

#         # print("\n[LieOptimizationSolver] Starting optimization with verbose output...")
#         # print("="*80)

#         res = least_squares(
#             lambda p: self._compute_residuals_and_jacobian(p)[0],
#             p0,
#             jac=lambda p: self._compute_residuals_and_jacobian(p)[1],
#             method='lm',
#             verbose=0,
#             **self.ls_kwargs
#         )

#         # print("="*80)
#         # print("[LieOptimizationSolver] Optimization finished.")

#         xiX_opt, xiY_opt, xiZ_opt = res.x[0:6], res.x[6:12], res.x[12:18]
#         xiX_opt[:3] /= self.s
#         xiY_opt[:3] /= self.s
#         xiZ_opt[:3] /= self.s

#         X = trexp(xiX_opt)
#         Y = trexp(xiY_opt)
#         Z = trexp(xiZ_opt)

#         return {'X': X, 'Y': Y, 'Z': Z, 'result': res}
    

import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import least_squares

# -- 아래 유틸은 기존 코드에 이미 있다고 가정합니다.
# trexp, trlog, vee, Ad, inv_T, left_jacobian_se3

class LieOptimizationSolverAXBYCZD:
    """
    Dual-robot calibration via Lie-derivative Newton/LM with optional covariance weighting.
    Residual: e_k = log( A_k X B_k Z^{-1} C_k^{-1} Y^{-1} ) in se(3).
    """

    def __init__(self, A, B, C, D, X0, Y0, Z0,
                 use_cov_weight=False,
                 var_t_A=None, var_a_A=None,
                 var_t_C=None, var_a_C=None,
                 eps_reg=1e-9,
                 **kwargs):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.X0, self.Y0, self.Z0 = X0, Y0, Z0
        self.n = self.A.shape[0]
        self.ls_kwargs = kwargs

        # se(3) log coords: [t(3), r(3)]
        self.xiA = np.zeros((6, self.n))
        self.xiB = np.zeros((6, self.n))
        self.xiC = np.zeros((6, self.n))

        # scale factor for translation components (your original code)
        self.s = 1.0

        # --- Covariance weighting options ---
        self.use_cov_weight = bool(use_cov_weight)
        self.eps_reg = float(eps_reg)

        # per-pose base covariances in *unscaled* se(3) (diag)
        # If None, cov-weighting is disabled even if use_cov_weight=True
        self.var_t_A = var_t_A  # scalar or (n,) or (n,3) optional
        self.var_a_A = var_a_A
        self.var_t_C = var_t_C
        self.var_a_C = var_a_C

        # Prepared per-pose SigmaA/SigmaC in *scaled* se(3) will be built in solve()
        self.SigmaA_scaled = None
        self.SigmaC_scaled = None

    # ---------- numerical 6x6 Jacobian wrt self.xiA[:,i] / self.xiC[:,i] for weighting ----------
    def _numerical_J_wrt_pose(self, which: str, i: int, xiX, xiY, xiZ):
        """
        which: 'A' or 'C'; i: pose index
        returns 6x6 matrix J_{which,i} = ∂e_i/∂xi_{which,i}  (in the *current scaled* se(3) coords)
        """
        h = 1e-6
        J = np.zeros((6, 6))
        base_ei = self._residual_block_i(i, xiX, xiY, xiZ)

        # perturb 6 basis directions
        for j in range(6):
            if which == 'A':
                self.xiA[j, i] += h
                e_pert = self._residual_block_i(i, xiX, xiY, xiZ)
                self.xiA[j, i] -= h
            elif which == 'C':
                self.xiC[j, i] += h
                e_pert = self._residual_block_i(i, xiX, xiY, xiZ)
                self.xiC[j, i] -= h
            else:
                raise ValueError("which must be 'A' or 'C'")
            J[:, j] = (e_pert - base_ei) / h
        return J

    def _residual_block_i(self, i, xiX, xiY, xiZ):
        """return 6-d residual for pose i (in scaled se(3) coords, consistent with self.s)"""
        G_A = trexp(self.xiA[:, i])
        G_B = trexp(self.xiB[:, i])
        G_C_inv = trexp(-self.xiC[:, i])
        G_X = trexp(xiX)
        G_Y_inv = trexp(-xiY)
        G_Z_inv = trexp(-xiZ)
        G_err = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv @ G_Y_inv
        return vee(trlog(G_err))
    
    def _analytic_JA_JC(self, i, xiX, xiY, xiZ):
        """
        Computes analytic Jacobians of e_i w.r.t. robot poses A_i and C_i.
        This version correctly applies the Adjoint transformation, consistent
        with the Jacobian calculation for X, Y, and Z.
        """
        def right_jacobian_se3(xi):
            return left_jacobian_se3(-xi)

        # --- 1. 스케일링 역변환 행렬 ---
        s_inv = 1.0 / self.s if self.s > 1e-9 else 1.0
        S_inv = np.diag([s_inv, s_inv, s_inv, 1.0, 1.0, 1.0])

        # 현재 스케일링된 파라미터
        xiA_scaled = self.xiA[:, i]
        xiC_scaled = self.xiC[:, i]

        # --- 2. 그룹 원소 및 리지듀얼 계산 ---
        G_A    = trexp(xiA_scaled)
        G_B    = trexp(self.xiB[:, i])
        G_Cinv = trexp(-xiC_scaled)
        G_X    = trexp(xiX)
        G_Yinv = trexp(-xiY)
        G_Zinv = trexp(-xiZ)

        G_err = G_A @ G_X @ G_B @ G_Zinv @ G_Cinv @ G_Yinv
        e_i   = vee(trlog(G_err))
        Jl_e_inv = np.linalg.inv(left_jacobian_se3(e_i))

        # --- 3. 야코비안 계산을 위한 unscaling ---
        xiA_unscaled = S_inv @ xiA_scaled
        xiC_unscaled = S_inv @ xiC_scaled

        # --- 4. Adjoint를 포함한 올바른 야코비안 공식 ---
        # 일반 공식: ∂e/∂ξ_k = Jl(e)⁻¹ @ Ad(prefix_k) @ Jl(ξ_k)

        # (4-1) JA 계산: A는 맨 앞에 있으므로 prefix는 Identity.
        # A = exp(ξ_A)
        Jl_xiA = left_jacobian_se3(xiA_unscaled)
        JA_unscaled_param = Jl_xiA  # Ad(I)는 Identity이므로 생략

        # (4-2) JC 계산: C⁻¹의 prefix는 (A @ X @ B @ Z⁻¹) 입니다.
        # C⁻¹ = exp(-ξ_C) 이므로, ξ_C에 대한 미분은 -Jr(ξ_C) 항을 만듭니다.
        prefix_C = G_A @ G_X @ G_B @ G_Zinv
        Ad_prefix_C = Ad(prefix_C)
        
        # C⁻¹=exp(-ξ)를 ξ로 미분하면 -Jr(ξ) = -Jl(-ξ) 항이 나옵니다.
        Jr_xiC = right_jacobian_se3(xiC_unscaled) # Jr(ξ) = Jl(-ξ)
        JC_unscaled_param = Ad_prefix_C @ (-Jr_xiC)

        # --- 5. 연쇄 법칙(Chain Rule) 적용 ---
        JA_final = JA_unscaled_param @ S_inv
        JC_final = JC_unscaled_param @ S_inv
        
        return JA_final, JC_final, e_i

    def debug_compare_JA_JC(self, xiX, xiY, xiZ, idxs=None, h=1e-7, use_central=True):
        """
        수치야코비안(∂e/∂xiA_i, ∂e/∂xiC_i) vs 해석야코비안(JA, JC) 상대오차 비교.
        - e, J는 모두 'scaled se(3)' 좌표계를 사용 (self.s 반영)
        - whitening(가중치) 적용 전의 'raw residual' 기준으로 비교합니다.
        """
        import numpy.linalg as npl

        if idxs is None:
            # 포즈 개수가 많으면 몇 개만 샘플
            n = min(8, self.n)
            idxs = np.linspace(0, self.n-1, n, dtype=int)

        max_relA = 0.0
        max_relC = 0.0

        for i in idxs:
            # --- 해석 야코비안 ---
            JA_an, JC_an, e_i = self._analytic_JA_JC(i, xiX, xiY, xiZ)

            # --- 수치 야코비안 (중앙차분 권장) ---
            JA_num = np.zeros((6,6)); JC_num = np.zeros((6,6))
            base = self._residual_block_i(i, xiX, xiY, xiZ)

            for j in range(6):
                if use_central:
                    # A
                    self.xiA[j, i] += h
                    e_plus = self._residual_block_i(i, xiX, xiY, xiZ)
                    self.xiA[j, i] -= 2*h
                    e_minus = self._residual_block_i(i, xiX, xiY, xiZ)
                    self.xiA[j, i] += h
                    JA_num[:, j] = (e_plus - e_minus) / (2*h)

                    # C
                    self.xiC[j, i] += h
                    e_plus = self._residual_block_i(i, xiX, xiY, xiZ)
                    self.xiC[j, i] -= 2*h
                    e_minus = self._residual_block_i(i, xiX, xiY, xiZ)
                    self.xiC[j, i] += h
                    JC_num[:, j] = (e_plus - e_minus) / (2*h)
                else:
                    # 전진차분(정확도↓): 참고용
                    self.xiA[j, i] += h
                    e_pert = self._residual_block_i(i, xiX, xiY, xiZ)
                    self.xiA[j, i] -= h
                    JA_num[:, j] = (e_pert - base) / h

                    self.xiC[j, i] += h
                    e_pert = self._residual_block_i(i, xiX, xiY, xiZ)
                    self.xiC[j, i] -= h
                    JC_num[:, j] = (e_pert - base) / h

            # --- 상대오차 ---
            relA = npl.norm(JA_an - JA_num) / max(1e-12, npl.norm(JA_num))
            relC = npl.norm(JC_an - JC_num) / max(1e-12, npl.norm(JC_num))
            max_relA = max(max_relA, relA)
            max_relC = max(max_relC, relC)

            print(f"[i={i}] ‖JA_an-JA_num‖/‖JA_num‖={relA:.2e}, "
                f"‖JC_an-JC_num‖/‖JC_num‖={relC:.2e}")

        print(f"=> max relative error: JA {max_relA:.2e}, JC {max_relC:.2e}")

    # ---------- residual & jacobian (optionally pre-whitened) ----------
    def _compute_residuals_and_jacobian_weighted(self, params):
        xiX, xiY, xiZ = params[0:6], params[6:12], params[12:18]

        # unweighted (raw) residuals/Jacobians wrt [xiX, xiY, xiZ]
        e = np.zeros(6 * self.n)
        J = np.zeros((6 * self.n, 18))

        G_X = trexp(xiX)
        G_Y_inv = trexp(-xiY)
        G_Z_inv = trexp(-xiZ)

        JlX = left_jacobian_se3(xiX)         # for left perturb X
        JrY = left_jacobian_se3(-xiY)        # for right perturb Y^{-1}
        JrZ = left_jacobian_se3(-xiZ)        # for right perturb Z^{-1}

        # storage for whitening
        if self.use_cov_weight and (self.SigmaA_scaled is not None) and (self.SigmaC_scaled is not None):
            e_w = np.zeros_like(e)
            J_w = np.zeros_like(J)
        else:
            # cov-weighting off: return raw e,J
            for i in range(self.n):
                G_A = trexp(self.xiA[:, i])
                G_B = trexp(self.xiB[:, i])
                G_C_inv = trexp(-self.xiC[:, i])

                G_err = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv @ G_Y_inv
                e_i = vee(trlog(G_err))
                e[6*i:6*(i+1)] = e_i

                g1 = G_A
                g2 = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv
                g3 = G_A @ G_X @ G_B

                JX = Ad(g1) @ JlX
                JY = Ad(g2) @ JrY
                JZ = Ad(g3) @ JrZ

                J[6*i:6*(i+1), :] = np.hstack([JX, -JY, -JZ])
            return e, J

        # --- cov-weighted: compute per-block whitening ---
        for i in range(self.n):
            # raw residual & variable Jacobians
            G_A = trexp(self.xiA[:, i])
            G_B = trexp(self.xiB[:, i])
            G_C_inv = trexp(-self.xiC[:, i])

            G_err = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv @ G_Y_inv
            e_i = vee(trlog(G_err))

            g1 = G_A
            g2 = G_A @ G_X @ G_B @ G_Z_inv @ G_C_inv
            g3 = G_A @ G_X @ G_B

            JX = Ad(g1) @ JlX
            JY = Ad(g2) @ JrY
            JZ = Ad(g3) @ JrZ

            # numerical Jacobians wrt A_i and C_i (6x6) for Sigma propagation
            JAi = self._numerical_J_wrt_pose('A', i, xiX, xiY, xiZ)
            JCi = self._numerical_J_wrt_pose('C', i, xiX, xiY, xiZ)

            # JAi, JCi, e_i = self._analytic_JA_JC(i, xiX, xiY, xiZ)

            # chain covariance in *current scaled* se(3):
            Sigma_chain = (JAi @ self.SigmaA_scaled[i] @ JAi.T) + \
                          (JCi @ self.SigmaC_scaled[i] @ JCi.T) + \
                          (self.eps_reg * np.eye(6))

            # weight: W = Sigma^{-1}; whiten with L such that L^T L = W
            # => cholesky of W (upper) so that ||e||_W^2 = ||L e||^2 with L = chol(W)
            W = np.linalg.inv(Sigma_chain)
            L = cholesky(W, lower=False)  # upper triangular

            # prewhiten residual/Jacobian rows
            e_w[6*i:6*(i+1)] = L @ e_i
            J_block = np.hstack([JX, -JY, -JZ])
            J_w[6*i:6*(i+1), :] = L @ J_block

        return e_w, J_w

    def solve(self):
        # log-coordinates for A,B_mod,C as in your code
        B_mod = np.zeros_like(self.B)
        for i in range(self.n):
            self.xiA[:, i] = vee(trlog(self.A[i]))
            self.xiC[:, i] = vee(trlog(self.C[i]))
            B_mod[i] = self.B[i] @ inv_T(self.D[i])
            self.xiB[:, i] = vee(trlog(B_mod[i]))

        xiX = vee(trlog(self.X0))
        xiY = vee(trlog(self.Y0))
        xiZ = vee(trlog(self.Z0))

        # scale factor between t/r (your original heuristic)
        w_phi = (np.linalg.norm(xiX[3:]) + np.linalg.norm(xiY[3:]) + np.linalg.norm(xiZ[3:])) / 3.0
        w_rho = (np.linalg.norm(xiX[:3]) + np.linalg.norm(xiY[:3]) + np.linalg.norm(xiZ[:3])) / 3.0
        self.s = w_phi / w_rho if w_rho > 1e-9 else 1.0

        # apply scaling in-place (consistent across all se(3) coords)
        S = np.diag([self.s, self.s, self.s, 1.0, 1.0, 1.0])
        self.xiA[:3, :] *= self.s
        self.xiB[:3, :] *= self.s
        self.xiC[:3, :] *= self.s
        xiX[:3] *= self.s
        xiY[:3] *= self.s
        xiZ[:3] *= self.s

        # prepare base parameters vector
        p0 = np.hstack([xiX, xiY, xiZ])

        # ---- build scaled covariances for A,C if requested ----
        if self.use_cov_weight and (self.var_t_A is not None) and (self.var_a_A is not None) \
        and (self.var_t_C is not None) and (self.var_a_C is not None):
            def _to_euler_cov(var_t, var_a, i):
                def _pick(v):
                    if np.isscalar(v): return np.array([v, v, v])
                    v = np.asarray(v)
                    if v.ndim == 1: return np.array([v[i]] * 3)
                    if v.ndim == 2: return v[i]
                    raise ValueError("var format must be scalar, (n,), or (n,3)")
                
                # [t_x, t_y, t_z, roll, pitch, yaw] 순서의 분산 벡터
                diag_euler = np.hstack([_pick(var_t), _pick(var_a)])
                return np.diag(diag_euler)

            SigmaA_list, SigmaC_list = [], []
            for i in range(self.n):
                # 1. 먼저 Euler+Translation 공간에서 공분산 행렬을 만듭니다.
                SigmaA_euler = _to_euler_cov(self.var_t_A, self.var_a_A, i)
                SigmaC_euler = _to_euler_cov(self.var_t_C, self.var_a_C, i)
                
                # 2. 현재 포즈(A, C)에서 Euler 각도를 추출합니다. (상태 의존적 야코비안 계산 위함)
                r_A = Rotation.from_matrix(self.A[i][:3, :3])
                roll_A, pitch_A, yaw_A = r_A.as_euler('xyz', degrees=False) # 'XYZ' extrinsic
                
                r_C = Rotation.from_matrix(self.C[i][:3, :3])
                roll_C, pitch_C, yaw_C = r_C.as_euler('xyz', degrees=False) # 'XYZ' extrinsic

                # 3. Euler 공간 -> se(3) 공간으로 변환하는 야코비안(J_map)을 계산합니다.
                J_map_A = get_XYZ_extrinsic_to_se3_jacobian(roll_A, pitch_A)
                J_map_C = get_XYZ_extrinsic_to_se3_jacobian(roll_C, pitch_C)
                
                # 4. 공분산 전파 법칙 (Σ_se3 = J_map @ Σ_euler @ J_map.T) 을 적용합니다.
                #    이것이 Euler 공분산을 se(3) 공분산으로 변환하는 핵심 단계입니다. 🌉
                SA_se3 = J_map_A @ SigmaA_euler @ J_map_A.T
                SC_se3 = J_map_C @ SigmaC_euler @ J_map_C.T
                
                # 5. 마지막으로, 최적화를 위한 파라미터 스케일링(S)을 적용합니다.
                SigmaA_list.append(S @ SA_se3 @ S.T)
                SigmaC_list.append(S @ SC_se3 @ S.T)

            self.SigmaA_scaled = np.stack(SigmaA_list, axis=0)
            self.SigmaC_scaled = np.stack(SigmaC_list, axis=0)
            # print(np.diag(self.SigmaA_scaled[0])) 
        else:
            self.SigmaA_scaled = None
            self.SigmaC_scaled = None

        nshow = min(6, self.n)
        idxs = np.linspace(0, self.n-1, nshow, dtype=int)
        # self.debug_compare_JA_JC(xiX, xiY, xiZ, idxs=idxs, h=1e-7, use_central=True)

        # choose which residual builder to use
        if self.use_cov_weight and (self.SigmaA_scaled is not None) and (self.SigmaC_scaled is not None):
            fun = lambda p: self._compute_residuals_and_jacobian_weighted(p)[0]
            jac = lambda p: self._compute_residuals_and_jacobian_weighted(p)[1]
        else:
            # fallback: unweighted (original)
            fun = lambda p: self._compute_residuals_and_jacobian_weighted(p)[0]  # same method returns raw if off
            jac = lambda p: self._compute_residuals_and_jacobian_weighted(p)[1]

        res = least_squares(fun, p0, jac=jac, method='lm', verbose=0, **self.ls_kwargs)

        # unscale back translations in solution
        xiX_opt, xiY_opt, xiZ_opt = res.x[0:6].copy(), res.x[6:12].copy(), res.x[12:18].copy()
        xiX_opt[:3] /= self.s
        xiY_opt[:3] /= self.s
        xiZ_opt[:3] /= self.s

        X = trexp(xiX_opt)
        Y = trexp(xiY_opt)
        Z = trexp(xiZ_opt)

        out = {'X': X, 'Y': Y, 'Z': Z, 'result': res}
        # optional: posterior covariance ≈ (Jᵀ W J)^{-1} at optimum
        try:
            e_opt, J_opt = self._compute_residuals_and_jacobian_weighted(res.x)
            # build W block-diagonal only if cov-weighting on; else identity
            if self.use_cov_weight and (self.SigmaA_scaled is not None) and (self.SigmaC_scaled is not None):
                # we already prewhitened -> effectively W = I, so posterior ≈ (J_optᵀ J_opt)^{-1}
                # because J_opt is whitened. (safer numerically)
                JTJ = J_opt.T @ J_opt
            else:
                JTJ = J_opt.T @ J_opt
            out['Cov_theta_approx'] = np.linalg.pinv(JTJ)
        except Exception:
            pass

        return out
