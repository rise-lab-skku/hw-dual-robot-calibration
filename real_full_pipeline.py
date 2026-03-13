# main_full_pipeline.py
# -*- coding: utf-8 -*-
# 1) cam1, cam2 division pre-calibration (intrinsics + division-undistort PnP)
# 2) solve_init_two_step_abcd 초기화
# 3) dual / dual-bicamera / (옵션) Lie 최적화 (intrinsics 고정)
# 4) 공통 metric으로 결과 비교
# 5) [NEW] 여러 번 시뮬레이션 반복 & 데이터 조합/결과 저장


############ 변수명 표기와 convention이 다르기 때문에 코드 내에서 주석 반드시 참고 ############
# TODO: 변수명 일관성 위해서 코드 전체적으로 aTb 표기법으로 바꾸는 리팩토링 필요
# Convention:
#   aTb == ^a T_b
#   meaning: transform points expressed in frame b into frame a.
#   p_a = aTb @ p_b
#
# Examples:
#   cTb = ^C T_B   : board point -> camera frame
#   bTe = ^B T_E   : ee point    -> base frame
#   eTb = ^E T_B   : base point  -> ee frame


import argparse
from pathlib import Path
import numpy as np
import yaml
import cv2
import json
import datetime as _dt
import traceback

# --- 프로젝트 모듈 ---
from solver.initialization import solve_axyb_dq, solve_init_two_step_abcd
from solver.uncertainty import (
    run_optimization_with_vce_unified,
    run_optimization_with_vce_dual,
    run_optimization_with_vce_dual_bicamera,
)
from solver.lie import LieOptimizationSolverAXBYCZD as LieOptimizationSolver

from utils.projection import DivisionIntrinsics, DivisionProjector
from utils.metric import Metrics
from sim.dual_datagen import undistort_points_division

# -------------------------
# Utilities
# -------------------------
def _inv4(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def _rvec_tvec_to_mat4(rvec, tvec):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.flatten()
    return T

def _log_se3(T):
    """[ρ, ω] with V^{-1} approx; 변환 차이 측정용"""
    R = T[:3, :3]; t = T[:3, 3]
    tr = np.trace(R)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)
    if abs(theta) < 1e-12:
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

def load_intrinsics_yaml(path: Path):
    with path.open("r") as f:
        data = yaml.safe_load(f)
    K = np.array(data["K"], dtype=float).reshape(3, 3)
    D = np.array(data["dist"], dtype=float).reshape(-1,)
    print(f"[정보] 카메라 파라미터 로드: {path}")
    return K, D

def save_division_yaml(path: Path, intr: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": "division", **intr}
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[저장] Division intrinsics → {path}")

def make_charuco_board(rows: int, cols: int, square_size: float, marker_size: float, dict_name: str):
    aruco_dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
    board = cv2.aruco.CharucoBoard_create(
        squaresX=rows,
        squaresY=cols,
        squareLength=square_size,
        markerLength=marker_size,
        dictionary=aruco_dict
    )
    print(f"[정보] Charuco 보드 {rows}x{cols} (dict={dict_name})")
    return board

def load_charuco_data(root: Path, board, K, D, min_corners: int, allowed_stems=None):
    """
    root: cam?/ (poses/*.yaml, images/*.png)
    """
    print(f"\n[단계 1] 데이터 로딩 + Charuco 코너 검출: {root}")
    pose_dir = root / "poses"
    img_dir  = root / "images"
    files = sorted(pose_dir.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"포즈 파일 없음: {pose_dir}")

    all_stems = [pf.stem for pf in files]

    if allowed_stems is not None:
        allowed_stems = set(allowed_stems)
        selected_files = [pf for pf in files if pf.stem in allowed_stems]
        skipped_stems  = [s for s in all_stems if s not in allowed_stems]
        selected_stems = [pf.stem for pf in selected_files]
        selected_indices_in_all = [all_stems.index(s) for s in selected_stems]
        files = selected_files
        print(f"  - 외부 선택 프레임만 사용: {len(files)}개 | 제외: {len(skipped_stems)}개")
    else:
        skipped_stems = []
        selected_stems = all_stems[:]
        selected_indices_in_all = list(range(len(all_stems)))

    aruco_dict = board.dictionary
    params = cv2.aruco.DetectorParameters_create()
    max_reproj_error_px = 10.0

    out = dict(
        T_base_to_ee_list=[],
        image_points_list=[],
        object_points_list=[],
        T_board_to_cam_list_for_dq=[],
        all_stems=all_stems,
        selected_stems=selected_stems,
        skipped_stems=skipped_stems,
        selected_indices_in_all=selected_indices_in_all,
    )

    for pf in files:
        data = yaml.safe_load(pf.read_text())
        bTe = np.array(data["T_Base_to_EE"], dtype=float) # ^B T_E  (EE -> Base)

        imf = img_dir / f"{pf.stem}.png"
        if not imf.exists():
            print(f"  - 경고: 이미지 없음: {imf.name} (skip)")
            continue

        img = cv2.imread(str(imf))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1) ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if ids is None or len(ids) == 0:
            print(f"  - 실패: 마커 없음: {imf.name}")
            continue

        # 2) Charuco
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if not (retval and ch_corners is not None and len(ch_corners) >= min_corners):
            print(f"  - 실패: 유효 코너 부족 ({len(ch_corners) if ch_corners is not None else 0}) @ {imf.name}")
            continue

        obj_pts = board.chessboardCorners[ch_ids.flatten()]
        img_pts = ch_corners

        # 3) Charuco pose (poly K,D)
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, K, D, None, None)
        if not ok:
            print(f"  - 경고: estimatePoseCharucoBoard 실패 @ {imf.name}")
            continue

        # 4) reproj filter
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        err = cv2.norm(img_pts, proj, cv2.NORM_L2) / len(obj_pts)
        if err >= max_reproj_error_px:
            print(f"  - 경고: 재투영 오차↑ {err:.2f}px @ {imf.name} (제거)")
            continue

        cTb = _rvec_tvec_to_mat4(rvec, tvec) # ^C T_W  (Board -> Camera), from estimatePoseCharucoBoard

        out["T_base_to_ee_list"].append(bTe)                    # list of ^B T_E
        out["image_points_list"].append(img_pts.reshape(-1, 2)) # (N,2)
        out["object_points_list"].append(obj_pts)               # (N,3)
        out["T_board_to_cam_list_for_dq"].append(cTb)           # list of ^C T_W

    print(f"  처리 프레임 수 = {len(out['T_base_to_ee_list'])} / 선택 {len(selected_stems)} / 전체 {len(all_stems)}")
    if skipped_stems:
        print(f"  (참고) 제외된 stem: {', '.join(skipped_stems[:10])}" + (" ..." if len(skipped_stems) > 10 else ""))
    return out

# -------------------------
# Division undistort + PnP
# -------------------------
def K_from_division_intrinsics(intr: dict) -> np.ndarray:
    fx_eff = intr["c"] / intr["sx"]
    fy_eff = intr["c"] / (intr["sy"] if intr.get("include_sy", False) else intr["sx"])
    cx, cy = intr["cx"], intr["cy"]
    K = np.array([[fx_eff, 0, cx],
                  [0, fy_eff, cy],
                  [0, 0, 1.0]], dtype=float)
    return K

def build_cTb_list_division_pnp(obj_pts_list, img_pts_list, intr_div):
    """
    division 언디스토트 -> PnP 로 ^C T_B 리스트를 만든다.
    """
    c, kappa = intr_div['c'], intr_div['kappa']
    sx = intr_div['sx']
    sy = intr_div.get('sy', sx)
    cx, cy = intr_div['cx'], intr_div['cy']
    K_eff = np.array([[c/sx, 0,   cx],
                      [0,   c/sy, cy],
                      [0,   0,    1.0]], dtype=np.float64)

    cTb_list = []
    for P, uv_meas in zip(obj_pts_list, img_pts_list):
        intr_obj = DivisionIntrinsics(c=c, kappa=kappa, sx=sx, sy=sy, cx=cx, cy=cy)
        uv_u = undistort_points_division(uv_meas, intr_obj)  # (N,2)

        okA, rvecA, tvecA = cv2.solvePnP(P, uv_u, K_eff, None, flags=cv2.SOLVEPNP_ITERATIVE)
        TA = None
        if okA:
            RA, _ = cv2.Rodrigues(rvecA)
            TA = np.eye(4); TA[:3,:3] = RA; TA[:3,3] = tvecA.flatten()

        # 정규화 평면 가정 (백업)
        xdyd = np.empty_like(uv_u, dtype=np.float64)
        xdyd[:,0] = (uv_u[:,0] - cx) * sx
        xdyd[:,1] = (uv_u[:,1] - cy) * sy
        I = np.eye(3, dtype=np.float64)
        okB, rvecB, tvecB = cv2.solvePnP(P, xdyd, I, None, flags=cv2.SOLVEPNP_ITERATIVE)
        TB = None
        if okB:
            RB, _ = cv2.Rodrigues(rvecB)
            TB = np.eye(4); TB[:3,:3] = RB; TB[:3,3] = tvecB.flatten()

        # 간단히 A 우선
        cTb_list.append(TA if TA is not None else TB)
    return cTb_list

# -------------------------
# Metrics & compose
# -------------------------
def _project_point_div(pc, c, kappa, sx, sy, cx, cy):
    X, Y, Z = pc
    if Z <= 1e-12:
        return np.nan, np.nan
    ux, uy = c * (X/Z), c * (Y/Z)
    ru2 = ux*ux + uy*uy
    if abs(kappa) < 1e-20 or ru2 < 1e-24:
        xd, yd = ux, uy
    else:
        g = 1.0 - 4.0*kappa*ru2
        if g <= 0.0:
            return np.nan, np.nan
        Delta = np.sqrt(g)
        ru = np.sqrt(ru2)
        rd = (1.0 - Delta) / (2.0*kappa*ru)
        s  = rd / (ru + 1e-12)
        xd, yd = s*ux, s*uy
    u = xd / sx + cx
    v = yd / sy + cy
    return u, v

def _rmse_reproject_division(T_cam_board, obj_pts, img_obs, intr):
    c, kappa = intr['c'], intr['kappa']
    sx, sy   = intr['sx'], intr.get('sy', intr['sx'])
    cx, cy   = intr['cx'], intr['cy']
    R = T_cam_board[:3,:3]; t = T_cam_board[:3,3]
    errs = []
    for P, (uo, vo) in zip(obj_pts, img_obs):
        pc = R @ P + t
        u, v = _project_point_div(pc, c, kappa, sx, sy, cx, cy)
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        errs.append((u-uo)**2 + (v-vo)**2)
    return float(np.sqrt(np.mean(errs))) if errs else None

def _compose_c1c2_chain(X1_EC, T_E1B1_i, T_B1B2, T_B2E2_i, E2_C2):
    """^C1T_C2 = X1_EC * E1B1_i * B1B2 * B2E2_i * E2_C2"""
    return (((X1_EC @ T_E1B1_i) @ (T_B1B2 @ T_B2E2_i)) @ E2_C2)

def compute_bidir_metrics(
    label: str,
    X1_EC, T_B1B2, E2_C2,
    T_E1B1_list, T_E2B2_list,
    c1Tb_list, c2Tb_list,         # ^C1T_B, ^C2T_B
    obj1_list, img1_list,         # cam1 보드 3D/2D
    obj2_list, img2_list,         # cam2 보드 3D/2D
    intr1, intr2
):
    n = min(len(T_E1B1_list), len(T_E2B2_list),
            len(c1Tb_list), len(c2Tb_list),
            len(obj1_list), len(img1_list),
            len(obj2_list), len(img2_list))
    if n == 0:
        print(f"[{label}] 평가할 공통 프레임이 없습니다.")
        return None

    sum_dt_c1 = sum_dr_c1 = 0.0
    sum_dt_c2 = sum_dr_c2 = 0.0
    max_dt_c1 = max_dr_c1 = 0.0
    max_dt_c2 = max_dr_c2 = 0.0

    chain_sq_errs = []
    rmse_chain_list_c1, rmse_chain_list_c2 = [], []

    pooled_num = 0.0
    pooled_den = 0

    def _num_points(obj):
        try:
            return int(obj.shape[0])
        except AttributeError:
            return int(len(obj))

    for i in range(n):
        M_i = _compose_c1c2_chain(X1_EC, T_E1B1_list[i], T_B1B2, _inv4(T_E2B2_list[i]), E2_C2)
        M_inv = _inv4(M_i)

        # cam2 → cam1
        c2Tb_pnp  = c2Tb_list[i]
        c1Tb_pred = M_i @ c2Tb_pnp
        c1Tb_pnp  = c1Tb_list[i]

        Terr1   = _inv4(c1Tb_pnp) @ c1Tb_pred
        xi1     = _log_se3(Terr1)
        dt1_mm  = float(np.linalg.norm(xi1[:3]) * 1000.0)
        dr1_deg = float(np.linalg.norm(xi1[3:]) * 180.0/np.pi)

        rmse_chain_c1 = _rmse_reproject_division(c1Tb_pred, obj1_list[i], img1_list[i], intr1)

        sum_dt_c1 += dt1_mm; sum_dr_c1 += dr1_deg
        max_dt_c1 = max(max_dt_c1, dt1_mm); max_dr_c1 = max(max_dr_c1, dr1_deg)
        if rmse_chain_c1 is not None:
            chain_sq_errs.append(rmse_chain_c1*rmse_chain_c1)
            rmse_chain_list_c1.append(rmse_chain_c1)

        # cam1 → cam2
        c1Tb_pnp2 = c1Tb_list[i]
        c2Tb_pred = M_inv @ c1Tb_pnp2
        c2Tb_pnp2 = c2Tb_list[i]

        Terr2   = _inv4(c2Tb_pnp2) @ c2Tb_pred
        xi2     = _log_se3(Terr2)
        dt2_mm  = float(np.linalg.norm(xi2[:3]) * 1000.0)
        dr2_deg = float(np.linalg.norm(xi2[3:]) * 180.0/np.pi)

        rmse_chain_c2 = _rmse_reproject_division(c2Tb_pred, obj2_list[i], img2_list[i], intr2)
        m2 = _num_points(obj2_list[i])

        sum_dt_c2 += dt2_mm; sum_dr_c2 += dr2_deg
        max_dt_c2 = max(max_dt_c2, dt2_mm); max_dr_c2 = max(max_dr_c2, dr2_deg)
        if rmse_chain_c2 is not None:
            chain_sq_errs.append(rmse_chain_c2*rmse_chain_c2)
            rmse_chain_list_c2.append(rmse_chain_c2)
            pooled_num += (rmse_chain_c2 * rmse_chain_c2) * m2
            pooled_den += m2

    mean_dt_c1 = sum_dt_c1 / n
    mean_dr_c1 = sum_dr_c1 / n
    mean_dt_c2 = sum_dt_c2 / n
    mean_dr_c2 = sum_dr_c2 / n

    bidir_rmse_chain = float(np.sqrt(np.mean(chain_sq_errs))) if chain_sq_errs else np.nan
    mean_rmse_c1 = float(np.mean(rmse_chain_list_c1)) if rmse_chain_list_c1 else np.nan
    mean_rmse_c2 = float(np.mean(rmse_chain_list_c2)) if rmse_chain_list_c2 else np.nan

    overall_mean_dt_mm = 0.5 * (mean_dt_c1 + mean_dt_c2)
    overall_mean_dr_deg = 0.5 * (mean_dr_c1 + mean_dr_c2)
    overall_mean_rmse_px = float(
        np.mean(rmse_chain_list_c1 + rmse_chain_list_c2)
    ) if (rmse_chain_list_c1 or rmse_chain_list_c2) else np.nan

    pooled_global_rmse = float(np.sqrt(pooled_num / pooled_den)) if pooled_den > 0 else np.nan

    print(f"\n=== [{label}] Bi-directional metrics ===")
    print(f"  cam2→cam1 평균 Δt = {mean_dt_c1:7.2f} mm, 평균 ΔR = {mean_dr_c1:6.2f} deg | 최대 Δt = {max_dt_c1:7.2f} mm, 최대 ΔR = {max_dr_c1:6.2f} deg")
    print(f"  cam1→cam2 평균 Δt = {mean_dt_c2:7.2f} mm, 평균 ΔR = {mean_dr_c2:6.2f} deg | 최대 Δt = {max_dt_c2:7.2f} mm, 최대 ΔR = {max_dr_c2:6.2f} deg")
    print(f"  cam2→cam1 평균 RMSE(chain, div) = {mean_rmse_c1:.3f} px")
    print(f"  cam1→cam2 평균 RMSE(chain, div) = {mean_rmse_c2:.3f} px")
    print(f"  양방향 RMSE(chain, division) = {bidir_rmse_chain:.3f} px")
    print(f"  >>> 전체 평균 Δt = {overall_mean_dt_mm:7.2f} mm, 전체 평균 ΔR = {overall_mean_dr_deg:6.2f} deg  (양방향 통합)")
    print(f"  >>> 전체 평균 RMSE(chain, div) = {overall_mean_rmse_px:.3f} px  (프레임-균등)")
    print(f"  >>> 전역 RMSE(chain, div) = {pooled_global_rmse:.3f} px  (픽셀-가중)")

    return {
        "label": label,
        "mean_dt_c1": float(mean_dt_c1), "mean_dr_c1": float(mean_dr_c1),
        "max_dt_c1": float(max_dt_c1),   "max_dr_c1": float(max_dr_c1),
        "mean_dt_c2": float(mean_dt_c2), "mean_dr_c2": float(mean_dr_c2),
        "max_dt_c2": float(max_dt_c2),   "max_dr_c2": float(max_dr_c2),
        "bidir_rmse_chain_px": float(bidir_rmse_chain),
        "overall_mean_dt_mm": float(overall_mean_dt_mm),
        "overall_mean_dr_deg": float(overall_mean_dr_deg),
        "overall_mean_rmse_px": float(overall_mean_rmse_px),
        "pooled_global_rmse_px": float(pooled_global_rmse),
        "n_eval": int(n),
    }

def short_summary(m):
    mean_dt_bi = 0.5 * (m["mean_dt_c1"] + m["mean_dt_c2"])
    mean_dr_bi = 0.5 * (m["mean_dr_c1"] + m["mean_dr_c2"])
    return mean_dt_bi, mean_dr_bi, m["bidir_rmse_chain_px"]

# -------------------------
# [NEW] 직렬화 & 저장 유틸
# -------------------------
def _to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64, np.integer)):
        return int(o)
    return o

def _dump_json(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_to_jsonable)
    print(f"[저장] {out_path}")

def _append_jsonl(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_to_jsonable))
        f.write("\n")

# -------------------------
# [NEW] 단일 실험 실행 함수
# -------------------------
def run_single_experiment(args, base_seed, run_idx):
    """
    하나의 데이터 조합을 샘플링해 전체 파이프라인 실행, 
    compute_bidir_metrics(학습/홀드아웃) 결과와 스템/인덱스 저장.
    """
    root = Path(args.data_root)
    cam1_root = root / "cam1"
    cam2_root = root / "cam2"
    cam1_intr = cam1_root / "camera_info.yaml"
    cam2_intr = cam2_root / "camera_info.yaml"

    # ---- (A) 두 카메라 pose 파일 스템 교집합 추출 + 샘플링 ----
    def _pose_stems(p: Path) -> set[str]:
        pose_dir = p / "poses"
        return {f.stem for f in pose_dir.glob("*.yaml")}

    stems1 = _pose_stems(cam1_root)
    stems2 = _pose_stems(cam2_root)
    all_common = sorted(stems1 & stems2)
    if not all_common:
        raise RuntimeError("cam1/cam2 공통 포즈 스템이 없습니다. 파일명을 맞추세요.")

    import random
    rng = random.Random(base_seed + run_idx)  # 각 run마다 고유 시드
    stems_shuffled = all_common[:]
    rng.shuffle(stems_shuffled)

    if args.num_frames > 0:
        selected = stems_shuffled[:args.num_frames]
    else:
        selected = stems_shuffled[:]  # 전체 사용

    allowed = set(selected)
    remaining = sorted(set(all_common) - allowed)

    if len(remaining) > 0 and args.holdout_frames > 0:
        k = min(args.holdout_frames, len(remaining))
        holdout_stems = rng.sample(remaining, k)
    else:
        holdout_stems = []

    print(f"\n[Run {run_idx}] 공통 프레임 샘플링: 선택={len(allowed)} / 교집합={len(all_common)} / hold-out={len(holdout_stems)}")

    # ================================
    # 1) cam1, cam2 pre-calibration
    # ================================
    print("\n================ CAM1: Pre-Calibration ================")
    res1 = pre_calibrate_one_camera(
        cam_root=cam1_root, intr_yaml=cam1_intr,
        rows=args.rows, cols=args.cols, square=args.square, marker=args.marker, dict_name=args.dict,
        sigma_px=args.sigma_px, sigma_deg=args.sigma_deg, sigma_mm=args.sigma_mm,
        include_sy=args.include_sy, c_known_mm=args.c_mm,
        allowed_stems=allowed,
    )

    print("\n================ CAM2: Pre-Calibration ================")
    res2 = pre_calibrate_one_camera(
        cam_root=cam2_root, intr_yaml=cam2_intr,
        rows=args.rows, cols=args.cols, square=args.square, marker=args.marker, dict_name=args.dict,
        sigma_px=args.sigma_px, sigma_deg=args.sigma_deg, sigma_mm=args.sigma_mm,
        include_sy=args.include_sy, c_known_mm=args.c_mm,
        allowed_stems=allowed,
    )

    # 초기/최종 RRMSE 요약
    init_summary = {
        "cam1_rrmse_init": float(res1['rrmse_init']),
        "cam1_rrmse_final": float(res1['rrmse_final']),
        "cam2_rrmse_init": float(res2['rrmse_init']),
        "cam2_rrmse_final": float(res2['rrmse_final']),
    }

    # ==========================================
    # 2) 초기화 (solve_init_two_step_abcd)
    # ==========================================
    print("\nSolve AXB=YCZD init...(Two-Step method, division PnP)")
    A_in = res1["valid_data"]["T_base_to_ee_list"]               # list of ^B1 T_E1
    C_in = res2["valid_data"]["T_base_to_ee_list"]               # list of ^B2 T_E2
    # 아래 두 줄은 Charuco(poly)를 참조하도록 고정
    B_in = res1["valid_data"]["T_board_to_cam_list_for_dq"]      # list of ^C1 T_W
    D_in = res2["valid_data"]["T_board_to_cam_list_for_dq"]      # list of ^C2 T_W

    if not (len(A_in) and len(C_in) and len(B_in) and len(D_in)):
        raise RuntimeError("초기화에 필요한 리스트 길이가 0입니다. 데이터를 확인하세요.")

    c1Te1, b2Tb1, c2Te2 = solve_init_two_step_abcd(A_in, B_in, C_in, D_in)
    print("초기값 계산 성공:")
    print("  - c1Te1 (^C1T_E1):\n", c1Te1) # ^E1 T_C1
    print("  - b2Tb1 (^B2T_B1):\n", b2Tb1) # ^B1 T_B2
    print("  - c2Te2 (^C2T_E2):\n", c2Te2) # ^E2 T_C2

    # ==========================================
    # 3) Dual / Dual-Bi / (옵션) Lie 최적화
    # ==========================================
    intr1 = res1["intrinsics_div_final"]
    intr2 = res2["intrinsics_div_final"]

    T_E1B1_list_init = [_inv4(T) for T in res1["valid_data"]["T_base_to_ee_list"]]   # ^E1 T_B1
    T_E2B2_list_init = [_inv4(T) for T in res2["valid_data"]["T_base_to_ee_list"]]   # ^E2 T_B2
    T_C1B_list_init  = res1["valid_data"]["T_board_to_cam_list_for_dq"]              # ^C1 T_W
    T_C2B_list_init  = res2["valid_data"]["T_board_to_cam_list_for_dq"]              # ^C2 T_W

    obj1_list = res1["valid_data"]["object_points_list"]
    img1_list = res1["valid_data"]["image_points_list"]
    obj2_list = res2["valid_data"]["object_points_list"]
    img2_list = res2["valid_data"]["image_points_list"]

    X1_EC_init = _inv4(c1Te1)          # ^C1 T_E1
    T_B1B2_init = b2Tb1                # ^B1 T_B2
    E2_C2_init = c2Te2                 # ^E2 T_C2

    print("\n[Dual] Optimize (intrinsics fixed)")
    # Projection Cam2 -> EE2 -> Base2 -> Base1 -> EE1 -> Cam1
    dual_out = run_optimization_with_vce_dual(
        model_type='division',
        X1_EC_init=X1_EC_init,
        T_B1B2_init=T_B1B2_init,
        E2_C2_init=E2_C2_init, # X,Y,Z init
        T_E1B1_list_init=T_E1B1_list_init, # Robot 1 pose init
        T_E2B2_list_init=T_E2B2_list_init, # Robot 2 pose init
        T_C2B_list_init=T_C2B_list_init, # Cam2 보드 포즈 init
        # Board -> Cam2 -> EE2 -> Base2 -> Base1 -> EE1 -> Cam1
        img_pts_list=img1_list,
        obj_pts_list=obj1_list,
        # reprojection 계산을 할 Cam1 쪽 observation
        T_E1B1_list_obs=T_E1B1_list_init,
        T_E2B2_list_obs=T_E2B2_list_init,
        T_C2B_list_obs=None,
        intrinsics_init=intr1,
        sigma_image_px=args.sigma_px,
        sigma_angle_deg=args.sigma_deg,
        sigma_trans_mm=args.sigma_mm,
        max_vce_iter=5,
        max_param_iter=15,
        term_thresh=1e-6,
        estimate_x1ec=True, # X
        estimate_b1b2=True, # Y
        estimate_e2c2=True, # Z
        estimate_e1b1=True, # Robot 1 pose
        estimate_b2e2=True, # Robot 2 pose
        estimate_c2b=False, # Cam2 보드 포즈는 고정
        estimate_intrinsics=False,
        include_sy=intr1.get('include_sy', False),
        is_scara_x1=False,
    )
    (X1_EC_dual, T_B1B2_dual, E2_C2_dual, # X, Y, Z
     T_E1B1_dual, T_E2B2_dual, T_C2B_dual, intr1_dual) = dual_out

    print("\n[Dual-Bi] Optimize (intrinsics fixed)")
    bi_out = run_optimization_with_vce_dual_bicamera(
        model_type='division',
        X1_EC_init=X1_EC_init, T_B1B2_init=T_B1B2_init, E2_C2_init=E2_C2_init,
        T_E1B1_list_init=T_E1B1_list_init, T_E2B2_list_init=T_E2B2_list_init,
        T_C2B_list_init=T_C2B_list_init, T_C1B_list_init=T_C1B_list_init,
        obj_pts_list=obj1_list, img1_pts_list=img1_list, img2_pts_list=img2_list,
        T_E1B1_list_obs=T_E1B1_list_init, T_E2B2_list_obs=T_E2B2_list_init,
        T_C2B_list_obs=None, T_C1B_list_obs=None,
        intr1_init=intr1, intr2_init=intr2,
        sigma_image_px=args.sigma_px, sigma_angle_deg=args.sigma_deg, sigma_trans_mm=args.sigma_mm,
        max_vce_iter=5, max_param_iter=15, term_thresh=1e-6,
        estimate_x1ec=True, estimate_b1b2=True, estimate_e2c2=True,
        estimate_e1b1=True, estimate_b2e2=True,
        estimate_c2b=False, estimate_c1b=False,
        estimate_intr1=False, estimate_intr2=False,
        include_sy1=intr1.get('include_sy', False), include_sy2=intr2.get('include_sy', False),
        is_scara_x1=False,
    )
    (X1_EC_bi, T_B1B2_bi, E2_C2_bi,
     T_E1B1_bi, T_E2B2_bi, T_C2B_bi, T_C1B_bi,
     intr1_bi, intr2_bi) = bi_out

    # Lie 방식(실패 가능성 있음 → try)
    X1_EC_lie = T_B1B2_lie = E2_C2_lie = None
    try:
        print("\n[Dual-Lie] Optimize (intrinsics fixed)")
        lie_solver = LieOptimizationSolver(
            A=A_in, B=B_in, C=C_in, D=D_in,
            X0=c1Te1, Y0=b2Tb1, Z0=c2Te2
        )
        lie_results = lie_solver.solve()
        X_lie = lie_results['X']; Y_lie = lie_results['Y']; Z_lie = lie_results['Z']
        X1_EC_lie  = _inv4(X_lie)
        T_B1B2_lie = Y_lie
        E2_C2_lie  = Z_lie
    except Exception as e:
        print(f"[Dual-Lie] 실행 실패(건너뜀): {e}")

    # ==========================================
    # 4) Hold-out 데이터 구성
    # ==========================================
    if len(holdout_stems) > 0:
        valid_ho_1 = load_charuco_data(
            cam1_root,
            make_charuco_board(args.rows, args.cols, args.square, args.marker, args.dict),
            *load_intrinsics_yaml(cam1_intr),
            min_corners=8,
            allowed_stems=set(holdout_stems)
        )
        valid_ho_2 = load_charuco_data(
            cam2_root,
            make_charuco_board(args.rows, args.cols, args.square, args.marker, args.dict),
            *load_intrinsics_yaml(cam2_intr),
            min_corners=8,
            allowed_stems=set(holdout_stems)
        )

        c1Tb_poly_ho = valid_ho_1["T_board_to_cam_list_for_dq"]
        c2Tb_poly_ho = valid_ho_2["T_board_to_cam_list_for_dq"]
        T_E1B1_list_ho = [_inv4(T) for T in valid_ho_1["T_base_to_ee_list"]]
        T_E2B2_list_ho = [_inv4(T) for T in valid_ho_2["T_base_to_ee_list"]]
        obj1_ho = valid_ho_1["object_points_list"]; img1_ho = valid_ho_1["image_points_list"]
        obj2_ho = valid_ho_2["object_points_list"]; img2_ho = valid_ho_2["image_points_list"]
    else:
        valid_ho_1 = valid_ho_2 = None
        c1Tb_poly_ho = c2Tb_poly_ho = []
        T_E1B1_list_ho = T_E2B2_list_ho = []
        obj1_ho = img1_ho = obj2_ho = img2_ho = []

    # ==========================================
    # 5) 동일 metric으로 결과 비교 (학습셋 & 홀드아웃셋)
    # ==========================================
    # 학습셋
    metrics_train = {}
    metrics_train["init"] = compute_bidir_metrics(
        "Train / Initial (Two-step DQ)",
        X1_EC_init, T_B1B2_init, E2_C2_init,
        T_E1B1_list_init, T_E2B2_list_init,
        T_C1B_list_init, T_C2B_list_init,
        obj1_list, img1_list, obj2_list, img2_list,
        intr1, intr2
    )
    metrics_train["dual"] = compute_bidir_metrics(
        "Train / Dual (cam1 only, fixed intr)",
        X1_EC_dual, T_B1B2_dual, E2_C2_dual,
        T_E1B1_dual, T_E2B2_dual,
        T_C1B_list_init, T_C2B_list_init,
        obj1_list, img1_list, obj2_list, img2_list,
        intr1, intr2
    )
    metrics_train["dual_bi"] = compute_bidir_metrics(
        "Train / Dual-Bi (cam1+cam2, fixed intr)",
        X1_EC_bi, T_B1B2_bi, E2_C2_bi,
        T_E1B1_bi, T_E2B2_bi,
        T_C1B_list_init, T_C2B_list_init,
        obj1_list, img1_list, obj2_list, img2_list,
        intr1, intr2
    )
    metrics_train["lie"] = None
    if X1_EC_lie is not None:
        metrics_train["lie"] = compute_bidir_metrics(
            "Train / Dual-Lie (fixed intr)",
            X1_EC_lie, T_B1B2_lie, E2_C2_lie,
            T_E1B1_list_init, T_E2B2_list_init,
            T_C1B_list_init, T_C2B_list_init,
            obj1_list, img1_list, obj2_list, img2_list,
            intr1, intr2
        )

    # 홀드아웃셋
    metrics_holdout = None
    if len(holdout_stems) > 0:
        metrics_holdout = {
            "init": compute_bidir_metrics(
                "Hold-out / Initial (Two-step DQ)",
                X1_EC_init, T_B1B2_init, E2_C2_init,
                T_E1B1_list_ho, T_E2B2_list_ho,
                c1Tb_poly_ho, c2Tb_poly_ho,
                obj1_ho, img1_ho, obj2_ho, img2_ho,
                intr1, intr2
            ),
            "dual": compute_bidir_metrics(
                "Hold-out / Dual (cam1 only, fixed intr)",
                X1_EC_dual, T_B1B2_dual, E2_C2_dual,
                T_E1B1_list_ho, T_E2B2_list_ho,
                c1Tb_poly_ho, c2Tb_poly_ho,
                obj1_ho, img1_ho, obj2_ho, img2_ho,
                intr1, intr2
            ),
            "dual_bi": compute_bidir_metrics(
                "Hold-out / Dual-Bi (cam1+cam2, fixed intr)",
                X1_EC_bi, T_B1B2_bi, E2_C2_bi,
                T_E1B1_list_ho, T_E2B2_list_ho,
                c1Tb_poly_ho, c2Tb_poly_ho,
                obj1_ho, img1_ho, obj2_ho, img2_ho,
                intr1, intr2
            ),
            "lie": None
        }
        if X1_EC_lie is not None:
            metrics_holdout["lie"] = compute_bidir_metrics(
                "Hold-out / Dual-Lie (fixed intr)",
                X1_EC_lie, T_B1B2_lie, E2_C2_lie,
                T_E1B1_list_ho, T_E2B2_list_ho,
                c1Tb_poly_ho, c2Tb_poly_ho,
                obj1_ho, img1_ho, obj2_ho, img2_ho,
                intr1, intr2
            )

    # 선택/인덱스 기록(둘 다 같은 교집합에서 골라서 cam1/2 동일 스템)
    selection_info = {
        "train_stems": sorted(list(allowed)),
        "holdout_stems": sorted(list(holdout_stems)),
        # 각 카메라에서의 선택 인덱스(해당 카메라 all_stems 기준)
        "cam1": {
            "all_stems": res1["valid_data"]["all_stems"],
            "selected_indices_in_all": res1["valid_data"]["selected_indices_in_all"],
        },
        "cam2": {
            "all_stems": res2["valid_data"]["all_stems"],
            "selected_indices_in_all": res2["valid_data"]["selected_indices_in_all"],
        },
    }

    return {
        "run_index": run_idx,
        "seed_used": int(base_seed + run_idx),
        "init_rrmse": init_summary,
        "selection": selection_info,
        "metrics_train": metrics_train,
        "metrics_holdout": metrics_holdout,
    }

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="./data/0725", help="cam1, cam2 폴더가 있는 루트")
    ap.add_argument("--dict", type=str, default="DICT_4X4_50")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--square", type=float, default=0.095)
    ap.add_argument("--marker", type=float, default=0.074)
    ap.add_argument("--sigma_px", type=float, default=0.1)
    ap.add_argument("--sigma_deg", type=float, default=0.1)
    ap.add_argument("--sigma_mm", type=float, default=1.0)
    ap.add_argument("--include_sy", action="store_true")
    ap.add_argument("--c_mm", type=float, default=2.3)
    ap.add_argument("--num-frames", type=int, default=50, help="두 카메라 공통 프레임 중 샘플링할 개수")
    ap.add_argument("--holdout-frames", type=int, default=20, help="홀드아웃에 사용할 프레임 수(남은 공통 프레임에서 랜덤 샘플)")
    # [NEW]
    ap.add_argument("--num-runs", type=int, default=10, help="반복 실험 횟수")
    ap.add_argument("--random-seed", type=int, default=1236, help="샘플링 기본 시드")
    ap.add_argument("--out-dir", type=str, default="./runs_out", help="결과 저장 폴더")
    ap.add_argument("--tag", type=str, default="", help="결과 파일명에 붙일 태그(옵션)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (f"_{args.tag}" if args.tag else "")
    out_jsonl = out_dir / f"results_{ts}{tag}.jsonl"
    out_summary = out_dir / f"summary_{ts}{tag}.json"

    print(f"[정보] 결과 저장 위치:\n  - JSONL:  {out_jsonl}\n  - SUMMARY:{out_summary}")

    all_runs = []
    n_ok = 0
    n_fail = 0

    for run_idx in range(args.num_runs):
        try:
            result = run_single_experiment(args, args.random_seed, run_idx)
            _append_jsonl(result, out_jsonl)
            all_runs.append(result)
            n_ok += 1
        except Exception as e:
            tb = traceback.format_exc()
            err_obj = {
                "run_index": run_idx,
                "error": str(e),
                "traceback": tb,
            }
            _append_jsonl(err_obj, out_jsonl)
            all_runs.append(err_obj)
            n_fail += 1
            print(f"[경고] Run {run_idx} 실패: {e}")

    # 전체 요약 저장(간단 집계)
    summary = {
        "timestamp": ts,
        "num_runs_requested": int(args.num_runs),
        "num_runs_ok": int(n_ok),
        "num_runs_fail": int(n_fail),
        "args": vars(args),
        "runs": all_runs,
    }
    _dump_json(summary, out_summary)

    print(f"\n[완료] OK={n_ok}, FAIL={n_fail}. 결과 파일을 확인하세요.")

# ================================
# 기존 pre_calibrate_one_camera 함수 (변경 없음)
# ================================
def pre_calibrate_one_camera(
    cam_root: Path,
    intr_yaml: Path,
    rows: int, cols: int, square: float, marker: float, dict_name: str,
    sigma_px: float, sigma_deg: float, sigma_mm: float,
    include_sy: bool = False,
    c_known_mm: float = 2.3,
    allowed_stems = None
):
    # 0) 입력 로드
    K, D = load_intrinsics_yaml(intr_yaml)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    board = make_charuco_board(rows, cols, square, marker, dict_name)

    # 1) 데이터 로딩 + Charuco PnP(poly)
    valid = load_charuco_data(cam_root, board, K, D, min_corners=8, allowed_stems=allowed_stems)
    T_be_list_obs = [_inv4(T) for T in valid["T_base_to_ee_list"]] # list of ^E T_B = inverse(^B T_E)
    T_cb_list_dq  = valid["T_board_to_cam_list_for_dq"]            # list of ^C T_B

    if len(T_be_list_obs) < 5:
        raise RuntimeError("유효 프레임이 5개 미만 → 안정적 캘리브 안 됨")

    # 2) DQ(AXYB) 초기값
    print("\n[단계 2] DQ(AXYB) 초기값 계산...")
    wTb, cTe = solve_axyb_dq(T_cb_list_dq, T_be_list_obs)
    #   wTb = ^W T_B   (Base -> Board)
    #   cTe = ^C T_E   (EE -> Cam)
    print("  - 초기 cTe(^C T_E):\n", cTe)
    print("  - 초기 wTb(^W T_B):\n", wTb)

    bTw = _inv4(wTb) # ^B T_W (Board -> Base)

    # 3) Division intrinsics 초기
    sx_init = c_known_mm / fx
    sy_init = (c_known_mm / fy) if include_sy else sx_init
    intr_div_init = dict(c=c_known_mm, kappa=1e-8, sx=sx_init, sy=sy_init, cx=cx, cy=cy, include_sy=include_sy)

    # 4) 초기 RRMSE (division)
    proj_init = DivisionProjector(DivisionIntrinsics(**intr_div_init))
    metrics = Metrics()
    # Projection into Board -> Base -> EE -> Cam
    rrmse_init, _ = metrics.reproj_rmse(
        projector=proj_init,
        X_EC=cTe, X_WB=bTw,
        T_BE_list=T_be_list_obs,
        obj_pts_list=valid["object_points_list"], img_pts_list=valid["image_points_list"]
    )
    print("\n==================================================")
    print("Division Model Reprojection Error (Initial)")
    print("==================================================")
    print(f"  RRMSE:   {rrmse_init:.6f} px")
    print("==================================================\n")

    # 5) 최적화 (intrinsics 포함)
    print("\n[단계 3] run_optimization_with_vce_unified (division, intrinsics 포함) ...")
    X_EC, X_WB, T_BE_list, intr_div_final = run_optimization_with_vce_unified(
        model_type='division',
        T_ee_cam_init=cTe, T_base_board_init=bTw,
        T_be_list_init=T_be_list_obs,
        img_pts_list=valid["image_points_list"],
        obj_pts_list=valid["object_points_list"],
        T_be_list_obs=T_be_list_obs,
        intrinsics_init=intr_div_init,
        sigma_image_px=sigma_px, sigma_angle_deg=sigma_deg, sigma_trans_mm=sigma_mm,
        max_vce_iter=5, max_param_iter=15, term_thresh=1e-6,
        is_target_based=True,
        estimate_ec=True, estimate_wb=True, estimate_be=True, estimate_intrinsics=True,
        is_scara=False
    )

    print("\n--- 최종 캘리브레이션 결과 ---")
    print("최적화된 ^C T_E:\n", X_EC) # ^C T_E (EE -> Cam)
    print("\n최적화된 ^B T_W:\n", X_WB) # ^B T_W (Board -> Base)
    print("Intrinsic(Before)", intr_div_init)
    print("Intrinsic(After) ", intr_div_final)

    # 6) 최종 RRMSE
    proj_final = DivisionProjector(DivisionIntrinsics(
        c=intr_div_final["c"], kappa=intr_div_final["kappa"],
        sx=intr_div_final["sx"], sy=intr_div_final.get("sy", intr_div_final["sx"]),
        cx=intr_div_final["cx"], cy=intr_div_final["cy"],
        include_sy=intr_div_final.get("include_sy", False)
    ))
    # Projection into Board -> Base -> EE -> Cam
    rrmse_final, _ = metrics.reproj_rmse(
        projector=proj_final,
        X_EC=X_EC, X_WB=X_WB,
        T_BE_list=T_BE_list,
        obj_pts_list=valid["object_points_list"], img_pts_list=valid["image_points_list"]
    )
    print("\n==================================================")
    print("Division Model Reprojection Error (Calibrated)")
    print("==================================================")
    print(f"  RRMSE:   {rrmse_final:.6f} px")
    print("==================================================\n")

    print("[단계 4] 최종 division intrinsics로 undistort+PnP (cTb_list_division_pnp) 생성...")
    cTb_div = build_cTb_list_division_pnp(
        valid["object_points_list"], valid["image_points_list"], intr_div_final
    )
    print(f"  - 생성된 보드 포즈 수: {len(cTb_div)}")

    return {
        "intrinsics_div_init": intr_div_init,
        "intrinsics_div_final": intr_div_final,
        "rrmse_init": float(rrmse_init),
        "rrmse_final": float(rrmse_final),
        "X_EC": X_EC, "X_WB": X_WB, "T_BE_list": T_BE_list,
        "valid_data": valid,
        "cTb_list_division_pnp": cTb_div
    }

if __name__ == "__main__":
    main()