import numpy as np
from scipy.spatial.transform import Rotation

def _inv4(T):
    """4x4 동차 변환 행렬의 역행렬을 계산합니다."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti