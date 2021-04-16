import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import pca

TRAIN_SEED=1
VALIDATION_SEED=2
TEST_SEED=3

def generate_data(num_points, seed):
    scale = np.diag(np.sqrt(np.array([0.01, 0.1, 1][::-1])))
    rotate1 = R.from_rotvec(np.array([np.deg2rad(45.), 0, 0])).as_matrix()
    rotate2 = R.from_rotvec(np.array([0, np.deg2rad(45.), 0])).as_matrix()
    rotate3 = R.from_rotvec(np.array([0, 0, np.deg2rad(45.)])).as_matrix()
    chol = rotate3 @ rotate2 @ rotate1 @ scale

    cov = chol @ chol.T
    pca_w, _ = pca(cov)
    assert np.allclose(pca_w, np.array([1., 0.1, 0.01]))

    rs = np.random.RandomState(seed=seed)
    samples = rs.randn(3, num_points)
    data = (chol @ samples).T
    return data