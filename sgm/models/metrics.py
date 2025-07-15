from typing import Tuple
import scipy
import numpy as np

from torchmetrics.image.inception import InceptionScore

def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)

def compute_IS(imgs):
    inception = InceptionScore()
    inception.update(imgs)
    IS = inception.compute()
    
    return float(IS)
