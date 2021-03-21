import numpy as np
import scipy.ndimage

import torch


class MSEFilterLoss:
    def __call__(self, x, y, val, device):
        k = torch.tensor(self.create_filter(x, val)).to(dtype=torch.float32, device=device)
        return torch.mean(torch.pow(k * (x - y), 2))

    def create_filter(self, x, val):
        n, c, h, w = x.shape
        k = np.zeros((h, w))
        k[h//2, w//2] = val
        f = scipy.ndimage.gaussian_filter(k, sigma=(h//4, w//4))
        return np.stack([[f for _ in range(c)] for _ in range(n)])
