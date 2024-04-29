import numpy as np
from scipy.ndimage import median_filter


class ClassWiseMedianFilter:
    def __init__(self, filter_lens=(1, 1, 1)):
        self.filter_lens = filter_lens

    def __call__(self, x, **kwargs):

        out = []
        for indx_cls in range(x.shape[-1]):
            smoothed = median_filter(x[..., indx_cls][..., None],
                                     (self.filter_lens[indx_cls], 1))[:, 0]

            out.append(smoothed)
        out = np.stack(out, -1)
        return out