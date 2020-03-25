import numpy as np
import torch


def slerp(val, low, high):
    """
    https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792
    """
    omega = np.arccos(
        np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1)
    )
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def slerp_grid(top_left, top_right, bottom_left, bottom_right, nrows, ncols):
    left_col = [
        slerp(x, top_left, bottom_left) for x in np.linspace(0, 1, nrows)
    ]
    right_col = [
        slerp(x, top_right, bottom_right) for x in np.linspace(0, 1, nrows)
    ]
    rows = []
    for left, right in zip(left_col, right_col):
        row = [
            slerp(x, left, right) for x in np.linspace(0, 1, ncols)
        ]
        rows.append(torch.from_numpy(np.vstack(row)))
    grid = torch.cat(rows, dim=0)
    return grid
