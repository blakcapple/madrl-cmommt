import numpy as np
def rgba2rgb(rgba):
    # rgba is a list of 4 color elements btwn [0.0, 1.0]
    # or a 2d np array (num_colors, 4)
    # returns a list of rgb values between [0.0, 1.0] accounting for alpha and background color [1, 1, 1] == WHITE
    if isinstance(rgba, list):
        alpha = rgba[3]
        r = max(min((1 - alpha) * 1.0 + alpha * rgba[0], 1.0), 0.0)
        g = max(min((1 - alpha) * 1.0 + alpha * rgba[1], 1.0), 0.0)
        b = max(min((1 - alpha) * 1.0 + alpha * rgba[2], 1.0), 0.0)
        return [r, g, b]
    elif rgba.ndim == 2:
        alphas = rgba[:, 3]
        r = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 0], 0, 1)
        g = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 1], 0, 1)
        b = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 2], 0, 1)
        return np.vstack([r, g, b]).T
