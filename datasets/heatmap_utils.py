import numpy as np
import cv2

def compute_heatmap(points, image_size, k_ratio=3, transpose=False):
    """
    Generate a heatmap based on point locations with Gaussian blur.

    Args:
        points (list): List of normalized coordinates, e.g., [[0.5, 0.5], [0.3, 0.8]].
        image_size (tuple): Image size as (height, width).
        k_ratio (int): Ratio to control Gaussian kernel size; larger means smaller kernel.
        transpose (bool): Whether to transpose the final heatmap.

    Returns:
        np.ndarray: Normalized heatmap.
    """
    points = np.asarray(points)
    height, width = image_size  

    heatmap = np.zeros((height, width), dtype=np.float32)
    n_points = points.shape[0]

    for i in range(n_points):
        x = points[i, 0] * width     # Normalize x to width
        y = points[i, 1] * height    # Normalize y to height
        col = int(x)
        row = int(y)
        try:
            heatmap[row, col] += 1.0
        except:
            col = min(max(col, 0), width - 1)
            row = min(max(row, 0), height - 1)
            heatmap[row, col] += 1.0

    k_size = int(np.sqrt(height * width) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    if transpose:
        heatmap = heatmap.transpose()

    return heatmap


def find_max_value_location(heatmap):
    """
    Find the normalized location of the maximum value in the heatmap.

    Args:
        heatmap (np.ndarray): The heatmap array.

    Returns:
        np.ndarray: Normalized coordinates (x, y) of the maximum value.
    """
    max_value = np.max(heatmap)
    h, w = heatmap.shape  # height, width
    y, x = np.where(heatmap == max_value)  # (row, col)

    max_loc = (x[0] / w, y[0] / h)  # (norm_x, norm_y)
    return np.array(max_loc)
