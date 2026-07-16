from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA

from totalsegmentator.aorta_report.centerline import get_distance


def vox_to_mm_space(point, affine):
    return affine[:3, :3] @ point + affine[:3, 3]


def vox_to_mm_space_without_offset(point, affine):
    return affine[:3, :3] @ point


def vox_to_other_vox_space(point, original_affine, target_affine):
    return (np.linalg.inv(target_affine) @ original_affine @ np.append(point, 1))[:3]


def draw_plane(image_shape, normal_vec, point):
    normal_vec = np.asarray(normal_vec, dtype=float)
    norm = np.linalg.norm(normal_vec)
    if norm == 0:
        return np.zeros(image_shape, dtype=np.uint8)
    normal_vec /= norm
    point = np.asarray(point, dtype=float)
    coordinates = np.ogrid[tuple(slice(0, size) for size in image_shape)]
    distance = sum(
        (coordinates[axis] - point[axis]) * normal_vec[axis]
        for axis in range(len(image_shape))
    )
    return (np.abs(distance) < 1).astype(np.uint8)


def cleanup_plane(mask, closing=True, fill_holes=True):
    """Apply the shared binary cleanup used before plane measurements."""
    cleaned = np.asarray(mask) > 0
    if not cleaned.any():
        return np.zeros(cleaned.shape, dtype=np.uint8)
    if closing:
        cleaned = ndimage.binary_closing(cleaned)
    if fill_holes:
        cleaned = ndimage.binary_fill_holes(cleaned)
    return cleaned.astype(np.uint8)


def _farthest_pair(points):
    if len(points) == 0:
        return 0.0, None, None
    if len(points) == 1:
        return 0.0, points[0].copy(), points[0].copy()
    distances = pdist(points)
    pair_index = int(np.argmax(distances))
    count = len(points)
    first = count - 2 - int(
        np.floor(np.sqrt(-8 * pair_index + 4 * count * (count - 1) - 7) / 2 - 0.5)
    )
    second = (
        pair_index
        + first
        + 1
        - count * (count - 1) // 2
        + (count - first) * (count - first - 1) // 2
    )
    return (
        distances[pair_index],
        points[first].copy(),
        points[second].copy(),
    )


def get_region_diameters(image):
    for region in regionprops(label(image)):
        mask = region.image
        boundary = np.logical_xor(mask, ndimage.binary_erosion(mask))
        points = np.transpose(np.nonzero(boundary))
        diameter, point_a, point_b = _farthest_pair(points)
        point_a += np.asarray(region.bbox[: image.ndim])
        point_b += np.asarray(region.bbox[: image.ndim])
        yield diameter, point_a, point_b


def _angle(first_a, first_b, second_a, second_b):
    first = first_b - first_a
    second = second_b - second_a
    return abs(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-7))


def max_perpendicular_line(mask, line_start, line_end):
    if line_start is None or line_end is None:
        return 0.0, None, None
    boundary = np.logical_xor(mask > 0, ndimage.binary_erosion(mask > 0))
    points = np.transpose(np.where(boundary))
    if len(points) < 2:
        return 0.0, None, None
    first_indexes, second_indexes = np.triu_indices(len(points), 1)
    vectors = points[second_indexes] - points[first_indexes]
    lengths = np.linalg.norm(vectors, axis=1)
    reference = np.asarray(line_end) - np.asarray(line_start)
    angles = np.abs(vectors @ reference) / (lengths * np.linalg.norm(reference) + 1e-7)
    for threshold in (0.1, 0.2, 0.5):
        eligible = np.flatnonzero(angles < threshold)
        if len(eligible):
            best = eligible[int(np.argmax(lengths[eligible]))]
            if threshold != 0.1 or lengths[best] > 5:
                return lengths[best], points[first_indexes[best]], points[second_indexes[best]]
    print("WARNING: did not find any perpendicular line. Returning None")
    return 0.0, None, None


def get_plane_diameters(mask, spacing=None):
    """Return major and perpendicular diameters with robust empty handling."""
    region = next(get_region_diameters(mask), None)
    if region is None:
        return 0.0, 0.0, None, None
    _, start, end = region
    _, start_pd, end_pd = max_perpendicular_line(mask, start, end)
    if spacing is None:
        distance = lambda first, second: float(np.linalg.norm(first - second))
    else:
        distance = lambda first, second: float(get_distance(first, second, spacing))
    diameter = distance(start, end)
    perpendicular = (
        distance(start_pd, end_pd) if start_pd is not None and end_pd is not None else 0.0
    )
    return diameter, perpendicular, (start, end), (
        (start_pd, end_pd) if start_pd is not None else None
    )


def get_region_diameters_pd(mask, spacing, affine, z=1):
    if mask.sum() < 5:
        return np.array(0.0), np.array(0.0), None, None
    diameter_mm, diameter_pd_mm, major, perpendicular = get_plane_diameters(mask, spacing)
    if major is None or perpendicular is None:
        return np.array(0.0), np.array(0.0), None, None
    start, end = major
    start_pd, end_pd = perpendicular
    endpoints = [
        point.tolist() + [int(z)] for point in (start, end, start_pd, end_pd)
    ]
    endpoints_mm = [
        vox_to_mm_space(point, affine).round(4).tolist() for point in endpoints
    ]
    return (
        np.asarray(diameter_mm),
        np.asarray(diameter_pd_mm),
        endpoints,
        endpoints_mm,
    )


def find_most_distant_points(mask1, mask2):
    points1, points2 = np.argwhere(mask1), np.argwhere(mask2)
    if not len(points1) or not len(points2):
        return None, None
    maximum = 0.0
    result = (None, None)
    for offset in range(0, len(points1), 1024):
        distances = cdist(points1[offset : offset + 1024], points2)
        flat_index = int(np.argmax(distances))
        distance = distances.flat[flat_index]
        if distance > maximum:
            first, second = np.unravel_index(flat_index, distances.shape)
            maximum = distance
            result = points1[offset + first], points2[second]
    return result


def find_center(mask, integer=False):
    points = np.argwhere(mask)
    if not len(points):
        return None
    center = points.mean(axis=0)
    return np.round(center).astype(np.int32) if integer else center


def _normalize(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _rotation_matrix(source, target):
    source, target = _normalize(source), _normalize(target)
    if not np.linalg.norm(source) or not np.linalg.norm(target):
        return np.eye(3)
    dot = np.clip(np.dot(source, target), -1.0, 1.0)
    if np.isclose(dot, 1):
        return np.eye(3)
    if np.isclose(dot, -1):
        axis = np.eye(3)[np.argmin(np.abs(source))]
        axis = _normalize(np.cross(source, axis))
        return 2 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    return np.eye(3) + skew + skew @ skew * (1 - dot) / np.linalg.norm(cross) ** 2


def rotate_affine_of_image_and_translate(
    image, source_vector, target_vector, center, nora_transformation=True, flip_negative=True
):
    if np.dot(source_vector, target_vector) < 0 and flip_negative:
        target_vector = -target_vector
    rotation = np.eye(4)
    rotation[:3, :3] = _rotation_matrix(source_vector, target_vector)
    to_origin, from_origin = np.eye(4), np.eye(4)
    to_origin[:3, 3] -= center
    from_origin[:3, 3] += center
    combined = from_origin @ rotation @ to_origin
    affine = combined @ image.affine if nora_transformation else image.affine @ combined
    return combined, affine


def find_mask_normal(mask):
    points = np.argwhere(mask > 0.5)
    if not points.size:
        raise ValueError("find_mask_normal: mask has no positive voxels")
    if len(points) == 1:
        return np.array([0.0, 0.0, 1.0])
    components = PCA(n_components=min(3, *points.shape)).fit(points - points.mean(axis=0)).components_
    normal = np.asarray(components[-1], dtype=float)
    if len(normal) < 3:
        normal = np.pad(normal, (0, 3 - len(normal)))
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])


def smooth_mask(mask, sigma=1):
    return (ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma) > 0.5).astype(
        np.uint8
    )
