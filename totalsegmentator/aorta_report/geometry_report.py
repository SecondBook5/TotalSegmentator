from __future__ import annotations

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.ndimage import binary_dilation

from totalsegmentator.aorta_report.centerline import (
    get_index_for_point,
    get_mid_of_points,
    get_normal_for_cl_point,
)
from totalsegmentator.aorta_report.geometry import (
    cleanup_plane,
    draw_plane,
    find_center,
    find_mask_normal,
    get_plane_diameters,
    rotate_affine_of_image_and_translate,
    vox_to_mm_space,
    vox_to_mm_space_without_offset,
    vox_to_other_vox_space,
)
from totalsegmentator.aorta_report.utils import crop_to_masks, crop_to_point


def keep_closest_blob(data, reference_point, debug=False):
    blob_map, count = ndimage.label(data, structure=np.ones((3, 3, 3)))
    distances = {}
    for blob_idx in range(1, count + 1):
        coordinates = np.asarray(np.where(blob_map == blob_idx)).T
        distances[blob_idx] = np.linalg.norm(
            coordinates - reference_point, axis=1
        ).mean()
    if debug:
        print("Blob distances:")
        print(distances)
    if not distances:
        return data
    closest = min(distances, key=distances.get)
    return (blob_map == closest).astype(np.uint8)


def get_plane_aorta_intersection(aorta, plane_point, normal_vec):
    plane = draw_plane(aorta.shape, normal_vec, plane_point)
    intersection = binary_dilation((plane + aorta) > 1, iterations=1)
    return keep_closest_blob(intersection, plane_point)


def get_plane_aorta_intersection_crop(aorta_img, plane_point, normal_vec, crop_size=30):
    cropped, intersection = _get_cropped_plane_intersection(
        aorta_img, plane_point, normal_vec, crop_size
    )
    return resample_from_to(
        nib.Nifti1Image(intersection, cropped.affine),
        aorta_img,
        order=0,
    )


def _get_cropped_plane_intersection(aorta_img, plane_point, normal_vec, crop_size):
    cropped = crop_to_point(
        aorta_img, plane_point, [crop_size] * 3, dtype=np.uint8
    )
    cropped_point = vox_to_other_vox_space(
        plane_point, aorta_img.affine, cropped.affine
    )
    intersection = get_plane_aorta_intersection(
        cropped.get_fdata().astype(np.uint8), cropped_point, normal_vec
    )
    return cropped, intersection.astype(np.uint8)


def _diameter_from_centerline_index(
    aorta, centerline, index, spacing, crop_size=30
):
    if index is None or not 0 <= index < len(centerline):
        return np.float32(0)
    if not isinstance(aorta, nib.spatialimages.SpatialImage):
        plane = get_plane_aorta_intersection(
            aorta,
            centerline[index].point,
            get_normal_for_cl_point(centerline, index),
        )
        diameter, _, _, _ = get_plane_diameters(plane, spacing)
        return np.float32(diameter)
    _, plane = _get_cropped_plane_intersection(
        aorta,
        centerline[index].point,
        get_normal_for_cl_point(centerline, index),
        crop_size,
    )
    diameter, _, _, _ = get_plane_diameters(plane, spacing)
    return np.float32(diameter)


def get_max_diameter_of_centerline_section(
    aorta_img,
    centerline,
    spacing,
    return_diameter=False,
    subsample=2,
    crop_size=30,
):
    if subsample < 1:
        raise ValueError("subsample must be at least 1")
    if len(centerline) < 2:
        print(f"WARNING: centerline section too short (len: {len(centerline)})")
        return (None, np.float32(0)) if return_diameter else None
    areas = {}
    for idx in range(0, len(centerline), subsample):
        vertex = centerline[idx]
        normal = get_normal_for_cl_point(centerline, idx)
        _, intersection = _get_cropped_plane_intersection(
            aorta_img, vertex.point, normal, crop_size
        )
        areas[tuple(vertex.point)] = intersection.sum()
    max_point = max(areas, key=areas.get)
    if not return_diameter:
        return max_point
    index = get_index_for_point(centerline, max_point)
    return max_point, _diameter_from_centerline_index(
        aorta_img, centerline, index, spacing, crop_size
    )


def get_aorta_section(aorta_img, start, end, centerline, crop_size=30):
    aorta = aorta_img.get_fdata().astype(np.uint8)
    if (
        len(centerline) < 2
        or start is None
        or end is None
        or not 0 <= start < len(centerline)
        or not 0 <= end < len(centerline)
    ):
        print("WARNING: centerline section too short to calculate aorta section")
        return nib.Nifti1Image(np.zeros_like(aorta), aorta_img.affine)
    if end < start:
        start, end = end, start
    start_crop, start_intersection = _get_cropped_plane_intersection(
        aorta_img,
        centerline[start].point,
        get_normal_for_cl_point(centerline, start),
        crop_size,
    )
    end_crop, end_intersection = _get_cropped_plane_intersection(
        aorta_img,
        centerline[end].point,
        get_normal_for_cl_point(centerline, end),
        crop_size,
    )

    def clear_crop(cropped, intersection):
        origin = np.rint(
            vox_to_other_vox_space((0, 0, 0), cropped.affine, aorta_img.affine)
        ).astype(int)
        slices = tuple(
            slice(origin[axis], origin[axis] + intersection.shape[axis])
            for axis in range(3)
        )
        aorta[slices][intersection > 0] = 0

    clear_crop(start_crop, start_intersection)
    clear_crop(end_crop, end_intersection)
    midpoint_idx = get_mid_of_points(centerline, start, end)
    midpoint = centerline[midpoint_idx].point
    section = keep_closest_blob(aorta, midpoint)
    return nib.Nifti1Image(section, aorta_img.affine)


def get_2d_plane(mask, ct_img, mask_2, mask_3, order=0, dtype=np.uint8, addon=(0, 0, 0)):
    mask_cropped = crop_to_masks(mask, [mask], addon, dtype=np.uint8, fixed_size=True)
    ct_cropped = crop_to_masks(ct_img, [mask], addon, dtype=dtype, fixed_size=True)
    mask_data = mask_cropped.get_fdata()
    mask_2_cropped = (
        crop_to_masks(mask_2, [mask], addon, dtype=np.uint8, fixed_size=True)
        if mask_2 is not None
        else None
    )
    mask_3_cropped = (
        crop_to_masks(mask_3, [mask], addon, dtype=np.uint8, fixed_size=True)
        if mask_3 is not None
        else None
    )
    center_vox = find_center(mask_data)
    if center_vox is None:
        center_vox = (np.asarray(mask_data.shape) - 1) / 2
        normal_vox = np.array([0.0, 0.0, 1.0])
    else:
        normal_vox = find_mask_normal(mask_data)
    center = vox_to_mm_space(center_vox, mask_cropped.affine)
    normal = vox_to_mm_space_without_offset(normal_vox, mask_cropped.affine)
    target = vox_to_mm_space_without_offset(
        np.array([0, 0, 1]), mask_cropped.affine
    )
    rotation, affine = rotate_affine_of_image_and_translate(
        mask_cropped, normal, target, center, nora_transformation=True
    )
    inverse_affine = np.linalg.inv(rotation) @ mask_cropped.affine

    def rotate(image, interpolation):
        if image is None:
            return None
        return resample_from_to(
            nib.Nifti1Image(image.get_fdata(), affine),
            mask_cropped,
            order=interpolation,
        )

    return (
        rotate(mask_cropped, 0),
        rotate(ct_cropped, order),
        rotate(mask_2_cropped, 0),
        rotate(mask_3_cropped, 0),
        inverse_affine,
    )


def process_plane(ct_img, mask_img, mask_img_2, output_path, smooth=20):
    from totalsegmentator.aorta_report.plotting import plot_img

    image = ct_img.get_fdata()
    mask = cleanup_plane(mask_img.get_fdata())
    mask_2 = (
        cleanup_plane(mask_img_2.get_fdata())
        if mask_img_2 is not None
        else None
    )
    center = find_center(mask, True)
    z = center[2] if center is not None else mask.shape[2] // 2
    image, mask = image[:, :, z], mask[:, :, z]
    if mask_2 is not None:
        mask_2 = mask_2[:, :, z]
    mask = cleanup_plane(mask)
    _, _, major, perpendicular = get_plane_diameters(mask)
    start, end = major if major is not None else (None, None)
    start_pd, end_pd = perpendicular if perpendicular is not None else (None, None)
    diameter_points = (
        [start, end, start_pd, end_pd] if major is not None else None
    )
    figure = plot_img(
        image,
        mask,
        mask_2,
        diameter_points,
        vmin=-700,
        vmax=1000,
        smooth=smooth,
    )
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    from matplotlib import pyplot as plt

    plt.close(figure)
