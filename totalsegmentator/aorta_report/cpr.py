import nibabel as nib
import numpy as np

from scipy.ndimage import gaussian_filter1d, map_coordinates, spline_filter
from tqdm import tqdm

from totalsegmentator.aorta_report.centerline import resample_points_by_arc_length


def _normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return None
    return vector / norm


def _resample_centerline(points_vox, vox_to_mm_affine, step_mm):
    """Sample a voxel-space polyline at approximately uniform mm intervals."""
    points_vox = np.asarray(points_vox, dtype=float)
    if len(points_vox) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float), 0.0
    if len(points_vox) == 1:
        return points_vox, np.array([0.0], dtype=float), 0.0

    points_mm = nib.affines.apply_affine(vox_to_mm_affine, points_vox)
    segment_lengths = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative_length[-1]

    if total_length < 1e-8:
        return points_vox, np.array([0.0], dtype=float), float(total_length)

    nr_points = max(2, int(np.ceil(total_length / step_mm)) + 1)
    sample_positions = np.linspace(0.0, total_length, nr_points)
    resampled_mm = resample_points_by_arc_length(points_mm, sample_positions)
    resampled_vox = nib.affines.apply_affine(
        np.linalg.inv(vox_to_mm_affine), resampled_mm
    )
    return resampled_vox, sample_positions, float(total_length)


def _axis_angle_to_matrix(axis, angle):
    axis = _normalize_vector(axis)
    if axis is None or abs(angle) < 1e-8:
        return np.eye(3)
    x, y, z = axis
    skew = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _rotation_between(vector_a, vector_b, fallback_axis=None):
    vector_a = _normalize_vector(vector_a)
    vector_b = _normalize_vector(vector_b)
    if vector_a is None or vector_b is None:
        return np.eye(3)

    cross = np.cross(vector_a, vector_b)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(vector_a, vector_b), -1.0, 1.0)
    if cross_norm >= 1e-8:
        return _axis_angle_to_matrix(cross / cross_norm, np.arctan2(cross_norm, dot))
    if dot > 0.999999:
        return np.eye(3)

    axis = fallback_axis
    if axis is None:
        axis = np.array([1.0, 0.0, 0.0])
    axis = _normalize_vector(axis - np.dot(axis, vector_a) * vector_a)
    if axis is None:
        for candidate in np.eye(3):
            axis = _normalize_vector(candidate - np.dot(candidate, vector_a) * vector_a)
            if axis is not None:
                break
    return _axis_angle_to_matrix(axis, np.pi)


def _pick_reference_axis(tangent):
    reference_axes = np.eye(3)
    return reference_axes[np.argmin(np.abs(reference_axes @ tangent))]


def _centerline_tangents(points_mm):
    tangents = np.zeros_like(points_mm)
    tangents[0] = points_mm[1] - points_mm[0]
    tangents[-1] = points_mm[-1] - points_mm[-2]
    if len(points_mm) > 2:
        tangents[1:-1] = points_mm[2:] - points_mm[:-2]

    for index, raw_tangent in enumerate(tangents):
        tangent = _normalize_vector(raw_tangent)
        if tangent is not None:
            tangents[index] = tangent
        elif index > 0:
            tangents[index] = tangents[index - 1]
        else:
            for candidate in tangents[1:]:
                tangent = _normalize_vector(candidate)
                if tangent is not None:
                    tangents[index] = tangent
                    break
            else:
                raise ValueError("Unable to determine tangent directions for CPR.")
    return tangents


def _parallel_transport_frames(tangents):
    """Return stable normal/binormal vectors along unit tangents."""
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)
    reference = _pick_reference_axis(tangents[0])
    normal = _normalize_vector(reference - np.dot(reference, tangents[0]) * tangents[0])
    if normal is None:
        raise ValueError("Unable to initialize CPR frame.")
    binormal = _normalize_vector(np.cross(tangents[0], normal))
    normals[0] = _normalize_vector(np.cross(binormal, tangents[0]))
    binormals[0] = binormal

    for index in range(1, len(tangents)):
        rotation = _rotation_between(
            tangents[index - 1], tangents[index], normals[index - 1]
        )
        normal = rotation @ normals[index - 1]
        normal = _normalize_vector(normal - np.dot(normal, tangents[index]) * tangents[index])
        if normal is None:
            reference = _pick_reference_axis(tangents[index])
            normal = _normalize_vector(
                reference - np.dot(reference, tangents[index]) * tangents[index]
            )
        if normal is None:
            normal = normals[index - 1]

        binormal = _normalize_vector(np.cross(tangents[index], normal))
        if binormal is None:
            binormal = binormals[index - 1]
        normal = _normalize_vector(np.cross(binormal, tangents[index]))
        if np.dot(normal, normals[index - 1]) < 0:
            normal *= -1.0
            binormal *= -1.0
        normals[index] = normal
        binormals[index] = binormal
    return normals, binormals


def _build_sampling_grid(max_diameter_mm, spacing_mm):
    in_plane_spacing_mm = float(np.clip(spacing_mm, 0.6, 1.0))
    field_of_view_mm = max(float(max_diameter_mm) * 1.6, 48.0)
    plane_size = max(int(np.ceil(field_of_view_mm / in_plane_spacing_mm)), 32)
    if plane_size % 2 == 0:
        plane_size += 1
    axis_mm = (
        np.arange(plane_size) - (plane_size - 1) / 2.0
    ) * in_plane_spacing_mm
    grid_u, grid_v = np.meshgrid(axis_mm, axis_mm, indexing="ij")
    return grid_u.astype(np.float32), grid_v.astype(np.float32), in_plane_spacing_mm


def _chunk_ranges(nr_slices, plane_size, target_points=1_000_000):
    points_per_slice = plane_size * plane_size
    chunk_size = max(1, min(nr_slices, target_points // points_per_slice))
    return range(0, nr_slices, chunk_size), chunk_size


def _sampling_plane_chunk(centers_mm, normals, binormals, grid_u, grid_v):
    """Construct flattened world-space sampling coordinates for a slice chunk."""
    plane_mm = (
        centers_mm[:, None, None, :]
        + grid_u[None, :, :, None] * normals[:, None, None, :]
        + grid_v[None, :, :, None] * binormals[:, None, None, :]
    )
    return plane_mm.reshape(-1, 3)


def cpr(ct_img, mask_img, cl, max_dia, fast=False, debug=False, extra_mask_imgs=None, return_info=False):
    """
    Create curved planar reformation (cpr) from ct image and binary mask.

    ct_img: nifti image
    mask_img: nifti image
    cl: centerline
    max_dia: maximum diameter of aorta

    extra_mask_imgs: optional additional binary nifti masks to sample on the
                     same CPR grid.

    returns: 3d nifti images: ct_img_cpr, mask_img_cpr
    """
    if len(cl) == 0:
        raise ValueError("Centerline is empty.")

    if extra_mask_imgs is None:
        extra_mask_imgs = []

    ct_spacing = np.array(ct_img.header.get_zooms()[:3], dtype=float)
    mask_spacing = np.array(mask_img.header.get_zooms()[:3], dtype=float)
    centerline_step_mm = float(np.min(mask_spacing))
    if fast:
        centerline_step_mm *= 2.0

    cl_vox = np.array([vertex.point for vertex in cl], dtype=float)
    cl_vox, sample_positions_mm, total_length_mm = _resample_centerline(cl_vox, mask_img.affine, centerline_step_mm)
    if cl_vox.shape[0] < 2:
        raise ValueError("Centerline needs at least two points for CPR.")

    cl_mm = nib.affines.apply_affine(mask_img.affine, cl_vox)
    if len(cl_mm) >= 5:
        # Smooth small voxel-level centerline jitter so the straightened vessel does
        # not wobble from slice to slice.
        cl_mm = gaussian_filter1d(cl_mm, sigma=2.0, axis=0, mode="nearest")
        cl_mm[0] = nib.affines.apply_affine(mask_img.affine, cl_vox[0])
        cl_mm[-1] = nib.affines.apply_affine(mask_img.affine, cl_vox[-1])

    tangents = _centerline_tangents(cl_mm)
    normals, binormals = _parallel_transport_frames(tangents)
    grid_u, grid_v, in_plane_spacing_mm = _build_sampling_grid(
        max_dia, np.min(ct_spacing)
    )
    plane_size = grid_u.shape[0]

    ct_data = np.asarray(ct_img.get_fdata(), dtype=np.float32)
    cpr_mask_imgs = [mask_img, *extra_mask_imgs]
    mask_datas = [np.asarray(curr_mask.get_fdata() > 0.5, dtype=np.float32) for curr_mask in cpr_mask_imgs]
    inv_ct_affine = np.linalg.inv(ct_img.affine)
    inv_mask_affines = [np.linalg.inv(curr_mask.affine) for curr_mask in cpr_mask_imgs]
    ct_background = float(np.min(ct_data))
    ct_interpolation_order = 3
    mask_interpolation_order = 1

    if ct_interpolation_order > 1:
        # `map_coordinates` otherwise recomputes the spline prefilter on the whole
        # CT volume for every sampling call, which dominates runtime.
        ct_data = spline_filter(ct_data, order=ct_interpolation_order, output=np.float32)
        ct_prefilter = False
    else:
        ct_prefilter = True

    nr_slices = len(cl_mm)
    res_ct = np.empty((nr_slices, plane_size, plane_size), dtype=np.float32)
    res_masks = [np.empty((nr_slices, plane_size, plane_size), dtype=np.uint8) for _ in cpr_mask_imgs]

    chunk_ranges, chunk_size = _chunk_ranges(nr_slices, plane_size)

    ct_linear = inv_ct_affine[:3, :3].astype(np.float32)
    ct_offset = inv_ct_affine[:3, 3].astype(np.float32)
    mask_linears = [inv_mask_affine[:3, :3].astype(np.float32) for inv_mask_affine in inv_mask_affines]
    mask_offsets = [inv_mask_affine[:3, 3].astype(np.float32) for inv_mask_affine in inv_mask_affines]
    shared_grids = [np.allclose(inv_ct_affine, inv_mask_affine) for inv_mask_affine in inv_mask_affines]

    iterator = chunk_ranges
    if debug:
        iterator = tqdm(iterator, total=len(chunk_ranges), desc="CPR chunks")

    for start_idx in iterator:
        end_idx = min(start_idx + chunk_size, nr_slices)
        centers_mm = cl_mm[start_idx:end_idx].astype(np.float32, copy=False)
        chunk_normals = normals[start_idx:end_idx].astype(np.float32, copy=False)
        chunk_binormals = binormals[start_idx:end_idx].astype(np.float32, copy=False)

        flat_mm = _sampling_plane_chunk(
            centers_mm, chunk_normals, chunk_binormals, grid_u, grid_v
        )

        coords_ct = (flat_mm @ ct_linear.T + ct_offset).T

        ct_chunk = map_coordinates(
            ct_data,
            coords_ct,
            order=ct_interpolation_order,
            mode="constant",
            cval=ct_background,
            prefilter=ct_prefilter,
        ).reshape(end_idx - start_idx, plane_size, plane_size)

        res_ct[start_idx:end_idx] = ct_chunk
        for mask_idx, mask_data in enumerate(mask_datas):
            if shared_grids[mask_idx]:
                coords_mask = coords_ct
            else:
                coords_mask = (flat_mm @ mask_linears[mask_idx].T + mask_offsets[mask_idx]).T

            seg_chunk = map_coordinates(
                mask_data,
                coords_mask,
                order=mask_interpolation_order,
                mode="constant",
                cval=0.0,
            ).reshape(end_idx - start_idx, plane_size, plane_size)
            res_masks[mask_idx][start_idx:end_idx] = seg_chunk > 0.5

    if np.issubdtype(ct_img.get_data_dtype(), np.integer):
        ct_dtype = np.dtype(ct_img.get_data_dtype())
        ct_info = np.iinfo(ct_dtype)
        res_ct = np.clip(np.rint(res_ct), ct_info.min, ct_info.max).astype(ct_dtype)
    else:
        res_ct = res_ct.astype(ct_img.get_data_dtype())

    if debug:
        print(f"cpr centerline points: {len(cl_mm)}")
        print(f"cpr plane_size: {plane_size}")
        print(f"cpr chunk_size: {chunk_size}")
        print(f"cpr output shape: {res_ct.transpose(1, 2, 0).shape}")

    output_affine = np.diag([in_plane_spacing_mm, in_plane_spacing_mm, centerline_step_mm, 1.0])
    ct_img_cpr = nib.Nifti1Image(res_ct.transpose(1, 2, 0), output_affine)
    mask_imgs_cpr = [
        nib.Nifti1Image(res_mask.transpose(1, 2, 0), output_affine)
        for res_mask in res_masks
    ]
    cpr_info = {
        "centerline_step_mm": centerline_step_mm,
        "sample_positions_mm": sample_positions_mm,
        "total_length_mm": total_length_mm,
        "nr_slices": nr_slices,
    }

    if extra_mask_imgs and return_info:
        return ct_img_cpr, mask_imgs_cpr[0], mask_imgs_cpr[1:], cpr_info
    if extra_mask_imgs:
        return ct_img_cpr, mask_imgs_cpr[0], mask_imgs_cpr[1:]
    if return_info:
        return ct_img_cpr, mask_imgs_cpr[0], cpr_info
    return ct_img_cpr, mask_imgs_cpr[0]
