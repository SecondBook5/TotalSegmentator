import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from totalsegmentator.aorta_report.geometry import (
    find_center,
    find_mask_normal,
    get_region_diameters,
    get_region_diameters_pd,
)
from totalsegmentator.aorta_report.geometry_report import (
    get_2d_plane,
    get_plane_aorta_intersection,
    process_plane,
)
from totalsegmentator.aorta_report.centerline import get_distance


def create_landmark_planes(landmarks, pulmonary_artery, spacing, logger):
    """Create measurement planes and return the largest preliminary diameter."""
    for landmark_index, landmark in landmarks.items():
        if landmark["empty"]:
            landmark["diameter_tmp"] = 0
            continue

        landmark_point = tuple(find_center(landmark["data"], True))
        landmark["cl_point"] = landmark_point
        if landmark_index == 1:
            pulmonary_plane = landmark["data"]
        else:
            normal = find_mask_normal(landmark["data"])
            pulmonary_plane = get_plane_aorta_intersection(
                pulmonary_artery, landmark_point, normal
            ).astype(np.uint8)
        landmark["roi"] = pulmonary_plane

        if pulmonary_plane.sum() == 0:
            logger.info(
                f"WARNING: for landmark {landmark_index} no pulmonary artery plane was found!"
            )
            landmark["empty"] = True
            landmark["diameter_tmp"] = 0
            continue

        if landmark_index > 1 and not landmarks[landmark_index - 1]["empty"]:
            previous_plane = landmarks[landmark_index - 1]["roi"]
            if np.logical_and(pulmonary_plane, previous_plane).any():
                logger.info(
                    f"WARNING: plane of landmark {landmark_index} touches previous landmark!"
                )

        region = next(get_region_diameters(pulmonary_plane), None)
        if region is None:
            logger.info(
                f"WARNING: for landmark {landmark_index} no diameter was found!"
            )
            landmark["diameter_tmp"] = 0
        else:
            _, start, end = region
            landmark["diameter_tmp"] = get_distance(start, end, spacing)
    return max(landmark["diameter_tmp"] for landmark in landmarks.values())


def measure_landmarks(landmarks, ct_img, affine, max_diameter, tmp_dir, logger, debug):
    for landmark_index, landmark in landmarks.items():
        logger.info(f"  Processing landmark {landmark_index}")
        if landmark["empty"]:
            for metric in (
                "area",
                "diameter",
                "diameter_pd",
                "start",
                "end",
                "start_pd",
                "end_pd",
            ):
                landmark[metric] = None
            continue

        plane_img = nib.Nifti1Image(landmark["roi"], affine)
        plane_2d, ct_2d, _, _, inverse_affine = get_2d_plane(
            plane_img,
            ct_img,
            None,
            None,
            order=3,
            dtype=np.int32,
            addon=[max_diameter * 1.4] * 3,
        )
        plane_path = tmp_dir / f"plane_{landmark_index}.png"
        process_plane(ct_2d, plane_2d, None, plane_path, smooth=20)
        landmark["plane_img"] = str(plane_path)

        plane_data = plane_2d.get_fdata().astype(np.uint8)
        center = find_center(plane_data, True)
        z_index = center[2]
        plane_data = binary_fill_holes(plane_data[:, :, z_index]).astype(np.uint8)
        slice_spacing = np.asarray(plane_2d.header.get_zooms())[:2]
        landmark["area"] = (
            plane_data.sum() * np.prod(slice_spacing) / 100
        ).round(1)

        diameter, perpendicular, endpoints, endpoints_mm = get_region_diameters_pd(
            plane_data, slice_spacing, inverse_affine, z=z_index
        )
        landmark["diameter"] = (diameter / 10).round(1)
        landmark["diameter_pd"] = (perpendicular / 10).round(1)
        (
            landmark["start"],
            landmark["end"],
            landmark["start_pd"],
            landmark["end_pd"],
        ) = endpoints_mm if endpoints_mm is not None else [None] * 4
        (
            landmark["start_voxel"],
            landmark["end_voxel"],
            landmark["start_pd_voxel"],
            landmark["end_pd_voxel"],
        ) = endpoints if endpoints is not None else [None] * 4

        if debug:
            nib.save(
                nib.Nifti1Image(landmark["roi"].astype(np.uint8), affine),
                tmp_dir / f"pulmonary_plane_lm{landmark_index}.nii.gz",
            )
