import shutil
from importlib import resources

import nibabel as nib
import numpy as np

from totalsegmentator.aorta_report.nifti import combine_as_nifti
from totalsegmentator.pulmonary_report.constants import REPORT_FRAMES
from totalsegmentator.reporting import generate_html
from totalsegmentator.serialization_utils import serialize_and_compress


def render_report_image(landmarks, metadata, tmp_dir):
    assets_dir = resources.files("totalsegmentator").joinpath(
        "resources/pulmonary_report"
    )
    slices = []
    for frame_index in range(REPORT_FRAMES):
        output_path = tmp_dir / f"pulmonary_report_frontpage_{frame_index:02d}.png"
        generate_html(
            str(assets_dir),
            "report_template_frontpage.html",
            {
                "landmarks": landmarks,
                "preview_3d": str(
                    tmp_dir / f"preview_3d_rotating_{frame_index:06d}.png"
                ),
                "metadata": metadata,
            },
            output_path,
            width=900,
            png_logo_hosts=("rigi", "rndapollolp01.uhbs.ch"),
        )
        slices.append(output_path)
    return combine_as_nifti(slices)


def prepare_results(landmarks, metadata):
    landmarks = dict(sorted(landmarks.items()))
    for landmark in landmarks.values():
        if not landmark["empty"]:
            for key in ("roi", "plane_img", "cl_point"):
                landmark.pop(key, None)
        for key in ("data", "diameter_tmp", "color", "txt_offset", "empty"):
            landmark.pop(key, None)
        landmark["name"] = landmark.pop("display_name")
    return {"landmarks": landmarks, "metadata": metadata}


def serialize_outputs(report_image, results, tmp_dir, logger):
    masks = {}
    mask_path = tmp_dir / "pul_annulus.nii.gz"
    if mask_path.exists():
        image = nib.load(mask_path)
        masks["pul_annulus"] = nib.Nifti1Image(
            image.get_fdata(dtype=np.float32).astype(np.uint8),
            image.affine,
            image.header,
        )
    else:
        logger.info(
            "WARNING: Mask pul_annulus not found for final output. Skipping."
        )
    return serialize_and_compress(report_image), serialize_and_compress(masks)


def cleanup_workdir(tmp_dir, delete_tmp, delete_aux_files):
    if delete_aux_files and tmp_dir.exists():
        for suffix in ("*.png", "*.html"):
            for path in tmp_dir.glob(suffix):
                path.unlink()
    if delete_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
