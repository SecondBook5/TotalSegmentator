import argparse
import json
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.pulmonary_report import VERSION
from totalsegmentator.pulmonary_report.report import create_pulmonary_report
from totalsegmentator.reporting import save_runtime, setup_logger
from totalsegmentator.serialization_utils import (
    convert_to_serializable,
    decompress_and_deserialize,
)


def _load_metadata(path, logger):
    if path is not None and path.exists():
        with path.open() as file:
            metadata = json.load(file, strict=False)
        for source, target in (
            ("PatientsAge", "PatientAge"),
            ("PatientsSex", "PatientSex"),
            ("PatientsBirthDate", "PatientBirthDate"),
        ):
            if source in metadata:
                metadata[target] = metadata.pop(source)
    else:
        logger.info(
            "WARNING: Could not find nodeinfo.json. Using default metadata."
        )
        metadata = {
            "PatientName": "Unknown",
            "PatientSex": "M",
            "PatientAge": "083Y",
            "PatientBirthDate": "19000101",
            "PatientID": "Unknown",
            "StudyID": "Unknown",
            "StudyDescription": "Unknown",
            "StudyDate": "19000101",
        }
    metadata.setdefault("PatientAge", "099Y")
    if "W" in str(metadata["PatientAge"]):
        logger.info(
            "WARNING: Patient age is in weeks. Patient too young. Stopping."
        )
        return None
    metadata["PatientAge"] = int(str(metadata["PatientAge"]).replace("Y", ""))
    metadata["version"] = VERSION
    return metadata


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a pulmonary artery report."
    )

    def resolved_path(value):
        return Path(value).resolve()

    parser.add_argument(
        "-i", "--ct_path", type=resolved_path, required=True, help="Path to CT file."
    )
    parser.add_argument(
        "-rt",
        "--rois_totalseg",
        type=resolved_path,
        help="Directory containing the TotalSegmentator ROIs.",
    )
    parser.add_argument(
        "-rd",
        "--rois_details",
        type=resolved_path,
        help="Directory containing the pulmonary landmark ROIs.",
    )
    parser.add_argument(
        "-n", "--nodeinfo", type=resolved_path, help="Optional path to nodeinfo file."
    )
    parser.add_argument(
        "-o",
        "--output_nifti",
        type=resolved_path,
        required=True,
        help="Path to output NIfTI file.",
    )
    parser.add_argument(
        "-j",
        "--output_json",
        type=resolved_path,
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "-l",
        "--output_log",
        type=resolved_path,
        required=True,
        help="Path to output log file.",
    )
    parser.add_argument(
        "-tmp",
        "--tmp_dir",
        type=resolved_path,
        help="Temporary directory. Uses a system temporary directory if omitted.",
    )
    parser.add_argument(
        "-a",
        "--pulmonary_fn",
        default="pulmonary_artery.nii.gz",
        help="Filename of the pulmonary artery mask.",
    )
    parser.add_argument(
        "-rm", "--run_models", action="store_true", help="Run all required models."
    )
    parser.add_argument(
        "-mp",
        "--models_parallel",
        action="store_true",
        help="Run models in parallel. Faster, but needs more RAM and GPU memory.",
    )
    parser.add_argument(
        "--host",
        choices=("local", "modal"),
        default="local",
        help="Run segmentation models locally or with the configured Modal app.",
    )
    parser.add_argument(
        "-t", "--test", default="None", help="Define which test mode to run."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Keep temporary and intermediate files.",
    )
    parser.add_argument(
        "-r",
        "--save_runtime",
        action="store_true",
        help="Save runtime to META/runtime.json.",
    )
    parser.add_argument("--version", action="version", version=VERSION)
    return parser


def _prepare_tmp_dir(args):
    if args.debug:
        print("Running in DEBUG mode.")
    tmp_dir = args.tmp_dir if args.tmp_dir is not None else Path(tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir, args.tmp_dir is None and not args.debug, not args.debug


def _validate_ct(ct_path, logger):
    ct_img = nib.load(ct_path)
    shape = ct_img.shape
    if len(shape) != 3:
        logger.info("WARNING: Image shape is not 3D. Stopping.")
        return False
    if max(shape) > 1500:
        logger.info("WARNING: Image shape is very large (>1500). Stopping.")
        return False
    if min(shape) < 20:
        logger.info("WARNING: Image shape is very small (<20). Stopping.")
        return False
    linear_affine = ct_img.affine[:3, :3].copy()
    if linear_affine.min() < 0:
        logger.info("WARNING: Affine contains negative values!")
    np.fill_diagonal(linear_affine, 0)
    if np.abs(linear_affine).max() > 0.01:
        logger.info("WARNING: Affine contains significant rotation!")
    return True


def _save_report(final_result, output_nifti, output_json):
    nib.save(decompress_and_deserialize(final_result["report_img"]), output_nifti)
    with output_json.open("w") as file:
        json.dump(
            convert_to_serializable(final_result["report_json"]),
            file,
            indent=4,
        )


"""
Run for one test case:

cd /mnt/nvme/data/test_data/pulmonary_report/ct15mm
totalseg_pulmonary_report \
  -i ct_15mm.nii.gz \
  -rt tmp_pulmonary_report/models \
  -rd tmp_pulmonary_report/models \
  -n None \
  -o result/pulmonary_report.nii.gz \
  -j result/pulmonary_report.json \
  -l result/pulmonary_report.txt \
  -tmp tmp \
  -a pulmonary_artery.nii.gz \
  -d
"""
def main():
    parser = _build_parser()
    args = parser.parse_args()
    if not args.run_models and (
        args.rois_totalseg is None or args.rois_details is None
    ):
        parser.error("--rois_totalseg and --rois_details are required without --run_models")

    for output_path in (args.output_nifti, args.output_json, args.output_log):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        args.output_log, name=f"totalseg_pulmonary_report.{args.output_log}"
    )
    started = time.time()
    logger.info("Getting metadata...")
    metadata = _load_metadata(args.nodeinfo, logger)
    if metadata is None:
        return
    if metadata["PatientAge"] < 5 or metadata["PatientAge"] > 150:
        logger.info("WARNING: Patient age is not in range [5, 150]. Stopping.")
        return
    if not _validate_ct(args.ct_path, logger):
        return

    tmp_dir, delete_tmp, delete_aux_files = _prepare_tmp_dir(args)
    logger.info("Creating report...")
    generator = create_pulmonary_report(
        args.ct_path,
        args.rois_totalseg,
        args.rois_details,
        metadata,
        tmp_dir,
        logger,
        delete_tmp=delete_tmp,
        delete_aux_files=delete_aux_files,
        pulmonary_fn=args.pulmonary_fn,
        test=args.test,
        debug=args.debug,
        run_models=args.run_models,
        models_parallel=args.models_parallel,
        host=args.host,
        version=VERSION,
    )
    final_result = None
    for result in generator:
        print(f"progress: {result['progress']}, status: {result['status']}")
        if result["progress"] == 100:
            final_result = result
    if final_result is None:
        raise RuntimeError("Pulmonary report generator did not return a final result.")
    _save_report(final_result, args.output_nifti, args.output_json)
    if args.save_runtime and args.nodeinfo is not None and args.nodeinfo.exists():
        save_runtime(started, "pulmonary_report", args.nodeinfo)


if __name__ == "__main__":
    main()
