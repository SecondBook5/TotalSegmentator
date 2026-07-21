import asyncio
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.aorta_report.utils import crop_to_masks
from totalsegmentator.nifti_ext_header import load_multilabel_nifti


MODAL_APP_NAME = "totalsegmentator"
MODAL_SEGMENTATION_FUNCTION = "run_totalsegmentator"
MODAL_NNUNET_FUNCTION = "run_nnunet_v2"

MODEL_RUNS = (
    (
        "Running TotalSegmentator - pulmonary_artery",
        "  Skipping TotalSeg (already exists)",
        ("pulmonary_artery.nii.gz",),
        ("-ta", "heartchambers_highres"),
    ),
    (
        "Running T514 annulus pulmonary",
        "  Skipping T514 (already exists)",
        (
            "pul_annulus.nii.gz",
            "pul_sinotubular_junction.nii.gz",
            "pul_bifurcation.nii.gz",
            "pul_left_start.nii.gz",
            "pul_right_start.nii.gz",
        ),
        ("-ta", "pulmonary_artery_landmarks"),
    ),
)


def _command(ct_path, output_dir, task_args):
    return [
        "TotalSegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(output_dir),
        "-ns",
        "1",
        *task_args,
    ]


def _outputs_exist(output_dir, names):
    return all((output_dir / name).exists() for name in names)


def _check_outputs(output_dir, names, description):
    missing = [str(output_dir / name) for name in names if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{description} did not create the expected output files:\n  - "
            + "\n  - ".join(missing)
        )


def _pending_model_runs(output_dir, logger):
    pending = []
    for status, skip_message, outputs, task_args in MODEL_RUNS:
        if _outputs_exist(output_dir, outputs):
            logger.info(skip_message)
        else:
            pending.append((status, outputs, task_args))
    return pending


def _run(command, description):
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


async def _run_async(command, description):
    process = await asyncio.create_subprocess_exec(*command)
    return_code = await process.wait()
    if return_code:
        raise RuntimeError(f"{description} failed with exit code {return_code}")


def _validate_host(host):
    if host not in {"local", "modal"}:
        raise ValueError(f"Unsupported host '{host}'. Expected 'local' or 'modal'.")


def _get_modal_function(function_name):
    try:
        import modal
    except ImportError as exc:
        raise ImportError(
            "Modal execution requires the optional 'modal' package."
        ) from exc
    return modal.Function.from_name(MODAL_APP_NAME, function_name)


def _load_for_remote(ct_path):
    image = nib.load(ct_path)
    return nib.Nifti1Image(
        np.asanyarray(image.dataobj), image.affine, image.header.copy()
    )


def _modal_request(task_args):
    task_index = task_args.index("-ta")
    task = task_args[task_index + 1]
    if task == "pulmonary_artery_landmarks":
        return MODAL_NNUNET_FUNCTION, {
            "task_id": 514,
            "model": "3d_fullres",
            "tr": "nnUNetTrainer",
            "folds": "-1",
        }
    options = {"task": task, "ml": True, "nr_thr_saving": 1}
    if "-rs" in task_args:
        roi_index = task_args.index("-rs")
        options["roi_subset"] = [
            value for value in task_args[roi_index + 1 :] if not value.startswith("-")
        ]
    if "--fast" in task_args:
        options["fast"] = True
    return MODAL_SEGMENTATION_FUNCTION, options


def _save_modal_masks(image, output_dir):
    image, label_map = load_multilabel_nifti(image)
    data = np.asanyarray(image.dataobj)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label_id, class_name in label_map.items():
        mask = (data == label_id).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(mask, image.affine, image.header),
            output_dir / f"{class_name}.nii.gz",
        )


def _run_modal(image, function_name, options):
    return _get_modal_function(function_name).remote(image, options)


async def _run_modal_async(image, function_name, options):
    return await _get_modal_function(function_name).remote.aio(image, options)


def get_pulmonary_fast(ct_path, output_dir, logger, host="local"):
    _validate_host(host)
    output_dir = Path(output_dir)
    if (output_dir / "heart.nii.gz").exists():
        logger.info("  Skipping TotalSeg pulmonary fast (already exists)")
        return
    task_args = ("-ta", "total", "-rs", "heart", "--fast")
    if host == "local":
        _run(
            _command(Path(ct_path), output_dir, task_args),
            "TotalSegmentator pulmonary fast",
        )
    else:
        function_name, options = _modal_request(task_args)
        image = _run_modal(_load_for_remote(ct_path), function_name, options)
        _save_modal_masks(image, output_dir)
    _check_outputs(output_dir, ("heart.nii.gz",), "TotalSegmentator pulmonary fast")


def crop_to_pulmonary(ct_path, output_dir, logger):
    output_dir = Path(output_dir)
    output_path = output_dir / "ct_cropped.nii.gz"
    if output_path.exists():
        logger.info("  already cropped (skipping)")
        return output_path
    ct_img = nib.load(ct_path)
    print("Cropping...")
    print(f"  shape Before: {ct_img.shape}")
    ct_img = crop_to_masks(
        ct_img,
        [nib.load(output_dir / "heart.nii.gz")],
        [20, 20, 20],
        dtype=np.int16,
    )
    print(f"  shape After: {ct_img.shape}")
    nib.save(ct_img, output_path)
    return output_path


def run_models_consecutive(ct_path, output_dir, logger, host="local"):
    _validate_host(host)
    started = time.time()
    ct_path, output_dir = Path(ct_path), Path(output_dir)
    pending = _pending_model_runs(output_dir, logger)
    pending_by_status = {run[0]: run for run in pending}
    remote_image = _load_for_remote(ct_path) if host == "modal" and pending else None
    for status, _, _, _ in MODEL_RUNS:
        yield status
        model_run = pending_by_status.get(status)
        if model_run is None:
            continue
        _, outputs, task_args = model_run
        if host == "local":
            _run(_command(ct_path, output_dir, task_args), status)
        else:
            function_name, options = _modal_request(task_args)
            image = _run_modal(remote_image, function_name, options)
            _save_modal_masks(image, output_dir)
        _check_outputs(output_dir, outputs, status)
    print(f"Models done (took: {time.time() - started:.2f}s)")
    yield "Models done"


async def run_models_parallel(ct_path, output_dir, logger, host="local"):
    _validate_host(host)
    started = time.time()
    ct_path, output_dir = Path(ct_path), Path(output_dir)
    pending = _pending_model_runs(output_dir, logger)
    if host == "local":
        await asyncio.gather(
            *(
                _run_async(_command(ct_path, output_dir, task_args), status)
                for status, _, task_args in pending
            )
        )
    elif pending:
        remote_image = _load_for_remote(ct_path)
        requests = [_modal_request(task_args) for _, _, task_args in pending]
        images = await asyncio.gather(
            *(
                _run_modal_async(remote_image, function_name, options)
                for function_name, options in requests
            )
        )
        for image in images:
            _save_modal_masks(image, output_dir)
    for status, outputs, _ in pending:
        _check_outputs(output_dir, outputs, status)
    print(f"Models ASYNC done (took: {time.time() - started:.2f}s)")
