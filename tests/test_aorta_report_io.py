import asyncio
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from totalsegmentator.aorta_report import model_runner
from totalsegmentator.aorta_report.utils import crop_to_aorta, crop_to_masks, lazy_load
from totalsegmentator.nifti_ext_header import add_label_map_to_nifti


def _image(data, affine=None):
    return nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)


def _save_image(path, data):
    nib.save(_image(data), path)
    return path


def _multilabel_image(class_names):
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    label_map = {}
    for label, class_name in enumerate(class_names, 1):
        data[label % 3, label // 3, 1] = label
        label_map[label] = class_name
    return add_label_map_to_nifti(_image(data), label_map)


def test_crop_to_masks_uses_binary_mask_without_promoting_input(tmp_path):
    ct = _image(np.arange(64, dtype=np.float32).reshape(4, 4, 4))
    mask = _image(np.pad(np.ones((2, 2, 2), dtype=np.uint8), 1))

    cropped = crop_to_masks(ct, [mask], addon=(0, 0, 0), dtype=np.float32)

    assert cropped.shape == (2, 2, 2)
    assert cropped.get_data_dtype() == np.dtype(np.float32)
    with pytest.raises(ValueError, match="at least one mask"):
        crop_to_masks(ct, [])


def test_lazy_load_explicitly_distinguishes_ct_and_mask(tmp_path):
    target = _image(np.zeros((4, 4, 4), dtype=np.float32))
    ct_data = np.zeros((4, 4, 4), dtype=np.float32)
    ct_data[0, 0, 0] = 0.25
    ct_data[3, 3, 3] = 0.75
    ct_path = _save_image(tmp_path / "source_ct.nii.gz", ct_data)
    mask_data = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_data[0, 0, 0] = 1
    mask_data[2:4, 2:4, 2:4] = 1
    mask_path = _save_image(tmp_path / "source_mask.nii.gz", mask_data)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    ct_image, ct = lazy_load(
        ct_path, target, cache_dir, is_mask=False, use_cache=False
    )
    mask_image, mask = lazy_load(
        mask_path, target, cache_dir, is_mask=True, use_cache=False
    )

    assert ct.dtype == np.float32
    assert ct_image.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_array_equal(ct, ct_data)
    assert mask.dtype == np.uint8
    assert mask_image.get_data_dtype() == np.dtype(np.uint8)
    assert mask.sum() == 8
    assert mask[0, 0, 0] == 0


def test_lazy_load_cache_and_optional_missing_mask_are_explicit(tmp_path):
    target = _image(np.zeros((2, 2, 2), dtype=np.float32))
    source = _save_image(
        tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8)
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    lazy_load(source, target, cache_dir, is_mask=True)
    _save_image(source, np.zeros((2, 2, 2), dtype=np.uint8))
    _, cached = lazy_load(source, target, cache_dir, is_mask=True)
    _, refreshed = lazy_load(
        source, target, cache_dir, is_mask=True, use_cache=False
    )

    assert cached.dtype == np.uint8
    assert cached.sum() == 8
    assert refreshed.sum() == 0

    missing = tmp_path / "missing.nii.gz"
    with pytest.raises(FileNotFoundError):
        lazy_load(missing, target, cache_dir, is_mask=True)
    empty_image, empty = lazy_load(
        missing, target, cache_dir, is_mask=True, required=False
    )
    assert empty_image.shape == target.shape
    assert empty.dtype == np.uint8
    assert not empty.any()


def test_lazy_load_does_not_treat_source_as_its_own_cache(tmp_path):
    source_data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    source = _save_image(tmp_path / "ct.nii.gz", source_data)
    target = _image(np.zeros((3, 3, 3), dtype=np.float32))

    image, data = lazy_load(source, target, tmp_path, is_mask=False)

    assert image.shape == target.shape
    assert data.shape == target.shape
    np.testing.assert_array_equal(nib.load(source).get_fdata(), source_data)


def test_crop_to_aorta_keeps_source_masks(tmp_path):
    ct_path = _save_image(
        tmp_path / "ct.nii.gz", np.arange(125, dtype=np.int16).reshape(5, 5, 5)
    )
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    aorta_path = _save_image(tmp_path / "aorta.nii.gz", mask)
    heart_path = _save_image(tmp_path / "heart.nii.gz", mask)
    logger = SimpleNamespace(info=lambda _message: None)

    result = crop_to_aorta(ct_path, tmp_path, logger)

    assert result.exists()
    assert aorta_path.exists()
    assert heart_path.exists()


def _create_outputs(output_dir, description):
    run = next(run for run in model_runner.MODEL_RUNS if run[0] == description)
    for name in run[2]:
        (output_dir / name).touch()


def test_sequential_runner_selects_tasks_once_and_preserves_commands(
    tmp_path, monkeypatch
):
    logger_messages = []
    first_run = model_runner.MODEL_RUNS[0]
    for name in first_run[2]:
        (tmp_path / name).touch()
    commands = []

    def fake_run(command, description):
        commands.append(command)
        _create_outputs(tmp_path, description)

    monkeypatch.setattr(model_runner, "_run", fake_run)
    logger = SimpleNamespace(info=logger_messages.append)

    statuses = list(
        model_runner.run_models_consecutive(Path("ct.nii.gz"), tmp_path, logger)
    )

    assert statuses == [run[0] for run in model_runner.MODEL_RUNS] + ["Models done"]
    assert logger_messages == [first_run[1]]
    assert commands == [
        model_runner._command(Path("ct.nii.gz"), tmp_path, run[3])
        for run in model_runner.MODEL_RUNS[1:]
    ]


def test_parallel_runner_uses_same_task_selection(tmp_path, monkeypatch):
    first_run = model_runner.MODEL_RUNS[0]
    for name in first_run[2]:
        (tmp_path / name).touch()
    descriptions = []

    async def fake_run_async(_command, description):
        descriptions.append(description)
        _create_outputs(tmp_path, description)

    monkeypatch.setattr(model_runner, "_run_async", fake_run_async)
    logger_messages = []
    logger = SimpleNamespace(info=logger_messages.append)

    asyncio.run(
        model_runner.run_models_parallel(Path("ct.nii.gz"), tmp_path, logger)
    )

    assert descriptions == [run[0] for run in model_runner.MODEL_RUNS[1:]]
    assert logger_messages == [first_run[1]]


def test_modal_fast_segmentation_saves_remote_multilabel_masks(
    tmp_path, monkeypatch
):
    ct_path = _save_image(tmp_path / "ct.nii.gz", np.zeros((3, 3, 3)))
    calls = []

    def fake_modal_run(_image_arg, function_name, options):
        calls.append((function_name, options))
        return _multilabel_image(["aorta", "heart"])

    monkeypatch.setattr(model_runner, "_run_modal", fake_modal_run)
    logger = SimpleNamespace(info=lambda _message: None)

    model_runner.get_aorta_fast(
        ct_path, tmp_path / "outputs", logger, host="modal"
    )

    assert calls == [
        (
            model_runner.MODAL_SEGMENTATION_FUNCTION,
            {
                "task": "total",
                "roi_subset": ["aorta", "heart"],
                "ml": True,
                "nr_thr_saving": 1,
                "fast": True,
            },
        )
    ]
    assert (tmp_path / "outputs" / "aorta.nii.gz").exists()
    assert (tmp_path / "outputs" / "heart.nii.gz").exists()


def test_modal_parallel_runner_dispatches_pending_tasks(tmp_path, monkeypatch):
    ct_path = _save_image(tmp_path / "ct.nii.gz", np.zeros((3, 3, 3)))
    task_outputs = {
        "total": model_runner.TOTAL_ROIS,
        "aortic_dissection": [
            "aorta_true_lumen",
            "aorta_false_lumen",
        ],
        "aorta_annulus": ["annulus_proper", "sinotubular_junction"],
        "renal_arteries": [
            "celiac_trunk",
            "superior_mesenteric_artery",
            "renal_arteries",
        ],
    }
    calls = []

    nnunet_outputs = {
        716: task_outputs["aortic_dissection"],
        713: task_outputs["aorta_annulus"],
        710: task_outputs["renal_arteries"],
    }

    async def fake_modal_run(_image_arg, function_name, options):
        calls.append((function_name, options))
        outputs = (
            task_outputs["total"]
            if function_name == model_runner.MODAL_SEGMENTATION_FUNCTION
            else nnunet_outputs[options["task_id"]]
        )
        return _multilabel_image(outputs)

    monkeypatch.setattr(model_runner, "_run_modal_async", fake_modal_run)
    logger = SimpleNamespace(info=lambda _message: None)

    asyncio.run(
        model_runner.run_models_parallel(
            ct_path, tmp_path / "outputs", logger, host="modal"
        )
    )

    assert [function_name for function_name, _ in calls] == [
        model_runner.MODAL_SEGMENTATION_FUNCTION,
        model_runner.MODAL_NNUNET_FUNCTION,
        model_runner.MODAL_NNUNET_FUNCTION,
        model_runner.MODAL_NNUNET_FUNCTION,
    ]
    assert [options["task_id"] for _, options in calls[1:]] == [716, 713, 710]
    for outputs in task_outputs.values():
        for output in outputs:
            assert (tmp_path / "outputs" / f"{output}.nii.gz").exists()


def test_modal_contrast_phase_is_cached(tmp_path, monkeypatch):
    ct_path = _save_image(tmp_path / "ct.nii.gz", np.zeros((3, 3, 3)))
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    calls = []

    class Remote:
        def __call__(self, image):
            calls.append(image)
            return {"phase": "arterial_early"}

    function = SimpleNamespace(remote=Remote())
    monkeypatch.setattr(
        model_runner, "_get_modal_function", lambda _name: function
    )
    logger = SimpleNamespace(info=lambda _message: None)

    assert (
        model_runner.get_contrast_phase(
            ct_path, output_dir, logger, host="modal"
        )
        == "arterial_early"
    )
    assert (
        model_runner.get_contrast_phase(
            ct_path, output_dir, logger, host="modal"
        )
        == "arterial_early"
    )
    assert len(calls) == 1
