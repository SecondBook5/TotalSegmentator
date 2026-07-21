import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REFERENCE_DIR = Path(__file__).parent / "reference_files" / "pulmonary_report"
AORTA_REFERENCE_DIR = Path(__file__).parent / "reference_files" / "aorta_report"
EXAMPLE_CT = AORTA_REFERENCE_DIR / "example_ct.nii.gz"
EXAMPLE_NODEINFO = AORTA_REFERENCE_DIR / "example_nodeinfo.json"
REFERENCE_JSON = REFERENCE_DIR / "pulmonary_report.json"
CACHED_MODELS_DIR = REFERENCE_DIR / "models"


def dicts_almost_equal(first, second):
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            dicts_almost_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            dicts_almost_equal(left, right)
            for left, right in zip(first, second)
        )
    if isinstance(first, int) and isinstance(second, int):
        return abs(first - second) <= 1
    if isinstance(first, float) and isinstance(second, float):
        return abs(first - second) <= 2
    return first == second


def remove_keys_with_prefix(value, prefixes):
    if isinstance(value, dict):
        return {
            key: remove_keys_with_prefix(item, prefixes)
            for key, item in value.items()
            if not any(key.startswith(prefix) for prefix in prefixes)
        }
    if isinstance(value, list):
        return [remove_keys_with_prefix(item, prefixes) for item in value]
    return value


@pytest.fixture(scope="module")
def pulmonary_report_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pulmonary_report")
    output_nifti = tmp_path / "pulmonary_report.nii.gz"
    output_json = tmp_path / "pulmonary_report.json"
    output_log = tmp_path / "pulmonary_report.log"
    shutil.copytree(CACHED_MODELS_DIR, tmp_path / "models")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "totalsegmentator.bin.totalseg_pulmonary_report",
            "-i",
            str(EXAMPLE_CT),
            "-n",
            str(EXAMPLE_NODEINFO),
            "-o",
            str(output_nifti),
            "-j",
            str(output_json),
            "-l",
            str(output_log),
            "-tmp",
            str(tmp_path),
            "-a",
            "pulmonary_artery.nii.gz",
            "-t",
            "basic",
            "--run_models",
            "--debug",
        ],
        check=True,
    )
    return {
        "output_nifti": output_nifti,
        "output_json": output_json,
        "output_log": output_log,
    }


def test_pulmonary_report_logs(pulmonary_report_result):
    logs = pulmonary_report_result["output_log"].read_text()
    for content in (
        "already cropped (skipping)",
        "Skipping TotalSeg pulmonary fast (already exists)",
        "Skipping TotalSeg (already exists)",
        "Skipping T514 (already exists)",
        "Processing landmark 1",
        "Processing landmark 5",
    ):
        assert content in logs, f"required content not found in logs: {content}"


def test_pulmonary_report_json(pulmonary_report_result):
    with REFERENCE_JSON.open() as file:
        expected = json.load(file)
    with pulmonary_report_result["output_json"].open() as file:
        actual = json.load(file)
    expected = remove_keys_with_prefix(expected, ("start", "end"))
    actual = remove_keys_with_prefix(actual, ("start", "end"))
    expected["metadata"].pop("version", None)
    actual["metadata"].pop("version", None)
    assert dicts_almost_equal(expected, actual)


def test_pulmonary_report_files_exist(pulmonary_report_result):
    assert pulmonary_report_result["output_nifti"].is_file()
