import nibabel as nib
import numpy as np
from PIL import Image

from totalsegmentator.aorta_report.centerline import Vertex
from totalsegmentator.aorta_report.landmarks import (
    attach_structure_anchors,
    create_landmarks,
    create_structures,
)
from totalsegmentator.aorta_report import rendering


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_sinotubular_junction_does_not_require_brachio_anchor():
    shape = (10, 10, 12)
    aorta = np.zeros(shape, dtype=np.uint8)
    aorta[4:7, 4:7, 1:11] = 1
    structures = create_structures()
    for structure in structures.values():
        structure["data"] = np.zeros(shape, dtype=np.uint8)
    structures["sinotub_junc"]["data"][2:8, 2:8, 2:10] = 1
    centerline = [Vertex((5, 5, z)) for z in range(1, 11)]
    logger = _Logger()

    result = attach_structure_anchors(
        structures, aorta, centerline, (1, 1, 1), logger
    )

    assert result["brachio"]["empty"]
    assert not result["sinotub_junc"]["empty"]
    assert result["sinotub_junc"]["cl_idx"] is not None


def test_empty_structure_dependencies_produce_empty_landmarks_without_index_errors():
    structures = create_structures()
    for structure in structures.values():
        structure["empty"] = True
    centerline = [Vertex((0, 0, z)) for z in range(5)]
    aorta_img = nib.Nifti1Image(np.ones((3, 3, 5), dtype=np.uint8), np.eye(4))
    logger = _Logger()

    landmarks = create_landmarks(
        structures,
        annulus_volume=0,
        centerline=centerline,
        aorta_img=aorta_img,
        spacing=(1, 1, 1),
        logger=logger,
    )

    assert all(landmark["empty"] for landmark in landmarks.values())
    assert set(landmarks) == set(range(1, 12))


def test_report_frontpage_renders_layout_once(monkeypatch, tmp_path):
    monkeypatch.setattr(rendering, "REPORT_FRAMES", 2)
    monkeypatch.setattr(rendering, "REPORT_PLOT_TYPES", ("first_", "second_"))
    colors = ((10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120))
    for index, (prefix, frame) in enumerate(
        (("first_", 0), ("first_", 1), ("second_", 0), ("second_", 1))
    ):
        Image.new("RGB", (4, 3), colors[index]).save(
            tmp_path / f"{prefix}{frame:06d}.png"
        )

    render_calls = []

    def fake_generate_html(_assets, _template, values, output_path, **_kwargs):
        render_calls.append(values["preview_3d"])
        image = Image.new("RGB", (10, 8), "black")
        image.paste(Image.open(values["preview_3d"]), (2, 2))
        image.save(output_path)
        return image

    captured = []

    def fake_combine(paths):
        captured.extend(np.asarray(Image.open(path))[3, 3].tolist() for path in paths)
        return "combined"

    monkeypatch.setattr(rendering, "generate_html", fake_generate_html)
    monkeypatch.setattr(rendering, "combine_as_nifti", fake_combine)

    result = rendering.render_report_image({}, {}, {}, tmp_path)

    assert result == "combined"
    assert len(render_calls) == 1
    assert captured == [list(color) for color in colors]
