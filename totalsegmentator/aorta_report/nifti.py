import nibabel as nib
import numpy as np

from totalsegmentator.reporting import (
    combine_rgb_slices_as_nifti,
    rgb_array_to_structured,
)


def _allocate_rgb_nifti_buffer(width, height, nr_frames):
    return np.zeros((width + 1, height + 1, nr_frames, 3), dtype=np.uint8)


def _write_rgb_frame(buffer, frame, frame_index):
    """Write one display-order RGB frame using the established report orientation."""
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    output_index = buffer.shape[2] - 1 - frame_index
    oriented = frame.transpose(1, 0, 2)[::-1, ::-1, :]
    buffer[:width, -height:, output_index, :] = oriented


def _save_rgb_nifti_buffer(buffer, output_path):
    structured = rgb_array_to_structured(buffer)[::-1, ::-1, :]
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[1, 1] = -1
    nib.save(nib.Nifti1Image(structured, affine), output_path)


def combine_as_nifti(slice_images, logger=None):
    return combine_rgb_slices_as_nifti(
        slice_images, logger=logger, reverse_z=True
    )


__all__ = ["combine_as_nifti", "rgb_array_to_structured"]
