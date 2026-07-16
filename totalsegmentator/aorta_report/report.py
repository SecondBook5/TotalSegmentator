"""Compatibility entry point for the aorta report pipeline."""

from totalsegmentator.aorta_report.measurements import diameter_profiles as _diameter_profiles
from totalsegmentator.aorta_report.pipeline import create_aorta_report

__all__ = ["create_aorta_report"]
