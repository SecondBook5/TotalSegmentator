# Pulmonary artery report

`totalseg_pulmonary_report` measures five standardized locations of the
pulmonary artery and writes the measurements as JSON and as a rotating RGB
NIfTI report.

The report is a research prototype and is not intended for clinical diagnosis.

## Requirements

Running the segmentation models requires a TotalSegmentator license. A free
non-commercial license is available from
[backend.totalsegmentator.com/license-academic](https://backend.totalsegmentator.com/license-academic/).
Configure it with:

```bash
totalseg_set_license -l aca_XXXXXXXX
```

The report renderer requires `wkhtmltopdf` and a virtual X server. On
Ubuntu/Debian:

```bash
sudo apt-get install wkhtmltopdf xvfb
```

## Complete pipeline

Use `--run_models` to produce all required masks from a CT:

```bash
totalseg_pulmonary_report \
  -i ct.nii.gz \
  -o pulmonary_report.nii.gz \
  -j pulmonary_report.json \
  -l pulmonary_report.log \
  --run_models
```

The pipeline creates a fast heart crop, segments the pulmonary artery with
`heartchambers_highres`, and predicts five landmark masks with
`pulmonary_artery_landmarks`.

Segmentation can instead run through the configured `totalsegmentator` Modal
app. Install the optional `modal` package and pass:

```bash
totalseg_pulmonary_report \
  -i ct.nii.gz \
  -o pulmonary_report.nii.gz \
  -j pulmonary_report.json \
  -l pulmonary_report.log \
  --run_models --host modal
```

Precomputed masks can be supplied instead:

```bash
totalseg_pulmonary_report \
  -i ct.nii.gz \
  -rt masks/ \
  -rd masks/ \
  -o pulmonary_report.nii.gz \
  -j pulmonary_report.json \
  -l pulmonary_report.log
```

The ROI directories must contain `pulmonary_artery.nii.gz`,
`pul_annulus.nii.gz`, `pul_sinotubular_junction.nii.gz`,
`pul_bifurcation.nii.gz`, `pul_left_start.nii.gz`, and
`pul_right_start.nii.gz`.

## Measurements

The five report positions are:

1. proximal main pulmonary artery;
2. pulmonary sinotubular junction;
3. pulmonary artery bifurcation;
4. left pulmonary artery;
5. right pulmonary artery.

For every available landmark, the JSON contains cross-sectional area in cm²,
maximum and near-perpendicular diameters in cm, and diameter endpoints in voxel
and world coordinates. Missing landmarks are represented by `null` values.

## Limitations

- Results depend directly on the pulmonary artery and landmark segmentations.
- The proximal main pulmonary artery and left pulmonary artery can be
  incomplete near segmentation boundaries.
- Rendering is computationally and memory intensive.
