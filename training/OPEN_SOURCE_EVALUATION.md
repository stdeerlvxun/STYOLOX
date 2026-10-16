# Isolated open-source evaluation

`evaluate_anomalib.py` adapts the official Anomalib 2.2.0 PatchCore implementation
to an existing frozen, reviewed detector dataset. It is an exploratory local
experiment, not a labeler deployment or an automatic model promotion.

## Data boundaries

- Keep the existing `human_confirmed_problem_boxes_v1` eligibility policy.
- Only explicit, accepted local normal feedback on eligible problem photographs
  may build the normal memory bank. Image-status photographs, drafts, unreviewed
  model suggestions, and unboxed background are not normal training examples.
- Recompute case/order/SHA/perceptual groups against current and historical
  holdouts. Split normal regions into memory and calibration groups without
  group overlap. Crop inward to avoid adding neighbouring unconfirmed pixels.
- Read NAS originals with SHA and EXIF geometry checks. Do not relocate originals
  or write classifications, boxes, review batches, or server configuration.
- Evaluate the existing validation split only. Do not read test image pixels.

## Fixed pilot protocol

The pilot uses the official torchvision ImageNet ResNet18 weights with
`layer2`/`layer3`, nine neighbours and a 5% coreset. A seeded reservoir caps
pre-coreset features at 20,000. It does not fine-tune the pretrained backbone.

At inference, the longest image side is capped at 1024 pixels. Overlapping
256-pixel tiles, stride 128, are resized to 224 pixels and ImageNet normalized.
Overlapping anomaly maps are averaged. The anomaly threshold is the fixed
99.5th percentile of held-out normal-region pixels, independent of validation
defect labels. Connected components with at least 16 pixels become candidates;
NMS IoU is 0.5 and at most ten candidates are retained per image.

Location evaluation reuses the existing one-to-one, score-ordered IoU 0.5
matcher. Heatmap response, a candidate touching a defect, and a correctly
localized defect are different measurements. This experiment does not predict
issue type or responsibility. Its raw anomaly distance is not a probability.

## Isolated environment and command

Use a separate Python environment with `anomalib==2.2.0` and compatible PyTorch
and torchvision. Do not upgrade the working detector environment. The explicit
backbone option below verifies the official complete SHA256 and avoids implicit
model-hub downloads.

```powershell
python evaluate_anomalib.py --dataset <frozen-dataset.json> --run <new-output-folder> --device cuda --backbone-weights <resnet18-f37072fd.pth> --baseline-json <retained-validation-report.json>
python -m unittest test_evaluate_anomalib -q
```

The output folder must be new. Outputs include memory-bank state, its normal
region split, predictions, detailed matches, three derived overlays and metrics.
Private run artifacts and business images must remain outside the source repo.
When sharing the GPU with the existing correction worker, acquire its
`.correction-cycle.lock` for the experiment. Keep the worker's schedule enabled.

## Upstream availability and limitations

- [Anomalib](https://github.com/open-edge-platform/anomalib) provides complete
  PatchCore and EfficientAD implementations. An EfficientAD architecture smoke
  test alone is not trained business accuracy evidence.
- [YOLO-MDEW](https://github.com/ansker3/YOLO-MDEW), audited at commit
  `fc3d1a6fa91a12ab96f79eaf7aecbcebcedf51f2`, provides module snippets but no
  complete architecture config, training/inference entrypoint, dependency lock,
  or ready detector checkpoint in the repository/releases. Its source also has
  unresolved class references. Reconstructing it would be a separate reproduction
  experiment, not direct use of author-trained weights.
- This lightweight PatchCore pilot is one backbone/data/proposal configuration.
  Its result does not establish the performance of every Anomalib model. Normal
  regions may poorly cover varied materials, watermarks, packaging and camera
  conditions in after-sales photographs. A heatmap-to-box conversion can also
  produce boxes much smaller than the reviewed ground truth.

Reject a regressing experiment. Keep the online detector and human annotations
unchanged; an exploratory accuracy report does not authorize automatic review.
