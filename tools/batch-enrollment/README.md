# Enroll a GPU for batch replay

On a quiet stock boot with native GL and Borealis, stage gpubench-gl and gpubench-vk from the same build, Pillow, GFXReconstruct and Waffle-enabled eglretrace. Use a known-good Mesa build whose GL and Vulkan info strings include its Git revision. The two reference images come from headless llvmpipe/lavapipe replay of the pinned public triangle captures. The GL enrollment permits differences along the outer triangle silhouette. Subsequent comparisons use the normal gate tolerance.

```sh
bash tools/batch-enrollment/prepare.sh kit /opt/gpubench/bin
python3 kit/run.py --board j413 --sha "$MESA_SHA" --corpus corpus --out boot1 --enroll
# On another stock boot with the same Mesa, firmware and kernel:
python3 kit/run.py --board j413 --sha "$MESA_SHA" --corpus corpus --out boot2
```

A visual-only row adds `--visual-only`. For an isolated prefix, pass the environment JSON file with `--env`. Keep corpus, boot metadata, raw rows, logs and screenshots alongside the health and driver library hashes from the lead. A timing FLAG is a calibration observation and leaves the visual evidence available for review.
