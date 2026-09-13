# EnergyTransformer

Experimental energy-function memory, native-text prototypes, and explicitly reviewed
protected-memory edits. **Not a competitive general-purpose LLM or an instruction-following assistant.**

## Current target: a bounded DGX Spark test

The complete self-contained **v23 Spark source** is in [`spark_kit/`](spark_kit/README.md).
It includes the native energy/softmax implementations, resumable training, local
corpus preparation, device acceptance, configs, tests, and a read-only host doctor.
The earlier experimental modules and their authorization boundaries remain unchanged.

- [Start on the Spark](spark_kit/SPARK_START_HERE.md)
- [Architecture and math](spark_kit/ARCHITECTURE_SPARK.md)
- [Executed v23 changes and tests](spark_kit/reports/v23/REPORT.md)
- [Original model installation](spark_kit/checkpoints/README.md)

```bash
cd spark_kit
# The standalone download already includes the original checkpoint.
# A source-only clone first installs it from the supplied local kit:
python -m energy_spark.install_checkpoint /path/to/EnergyTransformer_v23_Spark_Kit.zip
bash spark/run.sh doctor
bash spark/run.sh accept
```

The final focused kit suite passes 83 local CPU tests. The new importer uses less
Python-tracked preparation memory on the measured fixture, and resume preserves
crash-tail logs. **No new model training, actual GB10/CUDA result, or serious general
pretraining approval is claimed by v23.** Run acceptance on the target device before
starting any corresponding capped pilot.

## Source and binary publication are different

This source commit does not upload the original pretrained binary or historical
experiment archives. The complete available collection is preserved in the supplied
v1-v23 download: 22 original experiment ZIPs and 96 original standalone artifacts,
plus the current kit. Their inventory/checksums are not a substitute for an upload.

From a networked machine holding that download and normal GitHub CLI authorization:

```bash
python spark_kit/publish_release.py \
  --archive /path/to/EnergyTransformer_All_Research_v1_v23.zip --publish
```

The uploader refuses a public target and conflicting assets, verifies downloaded
bytes, and publishes only after verification. It never force-pushes or changes
repository visibility. Do not paste tokens into chat or commit credentials.

## Earlier research and bounded memory interfaces

[Earlier README](README_LEGACY.md) preserves the previous entry point byte for byte.
See [architecture](ARCHITECTURE.md), [model card](MODEL_CARD.md),
[rights](RIGHTS.md), [storage](STORAGE.md), and [history](reports/HISTORY.md).
The existing private releases contain bounded parser/answer checkpoints, not the
latest native-text model or all original historical bytes. Their limitations remain.

No new open-source license is imposed. Old failed experiments and classical controls
retain their original qualifications. Energy coefficients remain numerical internal
state; terminology alone does not establish novel mathematics or superiority over
attention. The next useful evidence is target-device numerical behavior and paired
validation curves against measured training cost.
