# EnergyTransformer Spark kit — v23

Read **[V23_START_HERE.md](V23_START_HERE.md)** for the final changes and
**[SPARK_START_HERE.md](SPARK_START_HERE.md)** for the detailed device/pilot sequence.

```bash
bash spark/run.sh doctor
bash spark/run.sh accept
```

The standalone download contains the unchanged 972,896-parameter native-text
checkpoint. A Git source checkout requires installing that original binary from
the supplied download; see [checkpoints/README.md](checkpoints/README.md).

No GPU or NVIDIA container was executed during this revision. There is no new
model training or accuracy claim. A passing test on the actual Spark permits
only a capped single-device pilot, not serious general pretraining.

The current source lives in `spark_kit/` within the historical repository so it
does not overwrite the earlier experiments or reviewed memory interfaces.
