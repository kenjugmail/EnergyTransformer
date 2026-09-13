# Current DGX Spark path — v23

The complete current Spark source is in [`spark_kit/`](spark_kit/README.md).
It is isolated from the earlier experimental modules and does not overwrite them.

Start with [the runbook](spark_kit/SPARK_START_HERE.md) and
[the executed v23 report](spark_kit/reports/v23/REPORT.md).

```bash
cd spark_kit
# Install the unchanged original model from the provided standalone download:
python -m energy_spark.install_checkpoint /path/to/EnergyTransformer_v23_Spark_Kit.zip
bash spark/run.sh doctor
bash spark/run.sh accept
```

The source commit is not a binary release. The model and complete historical
collection are in the supplied downloads until separately uploaded and verified.
No new language weights, actual GB10 result, or serious-pretraining approval is
claimed by this engineering revision. The repository's visibility and license are
unchanged. Use `spark_kit/publish_release.py` from a networked, authenticated machine
to transfer the original archive to a verified private GitHub release.
