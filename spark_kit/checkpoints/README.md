# Original native-text checkpoint

The standalone v23 Spark kit includes `v21_energy_973k.pt`. A source-only Git clone
does not include that binary. No substitute or regenerated weights are used.

SHA-256: `78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b`

From this kit directory, install it from the downloaded standalone ZIP or original
`.pt` file using only local, hash-verified copying:

```bash
python -m energy_spark.install_checkpoint /path/to/EnergyTransformer_v23_Spark_Kit.zip
```

The installer performs no network calls and refuses conflicting destination files.
Do not run the trainer with an invented/random checkpoint under this filename.
