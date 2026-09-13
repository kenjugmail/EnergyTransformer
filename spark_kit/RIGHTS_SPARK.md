# Scope, provenance, and rights

This kit specializes source developed in this EnergyTransformer conversation. It preserves the supplied v21 energy checkpoint byte-for-byte (SHA-256 recorded in the manifest). No external corpus bytes or downloaded third-party model weights are bundled. Third-party packages (PyTorch, NumPy, and optionally NVIDIA's container) remain governed by their own terms and are not relicensed here.

The inherited model was trained on the historical Wikipedia fixture described in `reference/V21_REPORT.md`; article/revision provenance appears in `reference/v21_wiki_manifest.json` where present. Packaging weights does not independently settle every question about applicable data or model licensing. No new public/open-source license is imposed on the user's repository by this kit.

Generated software-test fixtures are labeled as such. They do not contain user documents. The supplied scripts do not upload data, access credentials, modify host drivers, or provision a paid service. Local run metadata and checkpoints can nonetheless reveal data/source identities or learned information; inspect them before sharing.
