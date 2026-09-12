# Where to put the research

Hugging Face is optional. Use GitHub for source, issues and reproducibility; GitHub
Releases can host this project's full archive (each asset must be below 2 GiB).
Hugging Face is a useful optional home for individual model checkpoints and cards.
Neither uploading a file nor converting to safetensors makes it an AutoModel,
instruction-following assistant, or hosted inference service.

The repository is private. None of these scripts changes visibility or licensing.
Do not paste access tokens in chat or commit them. Authenticate locally with `gh auth
login` or `hf auth login` using repository-scoped permissions where possible.

## Full archive to GitHub

Download the full research ZIP to the same machine as the uploader, then:

```
python scripts/publish_release.py --archive /path/to/archive.zip --tag research-v1-v16 --publish
```

The uploader verifies remote bytes. Omitting `--publish` performs local validation.
The legacy publisher's release-note title refers to its v15 origin; archive filenames,
hashes and the explicit tag identify the actual supplied payload.

## Optional full archive to Hugging Face

Use an explicitly named EXISTING PRIVATE model repository. No name or owner is guessed:

```
pip install huggingface_hub
hf auth login
python scripts/publish_hf.py --repo-id YOUR_ACCOUNT/EnergyTransformer --archive /path/to/archive.zip --publish
```

This uploads under `research/<sha256>/`, verifies a re-download, and refuses public
repositories or conflicting existing content. It never automatically approves edits,
creates an account, or changes a repository to public. Tests use a fake HF backend;
a live upload from this environment has NOT been completed.

## Portable bounded model

`portable/seed11/` contains tensor weights and a JSON vocabulary/configuration.
Use `PortableEnergyTextSession` from `energy_transformer.portable`, with the current
source installed. All 3 seed exports preserve tensor values; this is the small v14
sentence parser and answer decoder, not the older unrestricted byte-LM backbone.
