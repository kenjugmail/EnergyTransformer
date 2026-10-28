from dataclasses import asdict
from pathlib import Path
import shutil

import pytest
import torch

from rewt.model import EnergyConfig, TransformerConfig, build_model
from rewt.stabilize import verify_reused_inference


@pytest.mark.parametrize("arm", ["transformer", "fourier_rewt", "full_rewt"])
def test_reused_checkpoint_guard_compares_archived_inference_and_rejects_changed_weights(tmp_path, arm):
    source = tmp_path / "source"
    source.mkdir()
    package = Path(__file__).parents[1] / "rewt"
    for name in ("__init__.py", "model.py"):
        shutil.copy2(package / name, source / name)
    config = (TransformerConfig(width=8, layers=1, feedforward=16) if arm == "transformer" else
              EnergyConfig(width=8, hidden=16, modes=4, support=arm == "full_rewt",
                           output_geometry="centered"))
    model = build_model(arm, config)
    checkpoint = dict(arm=arm, seed=17, config=asdict(config), weights=model.state_dict())
    original, copied = tmp_path / "original.pt", tmp_path / "copied.pt"
    torch.save(checkpoint, original)
    shutil.copy2(original, copied)
    ledger = [dict(original_checkpoint=str(original), copied_checkpoint=str(copied),
                   seed=17, variant=arm, phase="language")]
    assert verify_reused_inference(tmp_path, ledger)[0]["exact_logits_equal"]
    checkpoint["weights"]["embedding.weight"] = checkpoint["weights"]["embedding.weight"] + .25
    torch.save(checkpoint, copied)
    with pytest.raises(ValueError, match="inference changed"):
        verify_reused_inference(tmp_path, ledger)
