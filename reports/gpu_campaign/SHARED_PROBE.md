# Shared-device small-model probe

The user authorized preliminary training alongside existing Spark workloads with
4 GiB maximum Torch allocations and at least 16 GiB available host memory. This is
an explicit exception for this bounded diagnostic, not a change to the exclusive
campaign's requirements. No existing service is stopped or restarted.

Protocol: first run independent serial-recurrence and model streaming/reset/
causality/BF16 checks. If they pass, fit full attention, local-128 attention, pure
energy and hybrid at 1.05M and 3.33M parameters. Use seed 404, a paired byte stream,
context 512, microbatch 1, accumulation 16, eight updates (65,536 targets), fixed
peak LR 0.001 and validation every two updates on 2,048 targets. This learning rate
is a diagnostic setting, not validation-selected calibration. Retain checkpoints
and per-update loss records. Do not use test loss for selection.

The dedicated entry point is `python -m energy_campaign.shared_probe
runs/campaign runs/campaign/shared_probe_001`. It has one 300-second numerical
reservation and up to eight 120-second fit reservations. All actual subprocess
wall time charges the existing acceptance/backbone ledger. An incomplete fit is
retained and stops the probe. It is not silently compared with completed fits.

The supervisor checks host availability every two seconds and stops only its own
process group if the 16 GiB reserve is breached. It requires 20 GiB available
before launching each child. The Torch allocator is capped at 4 GiB; CUDA context
and other non-Torch allocations are not covered by that allocator cap. Numerical
and model memory measurements are recorded. No utilization or latency isolation
is claimed.

This can establish that configurations fit, updates remain finite, and early loss
curves are reproducible from saved artifacts. Eight updates and one seed cannot
establish convergence, scaling exponents, architecture superiority, or an unbiased
speed comparison. GPU compilation and full restart acceptance still belong to the
exclusive campaign. This probe never emits a production acceptance receipt.
