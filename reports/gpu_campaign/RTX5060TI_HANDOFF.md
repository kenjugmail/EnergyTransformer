# RTX 5060 Ti handoff: connection verified, launch pending constraints

Recovered from the user's Claude LoopTest session and its `rtx5060ti-pc.md` memory,
then verified over SSH. No services were changed.

- SSH: `kenju@100.105.42.58` (Windows OpenSSH, existing key authentication).
- WSL execution: `wsl.exe -e bash -lc ...`.
- Host: DESKTOP-SEVJ7KM; x86-64.
- GPU: RTX 5060 Ti, 16,311 MiB VRAM; live check found 14,066 MiB used and
  1,985 MiB free. Two VLLM engine processes belong to existing services.
- WSL memory: approximately 15 GiB total, 7.6 GiB available. This cannot satisfy
  the existing 16 GiB available-host-memory gate even with services stopped.
- Existing Python: `/root/arbiter/venv/bin/python`, Python 3.12.13,
  Torch 2.13.0+cu130 / CUDA 13.0. Do not replace this shared serving environment.
- Windows backing drive: about 36 GB free; WSL's reported 814 GB virtual free
  space is not a safe physical disk budget. Avoid copying venvs or large run trees.
- Sentinel scripts: `/root/arbiter/sentinel/sentinel_pc.sh`.

The Claude session records the user's instruction not to keep restarting Sentinel.
The approved research plan also forbids interrupting services. Permission to use
the GPU establishes the target, but does not resolve these explicit constraints.
A user question is pending: temporarily pause Sentinel and the PC writer, restore
them afterward, and use a 4 GiB host-memory reserve, or retain services and wait.
No pause/restart or reduced reserve is authorized yet.

## Preparation after that decision

1. Keep the approved cumulative 24-hour allocation. Transfer ledger ownership;
   never start separate 24-hour budgets on Spark and PC or duplicate the campaign.
2. Use a separate research directory/environment under `/root/arbiter`; preserve
   the shared serving environment. Transfer only frozen source/wheel, tests,
   manifests, corpus, donor and historical checkpoint needed by the study.
3. Make the historical checkpoint path portable (currently hardcoded to Spark),
   and bind the new runtime, device and explicitly accepted resource policy to the
   experiment specification. Original GB10 hardware receipts cannot authorize PC runs.
4. Re-run installed CPU checks on the x86 Torch 2.13 environment. Do not copy the
   Arm64 virtualenv or claim equivalence to Torch 2.14 without testing.
5. Run fresh GPU numerical acceptance and real-shape profiling. Size training
   allocations from measured 5060 Ti throughput. Use an explicitly bounded VRAM
   cap with headroom for the display; large contexts/batches may be infeasible.
6. Follow the existing stop-on-failure and no-automatic-promotion rules. Restore
   serving configuration and verify service health after the reserved training window.

The live setup differs from older Claude notes (which reported 23 GB WSL memory).
Current observations, not those historical figures, must determine launch settings.
