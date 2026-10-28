# REWT improvements supported by research and validation

October 9, 2026

REWT should first improve the relationship between its energy and its predictions. The existing implementation performs valid energy descent, but its output representation gives the energy directions that leave the current probabilities unchanged. A validation probe also shows that local descent can slightly worsen prediction loss. Longer training remains necessary, but extra modules and extra refinement rounds alone do not address these problems.

The proposed changes below preserve one joint energy over the recurrent, attention, forecast, diffusion, specialist, communication, and output states. Inference changes states and predictions with fixed weights. Future targets and correctness labels belong exclusively in training losses.

The existing [pilot findings](/home/kt/Documents/EnergyTransformer/reports/rewt_pilot_20261009_v2/FINDINGS.md) cover four models, three seeds, roughly 250K parameters, and only 61,440 language-training bytes per fit. The new [checkpoint measurements](/home/kt/Documents/EnergyTransformer/reports/rewt_research_20261009/checkpoint_diagnostics.json) use eight existing validation windows, 1,024 positions per checkpoint, for the three basic and three full Fourier models. Checkpoint hashes before and after the probe match. No weights change and no test data enter these new diagnostics.

| New validation observation | Interpretation |
| --- | --- |
| The mean share of squared initial output-gradient magnitude in the common-logit direction ranges from 39.5% to 60.6% across six checkpoints. | A substantial part of the output update changes a common offset. That component does not change the current softmax probabilities, although state coupling can still affect subsequent dynamics. |
| Adding one to every output logit changes mean energy by approximately 127–134 while changing probabilities by less than 2.4 × 10⁻⁷. | The energy strongly distinguishes output coordinates that represent the same immediate prediction. This confirms the representation mismatch independently of gradient decomposition. |
| With conditioning frozen to the original two-round prefixes, two local rounds worsen mean prediction bpb in every checkpoint while decreasing energy. | Energy descent and immediate prediction improvement are misaligned in this probe. This is a local conditional test, not a replacement for full autoregressive rollout evaluation. |
| The initial output wave-gradient norm averages approximately 13–19% of the quadratic-gradient norm on a per-position ratio basis. | Quadratic forces dominate the initial output gradient. Increasing wave strength is a hypothesis to test, not an automatic remedy. |
| The correct-byte candidate beats an arbitrary wrong byte about 97% of the time, but beats the model's strongest wrong alternative only about 30%. | Easy negatives are much less challenging than plausible wrong predictions. These are byte-ranking diagnostics; next-byte ambiguity also limits exact accuracy. |

The earlier full-rollout ablation found a tiny improvement with two rounds because changing the rounds also changes the carried recurrent state and subsequent conditioning. The frozen-prefix probe holds those influences fixed and isolates local refinement. Both measurements are needed.

1. **Make the output representation agree with prediction semantics.** For logits \(L\), adding the same constant to every coordinate leaves \(\operatorname{softmax}(L)\) unchanged. The current quadratic anchors, decoder residual, and Fourier waves still change under that transformation. This mathematical mismatch is directly visible in the common-offset diagnostic.

   A first revision should constrain logits to zero mean and make every output-dependent energy term use the same centered coordinates. An alternative should feed normalized probabilities into the energy while optimizing their underlying logits. Centering only after accepting an update would violate the existing energy guarantee; initialization, derivatives, constraints, and acceptance must use the revised representation consistently.

   The EBT tutorial reports better stability when candidate distributions are normalized with softmax and identifies optimization step size as a sensitive hyperparameter. Its numerical step sizes use a different energy scale and should not be copied into REWT. [Gladstone et al., Energy-Based Transformers, Appendix I](https://arxiv.org/html/2507.02092v1)

   Verify probability invariance under a common logit shift, correct analytic derivatives, full-energy decrease, and improved validation loss per refinement round. These changes require retraining; editing the geometry of an existing checkpoint is not a fair architecture comparison.

2. **Train the energy to repair imperfect predictions.** REWT already backpropagates through accepted refinement. The weak benefit suggests that this objective, its initializations, and its relative loss weights need a stronger test. Compare ordinary learned initialization with occasional corrupted or noisy initial guesses, using only observed-prefix conditioning. Compare final-round prediction loss with a small training-only penalty when refinement increases prediction loss. The latter is a proposed auxiliary objective, not an established REWT fix.

   Use wrong bytes that the model considers plausible as ranking negatives, alongside ordinary negatives. For reasoning, generate plausible incorrect traces and label the first invalid step with the existing deterministic training-task checker. Correct hidden states are not directly observed, so arbitrary latent vectors should not be described as gold reasoning states.

   End-to-end SPEN training optimizes task performance through gradient-based prediction. It provides evidence for evaluating the learned optimizer by its output quality rather than by convergence alone. [Belanger, Yang and McCallum, ICML 2017](https://proceedings.mlr.press/v70/belanger17a.html)

   Process supervision improved reward-model solution selection on MATH, and the study deliberately sought convincing incorrect solutions for labeling. Applying those principles to a tiny joint-state energy model is an adaptation; the published result does not establish that a new REWT loss will work. [Lightman et al., Let's Verify Step by Step, Sections 2.4–2.6](https://arxiv.org/html/2305.20050v1)

3. **Train across refinement depths and tune state-update scales.** The current primary fits train two rounds and later evaluate up to eight. Sample a bounded range of positive round counts during training, initially one to four, and compare validated step sizes. A later experiment can learn positive block-specific update scales while retaining acceptance against the complete energy. Keep the conditioning and energy fixed within each trajectory.

   EBT experiments use varied optimization trajectories, and their ablation shows that random step sizes can materially affect additional-inference gains. Their same-landscape inference recipe is relevant to the one-energy requirement. [Gladstone et al., Sections 3.3 and I.3](https://arxiv.org/html/2507.02092v1)

   Recurrent-depth language modeling also trains with sampled iteration counts and limits backward propagation through long recurrences. [Geiping et al., Section 3.3](https://arxiv.org/html/2502.05171v1)

   Tiny Recursive Models apply repeated supervised improvement cycles on puzzle tasks. A supervised cycle is different from one REWT gradient round, and their millions of parameters and substantial training budget differ from this pilot. Compare supervision strategies rather than assuming that supervising every energy round is universally superior. [Jolicoeur-Martineau, Sections 4.1 and experimental setup](https://arxiv.org/html/2510.04871v1)

4. **Give the wave representation a controlled capacity study.** Sixteen frequency directions limit the Fourier gradient to their span, even though the quadratic gradient can act outside it. This is a restriction on the wave contribution, not a proof that the entire energy is low rank. Compare 16, 32, and 64 modes, frequency bandwidths, and normalized block scales. Track wave and quadratic gradients separately and compare against equally tuned non-Fourier energies under parameter and compute controls.

   Fourier-feature research shows that bandwidth changes fitting and generalization in low-dimensional regression. It motivates tuning frequency scales but provides no direct language-model or joint-energy guarantee. [Tancik et al., NeurIPS 2020, Sections 4–5](https://proceedings.neurips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)

   Learned phase shifts already let a cosine mode represent a shifted sine. Simply adding sine functions at the same frequencies should not be presented as an independently justified improvement. Context-dependent phase or nonlinear mixing would be distinct architectural experiments.

5. **Establish basic learning before testing reasoning extrapolation.** Use a longer common training budget for the Transformer, non-Fourier energy control, and core REWT. A proposed byte-budget ladder is 0.25M, 1M, and 4M, subject to profiling and the course compute allowance. These are experimental checkpoints, not literature-derived sufficiency thresholds. Fit learning curves and tune settings on validation.

   For synthetic reasoning, first measure training-set and same-template held-out accuracy on one-step problems, then two- and three-step chains. Keep the withheld wording and four-step tests separate. This isolates inability to learn the task from inability to generalize its composition. The arithmetic and rule checks stay outside inference.

   Small algorithmic models can generalize much later than they memorize training examples; this is one reason that a short run cannot establish a reasoning limit. It does not mean that every architecture will eventually generalize. [Power et al., Grokking](https://arxiv.org/abs/2201.02177)

   Restore full support-module comparisons with sufficient capacity: the matched-parameter full pilot reduces its feedforward hidden width from 713 to 382, and its forecast teacher also receives only the brief pilot training. Test each component with retraining and charge all teacher, auxiliary-loss, and evaluation costs. Keep the complete full implementation available throughout.

6. **Make additional inference selective and cheaper.** Profile and batch the small proposal networks where possible, cache observed-history keys and values, and fuse compatible energy/gradient calculations. Preserve causal history and the shared total-energy acceptance rule. All bounded backtracking attempts currently run even when an earlier attempt succeeds; any early exit needs a measurement of its synchronization and batching costs.

   Choose round budgets from validation quality-versus-runtime curves. Relative energy change, gradient size, and probability changes can serve as stopping features, but a flat or low energy is not a correctness certificate. [Snell et al.](https://arxiv.org/html/2408.03314v1) show that effective inference-compute allocation depends on problem difficulty and the underlying model; their reasoning experiments are not an REWT speedup result.

   Implicit differentiation is a later memory experiment only after measuring actual equilibrium residuals and checking the required local conditions. Two successful descent rounds do not establish a fixed point. [Bai, Kolter and Koltun, Deep Equilibrium Models](https://proceedings.neurips.cc/paper_files/paper/2019/file/01386bd6d8e091c2ab4c7c7de644d37b-Paper.pdf)

7. **Revise gradient filtering as a separate experiment.** The current FFT filter replaces the raw parameter gradient with a low-pass reconstruction. Grokfast instead adds an amplified slow component to the current gradient; its ablation finds that replacing the original gradient with the slow component can destabilize training. [Lee et al., Sections 2.1 and 2.3](https://arxiv.org/html/2405.20233v1)

   Compare the original gradient, clipping, the existing replacement filter, and a residual mixture such as \(g'=g+\lambda\operatorname{LPF}(g)\), with \(\lambda\) selected on validation. Here \(g\) is the current parameter gradient and LPF is a causal low-pass filter over training steps. Grokfast addresses delayed generalization, not a general guarantee of spike removal. The pilot's main filter comparison has no preset spikes, so this remains lower priority than correcting output geometry and energy/prediction alignment. Weight EMA is another distinct option; it must not be confused with gradient EMA.

The implementation order should be output geometry, prediction-repair training, variable refinement depth, and a longer controlled core run. Frequency-capacity, support-module, and gradient-filter experiments follow once those changes are evaluated individually. A revised model should show better validation prediction and generated reasoning at useful cost, with improved text judged on a fresh blinded sample. The final test sample must be newly declared because the original test results already informed these choices.

The executable read-only probe is [checkpoint_diagnostics.py](/home/kt/Documents/EnergyTransformer/reports/rewt_research_20261009/checkpoint_diagnostics.py). The recommendations are pending implementation and controlled retraining; these checkpoint probes establish problems, not performance gains from the proposed fixes.
