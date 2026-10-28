# Recurrent Energy Transformer: Combining Attention and Energy-Guided Refinement

**Kenju Tomita**  
**Rochester Institute of Technology**  
**Email:** kt7432@g.rit.edu  
**Course:** CSCI 635: Introduction to Machine Learning  
**Project type:** Individual; proposed architecture and empirical study  
**Proposal date:** October 9, 2026

## Abstract

I propose the **Recurrent Energy Transformer (RET)**, combining my previous work on attention, recurrent energy modeling, iterative refinement, diffusion, forecasting, and specialist routing. This complete sequence model will be independently implemented and trained from scratch. RET will maintain state across tokens and refine each prediction using shared weights and learned energy gradients. Three-seed comparisons with attention and reduced-architecture controls will measure language prediction, associative recall, calibration, runtime, and memory. The contribution is the integrated architecture and an assessment of whether its components improve prediction and controlled reasoning tasks at a practical cost.

## 1. Introduction and Research Objectives

Transformers use attention to model relationships between tokens [1]. Energy-based models describe input/output compatibility and support prediction through optimization [2,3]. My earlier experiments explored recurrent energy state, local attention, and additional prediction-refinement modules separately. This project will combine those ideas into RET, a new proposed architecture whose contribution is their coordinated operation in one trainable model. The course implementation will be written independently in PyTorch and trained from random weights.

RET will have two recurrences: temporal state updates across tokens and repeated refinement of each prediction. Its research question is: **Does combining attention, energy-based state and prediction refinement, and specialized proposal mechanisms improve sequence prediction and recall enough to justify the added computation?** Published EBT work informs the energy-optimization component; the proposed RET additionally incorporates my recurrent backbone and experimental modules.

**Prior results (completed before this course project).** A three-seed CPU study of approximately 145,000-parameter models reported **2.5121 test bits per byte for my energy/local-attention hybrid**, compared with **2.5782 for full attention** and **2.5770 for local attention**, at 128-byte context and equal 90-second steady-state update budgets [5]. Compilation and evaluation were excluded. A historical 10.6-million-parameter comparison instead favored full attention: **1.6633 versus 1.7240 validation bits per byte for pure energy**, after 40.96 million training bytes [6].

Combining experimental modules has not yet demonstrated consistent gains. A three-seed, 24-update native-stack pilot reached **2.4265 test bits per byte**, versus **2.4247 for ordinary continuation** [7]. A separate small Transformer-energy decision adapter reached **76.95% synthetic-task accuracy**, while its combined variant reached **72.66%** [8]. These limited experiments motivate ablations; full-RET quality and general reasoning remain open questions.

## 2. Proposed RET Architecture and Training

**Complete architecture.** I will implement embeddings, positional information, normalization, feedforward blocks, the decoder, and the following integrated mechanisms:

**Attention/energy backbone.** Two recurrent energy blocks followed by one causal local-attention block will repeat through the network. Energy blocks will update learned precision/information state and perform learned reads; attention will preserve detailed nearby interactions. The final attention output and energy-state summary will condition the prediction workspace.

**Proposal and communication workspace.** Feedforward and shared-weight recurrent refiners, energy-gradient refinement, diffusion-generated candidates, and specialist blocks will propose alternative latent predictions. Forecast heads will predict future summaries from observed history, with a Fourier-history variant as a representation control. A learned latent-message channel will connect specialists. A critic/router will combine proposals with a keep-backbone option into initial candidate logits. All these components will be trainable.

**Full prediction-energy refinement.** A Transformer energy network will evaluate the context and candidate distribution, rather than functioning solely as an energy memory. For candidate logits \(z^{(r)}\), observed-prefix representation \(h_t\), and candidate probabilities \(p^{(r)}\), refinement is:

\[
p^{(r)}=\operatorname{softmax}(z^{(r)}),\qquad
E_\theta(h_t,p^{(r)})\in\mathbb{R}.
\]

\[
z^{(r+1)}=z^{(r)}-\eta\nabla_{z^{(r)}}E_\theta(h_t,p^{(r)}).
\]

The same energy network will be reused across steps, and the final candidate will produce next-byte probabilities. Observed state will be updated from actual input tokens; generated forecasts and candidate states will occupy a separate workspace. The proposed novelty is the joint RET composition and its training/evaluation design; established mechanisms will be credited to related work.

**Training.** I will train the backbone from random initialization, then enable the refinement modules and continue joint training. The primary loss will be next-byte cross-entropy, including supervision of refined predictions. Forecast heads will use future-summary targets from an immutable snapshot of the course-trained backbone, with invalid horizons masked; inference receives only observed history and predictions. Diffusion will use denoising supervision, while critic/router targets will use training-only candidate utility. Auxiliary-loss weights will be fixed before main experiments. Two initial energy-refinement steps will require differentiation through energy gradients; local detachment and a fully unrolled ablation will be recorded explicitly.

**Controls and data.** Four families at approximately 250,000 parameters and three seeds each will provide 12 main fits: full causal attention; attention/energy backbone alone; backbone plus Transformer energy refinement; and full RET. A second size near one million parameters is optional. I will use enwik8 [4], a 256-byte vocabulary, fixed 90M/5M/5M-byte splits, and 128-byte training context. Parameter counts will match within 0.5% where feasible, with actual counts and component costs reported. Training streams, effective batches, AdamW, and learning-rate search allowances will be paired; validation will choose settings before test evaluation.

## 3. Evaluation, Feasibility, and Timeline

**Hypotheses and evaluation.** H1: Full RET improves prediction loss against backbone and attention controls. H2: Extra refinement justifies its latency. H3: Proposal, forecast, and communication ablations identify useful components. Metrics are per-seed bits per byte, recall accuracy, negative log-likelihood, Brier score, and quality-versus-cost curves. Split-disjoint recall episodes with distractors and varying delays will use answer-only accuracy to measure controlled retrieval.

The core comparison matches parameters/data; a smaller follow-up will match end-to-end training time. Inference sweeps of 1/2/4/8 refinement steps will record energy/gradient calls, peak memory, carried state, setup, and latency. The initial router computes all candidates, so its routing decisions will not be assumed to save computation. Fixed nonoverlapping test windows and a declared reset policy will support evaluation. Longer-context results will be labeled extrapolation, and lower energy alone will not count as better prediction.

Ablations will remove diffusion, forecasting/Fourier features, or specialist messaging one at a time, starting at the smaller size. Training checks will cover causality, state resets, chunked/serial equivalence, mixed second derivatives, and restart. Nonfinite training or divergent refinement will stop the affected run, preserving failures. Model and checkpoint selection will use validation only.

**Feasibility.** Implementation will proceed through backbone, energy refiner, then proposal/forecast/specialist modules. The budget is 12 new GPU hours: two for verification/profiling, eight for training, and two for evaluation. Profiling will set data and ablation budgets with a 20% timing margin. Resource limits will reduce scale while preserving replication; the second size is optional. Earlier work informs the design; course code and weights will be new.

**Individual work plan.** I will implement, train, analyze, and write the project, credit sources and prior work, and arrange peer proofreading.

| Dates | Milestone and output |
|---|---|
| October 9 | Submit the RET proposal with prior results and the proposed combined architecture. |
| October 10–16 | Implement attention/energy backbone fixtures; present the proposal. |
| October 17–November 1 | Build energy refinement and experimental modules; verify training and freeze the protocol. |
| November 2–15 | Train the three-seed core grid; run recall tests and smaller ablations. |
| November 16–29 | Measure refinement/cost tradeoffs, calibration, and time-matched controls; freeze selections and test. |
| November 30–December 8 | Analyze results, obtain peer feedback, and finalize figures and reproducible artifacts. |
| December 9 | Submit the final report and required code/artifacts. |
| December 10–13 | Rehearse and deliver the final presentation. |

**Expected contribution.** Deliverables are an independently implemented RET, trained controls, reproducible artifacts, and evidence about component effects on quality, recall, calibration, and efficiency. Both favorable and unfavorable findings will inform its limits.

## References

1. Vaswani, A., et al. “Attention Is All You Need.” *NeurIPS*, 2017. [Conference proceedings](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
2. Gladstone, A., et al. “Energy-Based Transformers Are Scalable Learners and Thinkers.” *ICLR*, 2026. [Conference proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e19a65fd53b6f9a88b354da98813465d-Abstract-Conference.html).
3. Hoover, B., et al. “Energy Transformer.” *NeurIPS*, 2023. [Conference proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html).
4. Mahoney, M. “About the Test Data,” Large Text Compression Benchmark. [Dataset source and checksums](https://mattmahoney.net/dc/textdata.html).
5. EnergyTransformer project. “Compute-Controlled Confirmation.” Prior local report, September 2026. [Local report](../reports/compute_confirmation/REPORT.md).
6. EnergyTransformer project. “Reconciled Evidence and Repair Results.” Prior historical-results audit, September 2026. [Local report](../FINDINGS.md).
7. EnergyTransformer project. “Native Real-Text Pilot: Negative Result.” Prior experimental-stack study, September 2026. [Local report](../../EnergyTransformer-diffusion-state/reports/experimental_stack/real_text_pilot_001/REPORT.md).
8. EnergyTransformer project. “Calibration and Reproducibility Follow-up.” Prior synthetic decision-adapter study, September 2026. [Local report](../../EnergyTransformer-diffusion-state/reports/sentinel_reasoning/POLISH.md).
