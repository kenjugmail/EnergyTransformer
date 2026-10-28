# Recurrent Energy Transformer: Learning a Joint Fourier Energy Landscape

**Kenju Tomita**  
**Rochester Institute of Technology**  
**Email:** kt7432@g.rit.edu  
**Course:** CSCI 635: Introduction to Machine Learning  
**Project type:** Individual; proposed architecture and empirical study  
**Proposal date:** October 9, 2026

## Abstract

I propose the **Recurrent Energy Transformer (RET)**, a fully energy-based language model with a learned Fourier energy landscape. It will score all internal states and candidate predictions together. Training will learn its wave components and state priors; inference will minimize energy with weights fixed. RET will combine my previous attention-energy, diffusion, forecasting, and specialist mechanisms. I will implement it from scratch and test whether the spectral landscape improves prediction, recall, learning efficiency, and stability at a practical cost.

## 1. Introduction and Research Objectives

Language models predict what comes next from observed text. Transformers use attention to connect relevant information [1]; energy-based models score how well states or predictions fit their context [2,3]. RET will express attention, recurrence, communication, and prediction through one learned energy. Its **energy landscape** describes how that score changes as the states and candidate prediction change.

A **latent state** is an internal representation. Fourier wave components will describe the joint energy over these states and outputs. A finite representation will capture selected structure; it will not reconstruct every detail or guarantee a global minimum. Training learns this landscape; inference minimizes it with fixed weights. The training loss over weights is separate.

The research question is: **Can a learned Fourier representation of the full joint energy improve RET's prediction and learning efficiency?** Fourier features have helped low-dimensional function fitting [10]; their usefulness here remains unproven. My contribution will be the combined architecture and evaluation, crediting established methods. Efficient adaptation could benefit language-processing systems with limited computing resources.

**Prior results, completed before this project.** A three-seed CPU study of approximately 145,000-parameter models reached **2.5121 test bits per byte** for my energy/local-attention hybrid, versus **2.5782** for full attention and **2.5770** for local attention. Lower is better. Models used 128-byte context and equal 90-second training-update budgets, excluding setup and evaluation [5]. A historical 10.6-million-parameter study instead favored full attention: **1.6633 versus 1.7240 validation bits per byte** for pure energy after 40.96 million training bytes [6].

Combining mechanisms has not consistently helped. A three-seed, 24-update pilot reached **2.4265 test bits per byte**, versus **2.4247** for ordinary continuation [7]. A synthetic decision study reached **76.95% accuracy** with an energy refiner, versus **72.66%** with its combined variant [8]. These precursor experiments motivate comparisons; they do not establish fully energy-based RET performance.

## 2. Proposed Model and Learning Process

**Fully energy-based architecture.** I will independently implement RET in PyTorch. All recurrent states, attention-related states, specialist workspaces, and candidate output logits will form one joint state. A conditional scalar energy will govern their updates. Embeddings and proposal networks will supply context and initial guesses; accepted states and predictions will result from energy minimization.

**Learning the joint Fourier landscape.** Let \(s\) collect all state and output variables, and \(h_t\) the observed context. A starting energy family will be:

\[
E_\theta(h_t,s)=Q_\theta(h_t,s)+\sum_{k=1}^{K}a_k(h_t)\cos(\omega_k^\top s+\phi_k).
\]

Q will be a positive quadratic state prior. Modes will have learned amplitudes, phases, and frequency directions. Directions spanning blocks will encode attention, recurrence, forecast/specialist agreement, and output compatibility. Context-dependent amplitudes will use observed text and causal history. I will start with 16 modes and constrain frequency magnitudes. This finite energy family avoids exhaustive Fourier grids.

**Minimization during inference.** Each block will update its state using the partial gradient of the same total energy. Jointly, the update is:

\[
s^{(r+1)}=s^{(r)}-\eta\nabla_{s^{(r)}}E_\theta(h_t,s^{(r)}).
\]

In plain language: **learn the landscape, use its slope to revise every state and the prediction, and repeat.** Blocks will initialize from prior states and new input. Context coefficients can be cached; Fourier derivatives are directly computable. Softmax will normalize output logits. Smaller steps or backtracking will address instability. Costs will include conditioning and all energy/gradient evaluations.

**Spectral scale and noise.** Coarse-to-fine schedules will be tested against all modes throughout; accepted steps must reduce the full learned energy. A separate gradient-history filtering experiment will test spike reduction. Damping useful high-frequency information could harm quality. Fourier history [9] may condition the landscape.

**Combining my previous mechanisms.** Diffusion, forecasts, recurrent/feedforward proposals, and communicating specialists will initialize candidates and inform energy compatibility. Critic/routing mechanisms will parameterize the joint energy. All candidates will undergo minimization. Forecasts and output candidates will remain separate from observed-input state.

**Training.** Randomly initialized parameters will learn in stages, then jointly through two initial refinement rounds. Next-byte cross-entropy and a training-only energy-ranking loss will favor useful states and correct candidates over sampled alternatives. This includes learning the Fourier parameters. Forecasts will use frozen course-trained targets; diffusion will learn denoising. AdamW and auxiliary settings will use validation data. Inference will update states and predictions only.

**Data and controls.** enwik8 [4] will use 256 byte values, 90M/5M/5M-byte training/validation/test splits, and 128-byte context. Four families will compare a standard Transformer, non-Fourier energy RET, basic learned-Fourier RET, and full spectral RET with supporting mechanisms. Approximately 250,000 parameters and three seeds per family give 12 fits. Counts will match within 0.5% where feasible. Data, batches, optimizer, and tuning allowances will be paired.

## 3. Evaluation, Feasibility, and Timeline

**Research hypotheses.** H1: Learned-Fourier RET improves prediction and recall against attention and non-Fourier energy controls. H2: It reduces training time or inference cost at a given prediction quality. H3: Spectral damping reduces spikes without harming quality. H4: Component tests identify useful mechanisms. These claims remain to be tested.

**What will count as improvement?** Bits per byte will measure text prediction; lower is better. Recall accuracy will use distractors and varying delays; the Brier score will assess confidence. Learning and inference efficiency will compare held-out quality with training time, latency, gradient calls, and memory. Spike rates and gradient norms will measure stability, with thresholds set before final runs. Lower energy alone will not count as success.

**Testing each mechanism.** Ablations will compare learned versus fixed frequencies, Fourier versus equally sized non-Fourier energies, joint versus separate block modes, and full-band versus coarse-to-fine refinement. Gradient filtering will be compared with no filtering, clipping, and exponential smoothing. Other tests will remove proposal/forecast/communication mechanisms. Inference will use 1/2/4/8 rounds, varying state and output updates separately. All Fourier, energy, and proposal costs will be counted.

Comparisons will match model size and data; a smaller follow-up will match total training time. Validation will select settings before final testing. Checks will cover past-only inputs, state resets, consistent serial/chunked processing, and refinement gradients. Unstable runs will be recorded. Fixed test windows and reset rules will support comparison; recall will not establish general reasoning.

**Feasibility and individual work plan.** I will build the spectral energy and recurrent inference first, then add stability studies and remaining modules. The planned 12 GPU hours allocate two to verification/profiling, eight to training, and two to evaluation. Profiling will prioritize ablations with a 20% margin. Reduced scale will preserve replication. I will implement, experiment, analyze, and write, with peer proofreading. Course code and weights will be new.

| Dates | Milestone and output |
|---|---|
| October 9 | Submit the proposal. |
| October 10–16 | Build the joint Fourier energy model and present the proposal. |
| October 17–November 1 | Implement learned spectral modes, recurrent minimization, stability studies, and remaining modules. |
| November 2–15 | Train the three-seed comparisons; run recall and component tests. |
| November 16–29 | Measure learning efficiency and refinement costs; finish validation selection and final testing. |
| November 30–December 8 | Analyze results, prepare figures, obtain peer feedback, and finalize the report. |
| December 9 | Submit the final report and code/artifacts. |
| December 10–13 | Prepare and deliver the final presentation. |

**Expected contribution.** Deliverables will be fully energy-based RET, controls, and reproducible evidence about the benefits and limitations of a learned joint Fourier landscape.

## References

1. Vaswani, A., et al. “Attention Is All You Need.” *NeurIPS*, 2017. [Conference proceedings](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
2. Gladstone, A., et al. “Energy-Based Transformers Are Scalable Learners and Thinkers.” *ICLR*, 2026. [Conference proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e19a65fd53b6f9a88b354da98813465d-Abstract-Conference.html).
3. Hoover, B., et al. “Energy Transformer.” *NeurIPS*, 2023. [Conference proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html).
4. Mahoney, M. “About the Test Data,” Large Text Compression Benchmark. [Dataset source and checksums](https://mattmahoney.net/dc/textdata.html).
5. EnergyTransformer project. “Compute-Controlled Confirmation.” Prior local report, September 2026. [Local report](../reports/compute_confirmation/REPORT.md).
6. EnergyTransformer project. “Reconciled Evidence and Repair Results.” Prior historical-results audit, September 2026. [Local report](../FINDINGS.md).
7. EnergyTransformer project. “Native Real-Text Pilot: Negative Result.” Prior experimental-stack study, September 2026. [Local report](../../EnergyTransformer-diffusion-state/reports/experimental_stack/real_text_pilot_001/REPORT.md).
8. EnergyTransformer project. “Calibration and Reproducibility Follow-up.” Prior synthetic decision-adapter study, September 2026. [Local report](../../EnergyTransformer-diffusion-state/reports/sentinel_reasoning/POLISH.md).
9. Lee-Thorp, J., et al. “FNet: Mixing Tokens with Fourier Transforms.” *NAACL*, 2022. [Conference proceedings](https://aclanthology.org/2022.naacl-main.319/).
10. Tancik, M., et al. “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.” *NeurIPS*, 2020. [Conference proceedings](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html).
