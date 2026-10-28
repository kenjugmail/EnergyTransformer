# Recurrent Energy Wave Transformer: Learning a Joint Fourier Energy Landscape

**Kenju Tomita**  
**Rochester Institute of Technology**  
**Email:** kt7432@g.rit.edu  
**Course:** CSCI 635: Introduction to Machine Learning  
**Project type:** Individual; proposed architecture and empirical study  
**Proposal date:** October 9, 2026

## Abstract

I propose the **Recurrent Energy Wave Transformer (REWT)**, a fully energy-based language model that learns a Fourier description of its energy landscape. One energy function will score all internal states and candidate predictions together. Training will learn the wave components; inference will repeatedly revise states and predictions with trained weights fixed. REWT will combine my previous attention-energy, diffusion, forecasting, and specialist mechanisms. I will implement it from scratch and compare it with attention and non-Fourier energy models across three random seeds. Experiments will test prediction, recall, training stability, and quality relative to computational cost.

## 1. Introduction and Research Objectives

Language models predict what comes next from observed text. Transformers use attention to connect relevant information [1]; energy-based models score how well states or predictions fit their context [2,3]. REWT will express these relationships through one learned energy. Here, **energy** is a numerical score, and **wave** means Fourier sine/cosine components. The **energy landscape** describes how the score changes as internal states and candidate predictions change.

A **latent state** is an internal representation. REWT will learn Fourier components covering the joint energy over all states and outputs. A finite representation will capture selected structure, without guaranteeing an exact reconstruction or global minimum. **Training changes the parameters that define the landscape; inference follows its gradients to change states and predictions.** The training loss over weights is a separate objective.

The research question is: **Does a learned Fourier energy landscape improve prediction enough to justify its training and inference cost?** Fourier features have helped low-dimensional function fitting [10]; their usefulness here remains unproven. My contribution will be the combined architecture and evaluation, crediting established methods. Efficient state adaptation could benefit language-processing systems with limited computing resources.

**Prior results, completed before this project.** A three-seed CPU study of approximately 145,000-parameter models reached **2.5121 test bits per byte** for my energy/local-attention hybrid, versus **2.5782** for full attention and **2.5770** for local attention. Lower is better. Models used 128-byte context and equal 90-second training-update budgets, excluding setup and evaluation [5]. A historical 10.6-million-parameter study instead favored full attention: **1.6633 versus 1.7240 validation bits per byte** for pure energy after 40.96 million training bytes [6].

Combining mechanisms has not consistently helped. A three-seed, 24-update pilot reached **2.4265 test bits per byte**, versus **2.4247** for ordinary continuation [7]. A synthetic decision study reached **76.95% accuracy** with an energy refiner, versus **72.66%** with its combined variant [8]. These precursor experiments motivate comparisons; they do not establish fully energy-based REWT performance.

## 2. Proposed Model and Learning Process

**Fully energy-based architecture.** I will independently implement REWT in PyTorch. For each next-byte prediction, recurrent states, attention-related states, specialist workspaces, and output logits will form one joint state. Embeddings and proposal networks will provide observed context and initial guesses. Every block and output will then be updated through the same scalar energy.

**Learning the joint Fourier landscape.** Let \(s\) collect all state and output variables, and \(h_t\) the observed context. A starting energy family will be:

\[
E_\theta(h_t,s)=Q_\theta(h_t,s)+\sum_{k=1}^{K}a_k(h_t)\cos(\omega_k^\top s+\phi_k).
\]

Q will be a positive quadratic penalty anchoring states near input-informed values. Each mode will have a learned amplitude (strength), phase (shift), and frequency direction that mixes state coordinates. Shared modes will link attention, recurrent, forecast/specialist, and output states. Amplitudes will depend on observed context. I will start with 16 modes and constrain frequency magnitudes.

**Minimization during inference.** Each block will update its state using the partial gradient of the same total energy. Jointly, the update is:

\[
s^{(r+1)}=s^{(r)}-\eta\nabla_{s^{(r)}}E_\theta(h_t,s^{(r)}).
\]

In plain language: **start with a guess, score all states and the prediction, revise them toward lower energy, and repeat.** Prior refined states will carry information across tokens. Context coefficients can be cached, and Fourier derivatives computed directly. Softmax will normalize output logits. Backtracking will reject steps that raise the full energy. All evaluations will count toward cost.

**Spectral scale and noise.** I will test coarse-to-fine refinement against all modes throughout. Separately, filtering past gradient histories will test whether training spikes decrease. High-frequency information can be useful; validation will assess any damping. Fourier history [9] may condition the learned landscape.

**Combining previous mechanisms.** Diffusion, forecasts, recurrent/feedforward proposals, specialists, and critic/routing mechanisms will initialize candidates or parameterize energy compatibility. All candidates will undergo minimization. Forecasts and output candidates will remain separate from observed input.

**Training.** All parameters, including Fourier parameters, will start randomly. Staged then joint training will differentiate through two refinement rounds. Next-byte cross-entropy will penalize incorrect predictions; an energy-ranking loss will favor correct candidates over sampled alternatives. Training-only forecasts will use frozen course-trained targets; diffusion will learn denoising. AdamW will update weights; validation will select learning rates and auxiliary-loss settings. Inference will keep weights fixed.

**Data and controls.** enwik8 [4] will use 256 byte values, 90M/5M/5M-byte training/validation/test splits, and 128-byte context. Four families will compare a standard Transformer, joint energy with a non-Fourier neural score, basic Fourier REWT, and full REWT with supporting mechanisms. Basic energy controls will share states, priors, and conditioning. Approximately 250,000 parameters and three seeds per family give 12 fits. Counts will match within 0.5% where feasible. Data, batches, optimizer, and tuning allowances will be paired.

## 3. Evaluation, Feasibility, and Timeline

**Research hypotheses.** H1: Fourier REWT improves prediction and recall against attention and non-Fourier energy controls. H2: It reaches a given prediction quality with less training or inference time. H3: Training-gradient filtering reduces spikes without harming quality. Component tests will identify which mechanisms help. These are hypotheses to test.

**What will count as improvement?** Bits per byte will measure prediction error; lower is better. Recall accuracy will use distractors and varying delays. The Brier score will compare confidence with outcomes. Quality will be plotted against training time, inference latency, gradient calls, and memory. Spike rates and gradient norms will measure stability, using thresholds fixed before main runs. Lower energy alone will not establish better prediction.

**Testing each mechanism.** Ablations will remove or replace individual components: learned/fixed frequencies, Fourier/non-Fourier scores, coupled/separate block modes, and full-band/coarse-to-fine refinement. Gradient filtering will be compared with no filtering, clipping, and exponential smoothing. Smaller tests will remove proposals, forecasts, or communication. Inference will use 1/2/4/8 rounds, varying state and output updates separately. All Fourier, energy, and proposal costs will be counted.

Comparisons will match model size and data; a follow-up will match training time. Core tests will reset all carried states at the same 128-byte window boundaries. Validation will select settings before final testing. Checks will verify past-only inputs, resets, serial/chunked consistency, and refinement gradients. Unstable runs will be recorded; controlled recall will not establish general reasoning.

**Feasibility and individual work plan.** I will build the joint energy and recurrent inference first, then integrate the remaining mechanisms. The planned 12 GPU hours allocate two to verification/profiling, eight to training, and two to evaluation. Profiling will set a common training-byte budget with a 20% timing margin. Core comparisons take priority; smaller ablations depend on remaining time. I will implement, experiment, analyze, and write, with peer proofreading. Course code and weights will be new.

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

**Expected contribution.** Deliverables will be fully energy-based REWT, controls, and reproducible evidence about the benefits and limitations of a learned joint Fourier landscape.

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
