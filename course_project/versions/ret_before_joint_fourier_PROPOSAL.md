# Recurrent Energy Transformer: Fully Energy-Based Learning with Fourier Features

**Kenju Tomita**  
**Rochester Institute of Technology**  
**Email:** kt7432@g.rit.edu  
**Course:** CSCI 635: Introduction to Machine Learning  
**Project type:** Individual; proposed architecture and empirical study  
**Proposal date:** October 9, 2026

## Abstract

I propose the **Recurrent Energy Transformer (RET)**, a fully energy-based language model. Attention interactions, recurrent states, specialist communication, and predictions will be governed by one learned total energy. During inference, every block will refine its state while trained weights stay fixed. Fourier features may support faster refinement; Fourier filtering may also reduce training fluctuations. RET will combine my previous attention-energy, diffusion, forecasting, and specialist mechanisms. I will implement it from scratch and compare it with a standard Transformer and reduced energy-based models. Experiments will test prediction, recall, learning efficiency, stability, and inference cost.

## 1. Introduction and Research Objectives

Language models predict what comes next from observed text. Standard Transformers use attention to connect relevant parts of that text [1]. Energy-based models learn scores for how well states or predictions fit their context [2,3]. RET will express attention, recurrent memory, communication, and prediction as interacting terms in a total energy. Minimizing that energy will drive its computation. Energy here means a numerical compatibility score, not physical energy.

An internal or **latent state** is the model's hidden representation of the input. Every RET block will refine its state through energy minimization. Training will learn the energy functions; inference will adapt states and predictions with weights fixed. Embeddings and neural networks will parameterize energies and initialize states; their subsequent updates will follow the shared energy objective.

The research question is: **Can a fully energy-based recurrent model with energy-derived attention and Fourier features improve prediction, stability, and efficiency?** My contribution will be the combined architecture and evidence about its components; established methods will be credited [2,3,9]. Efficient adaptation could benefit language-processing systems with limited computing resources, but RET's efficiency remains unproven.

**Prior results, completed before this project.** A three-seed CPU study of approximately 145,000-parameter models reached **2.5121 test bits per byte** for my energy/local-attention hybrid, versus **2.5782** for full attention and **2.5770** for local attention. Lower is better. Models used 128-byte context and equal 90-second training-update budgets, excluding setup and evaluation [5]. A historical 10.6-million-parameter study instead favored full attention: **1.6633 versus 1.7240 validation bits per byte** for pure energy after 40.96 million training bytes [6].

Combining mechanisms has not consistently helped. A three-seed, 24-update pilot reached **2.4265 test bits per byte**, versus **2.4247** for ordinary continuation [7]. A synthetic decision study reached **76.95% accuracy** with an energy refiner, versus **72.66%** with its combined variant [8]. These precursor experiments motivate comparisons; they do not establish fully energy-based RET performance.

## 2. Proposed Model and Learning Process

**Fully energy-based architecture.** I will independently implement RET in PyTorch. The initial layout will repeat two recurrent-state energy modules and one local-attention energy module. All will contribute to a single scalar energy conditioned on the observed prefix. Attention and feedforward networks will define energy terms and starting states; state evolution and prediction will follow energy minimization.

**One shared energy, every state updated.** Total energy will sum recurrent-state consistency, masked query/key attention interactions, forecast and specialist compatibility, and output terms. Positive quadratic state priors will discourage drift. Each block will update its state using the partial gradient of this total energy, including its interactions with other states. Candidate output logits will also be optimized under the same energy; softmax will normalize them into next-byte probabilities.

Let \(s^{(r)}\) collect the hidden states and output logits, \(h_t\) the observed context, and \(f_t\) the Fourier features. A joint refinement step is:

\[
s^{(r+1)}=s^{(r)}-\eta\nabla_{s^{(r)}}E_\theta(h_t,f_t,s^{(r)}).
\]

In plain language: **score the whole system, revise its states and prediction toward lower energy, and repeat.** Here, \(\eta\) controls step size. Blocks will initialize from their previous refined states and new input; weights will be reused across steps. Finite refinement does not guarantee a global minimum or a correct prediction.

**Fourier features during inference.** Every block will transform its bounded past-state history along the time axis into sine/cosine coordinates. These features will condition energy terms and initialization, representing slow and fast variations using observed history only. Prior Fourier token mixing [9] motivates this representation; faster state adaptation remains a hypothesis.

**Fourier analysis during training.** A separate stability experiment will analyze rolling, past-only loss and gradient histories. Spectral filtering over training steps will damp selected rapid fluctuations in parameter updates. I will test whether it reduces loss and gradient spikes without harming validation quality. High-frequency variation will not automatically be labeled noise.

**Combining my previous mechanisms.** Diffusion, forecasts, and specialist communication will provide candidate states and energy-compatibility terms. Feedforward and recurrent proposal networks will initialize candidates; critic/routing terms will contribute to the shared energy. All candidates will undergo energy refinement. Forecasts and output candidates will remain in a separate workspace from observed-input state.

**Learning the energies.** From random weights, I will train in stages and then jointly. Next-byte cross-entropy through two initial refinement rounds will teach useful state updates. A training-only energy-ranking loss will favor correct output candidates over sampled alternatives. Forecast targets will use a frozen course-trained backbone; diffusion will learn denoising. AdamW and auxiliary settings will be selected on validation data. During inference, gradients will change states and predictions only.

**Data and main comparisons.** I will use enwik8 [4], 256 byte values, fixed 90M/5M/5M-byte training/validation/test splits, and 128-byte context. Four families will compare a standard Transformer, basic fully energy-based RET, basic RET with Fourier history, and full RET with the additional mechanisms. Approximately 250,000 parameters and three seeds per family give 12 fits. Sizes will match within 0.5% where feasible; actual counts will be reported. Data, batches, optimizer, and tuning allowances will be paired.

## 3. Evaluation, Feasibility, and Timeline

**Research hypotheses.** H1: Fully energy-based RET improves prediction and recall against standard attention and reduced energy controls. H2: Fourier history reduces inference steps or time at a given prediction quality. H3: Fourier filtering reduces training spikes without harming quality. H4: Component tests identify useful energy terms. These claims remain to be tested.

**What will count as improvement?** Bits per byte will measure text prediction; lower is better. Recall accuracy will use distractors and varying delays; the Brier score will assess confidence. Learning and inference efficiency will compare held-out quality with training time, latency, gradient calls, and memory. Spike rates and gradient norms will measure stability, with thresholds set before final runs. Lower energy alone will not count as success.

**Testing each mechanism.** Fourier history will be compared with raw history and a similarly sized learned transform. Training-filter ablations will compare no filtering, gradient clipping, and exponential smoothing at matched budgets. Other ablations will remove attention-energy terms, state/output refinement, diffusion, forecasts, or communication. Inference will use 1/2/4/8 rounds, varying state and output updates separately. All Fourier, energy, and proposal costs will be counted.

Comparisons will match model size and data; a smaller follow-up will match total training time. Validation will select settings before final testing. Checks will cover past-only inputs, state resets, consistent serial/chunked processing, and refinement gradients. Unstable runs will be recorded. Fixed test windows and reset rules will support comparison; recall will not establish general reasoning.

**Feasibility and individual work plan.** I will build the joint energy and recurrent inference first, then add Fourier features, stability experiments, and remaining modules. The planned 12 GPU hours allocate two to verification/profiling, eight to training, and two to evaluation. Profiling will set sizes with a 20% margin. A larger model is optional; reduced scale will preserve replication. I will implement, experiment, analyze, and write, with peer proofreading. Course code and weights will be new.

| Dates | Milestone and output |
|---|---|
| October 9 | Submit the proposal. |
| October 10–16 | Build the joint energy model and present the proposal. |
| October 17–November 1 | Implement recurrent minimization, Fourier features/filtering, and remaining modules; verify training. |
| November 2–15 | Train the three-seed comparisons; run recall and component tests. |
| November 16–29 | Measure learning efficiency and refinement costs; finish validation selection and final testing. |
| November 30–December 8 | Analyze results, prepare figures, obtain peer feedback, and finalize the report. |
| December 9 | Submit the final report and code/artifacts. |
| December 10–13 | Prepare and deliver the final presentation. |

**Expected contribution.** Deliverables will be fully energy-based RET, controls, reproducible experiments, and evidence about state adaptation and training stability. Results will identify useful mechanisms and their costs.

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
