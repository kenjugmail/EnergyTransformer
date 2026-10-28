# Failed initial REWT pilot

This run is incomplete and must not be pooled with the later completed pilot.
The non-Fourier recurrent energy controls encountered non-finite objectives or
gradients after switching from proposal warm-up to joint refinement. Only the
Transformer language fits completed. The original frozen source, data plan,
partial checkpoints and update traces are preserved.

The learned feedforward recurrent proposal could amplify an unbounded previous
state across a 128-byte window. The corrective implementation uses tanh of the
previous recurrent state as conditioning for new guesses. This bounds an input
to the proposal network; it does not clip an accepted energy-refined state.
The correction is shared by all energy controls and the full model.

The new experiment uses fresh random initializations and new output/checkpoint
directories. No partially trained weights from this failed attempt are reused.
