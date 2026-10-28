# Long-run reasoning stability

All recorded optimizer updates completed with finite objectives and gradient components. The new full fits use float64 norm reduction and clip the applied vector to approximately norm one. Reused controls retain their original float32 norm diagnostics, with rounding differences up to 1.1e-5 at that boundary. Completion alone does not establish smooth learning: a large raw norm or loss remains a material diagnostic, even if clipping allows the next update. The full model bounds credit through carried latent states to sixteen observed tokens during training; forward memory is preserved.

| Variant / seed | Maximum raw gradient norm | Median raw norm | Clipped updates % | Maximum total loss |
|---|---:|---:|---:|---:|
| legacy_fourier / 101 | 4.34 | 0.519 | 3.5 | 4.02 |
| neural_energy / 101 | 4.76 | 0.651 | 7.2 | 4.63 |
| fourier_rewt / 101 | 4.68 | 0.656 | 6.4 | 4.6 |
| full_rewt / 101 | 1.49e+17 | 0.668 | 35.1 | 25.9 |
| legacy_fourier / 202 | 4.03 | 0.522 | 3.6 | 4.03 |
| neural_energy / 202 | 4.31 | 0.632 | 5.4 | 4.05 |
| fourier_rewt / 202 | 4.25 | 0.635 | 5.1 | 4.06 |
| full_rewt / 202 | 1.22e+20 | 146 | 68.9 | 99.8 |
| legacy_fourier / 303 | 4.11 | 0.518 | 3.2 | 4.39 |
| neural_energy / 303 | 5.1 | 0.623 | 6.4 | 4.35 |
| fourier_rewt / 303 | 13 | 0.627 | 5.4 | 4.41 |
| full_rewt / 303 | 2.14e+20 | 0.696 | 42.4 | 284 |
