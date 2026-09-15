# ASSB Digital Twin — Physics-Informed FNO + DeepONet Surrogate

Physics-informed neural operator surrogate for coupled electro-chemo-mechanical
degradation in all-solid-state battery (ASSB) composite cathodes, built on the
finite-element formulation of **Taghikhani & Kee, *J. Mech. Phys. Solids* 198
(2025) 106060**.

**Live app:** [sfcptu6uax4kademyfsc8v.streamlit.app](https://sfcptu6uax4kademyfsc8v.streamlit.app)
**Source data / training pipeline:** this repository

---

## 1. What this is

A Fourier Neural Operator (FNO) + DeepONet surrogate trained on 600 finite-element
solutions of a coupled mechanics–electrochemistry–phase-field model. It predicts,
in under one second, the spatial fields (von Mises stress, phase-field damage ξ,
σ_xx) and scalar cell response (voltage, capacity) that the full FEM model
produces in hours.

The goal is not to replace the full-fidelity model, but to demonstrate that its
physics can be distilled into a lightweight surrogate suitable for real-time
use battery management system (BMS) firmware, onboard vehicle compute, or
edge nodes without commercial FEM software.

### What this demonstrates

- **Feasibility** a coupled three-physics FEM model (electrochemistry +
  mechanics + fracture) that takes hours per case can be distilled into a
  surrogate that runs in under a second, from only 600 training samples.
- **Physical fidelity, not curve-fitting** seven independent physics checks
  (monotonicity, voltage bounds, fracture location, C-rate ordering, etc.) all
  pass, and the model's failure modes trace back to specific, named modeling
  choices rather than unexplained error (§5).
- **Deployability** under 5 MB and sub-second inference is small enough for
  BMS firmware or vehicle-onboard compute, closing part of the gap between
  a research-grade FEM model and a real-time control-loop-ready component.
- **Traceable limitations** every gap between this surrogate and the
  reference paper is tied to a specific, documented modeling or tooling
  choice, verified rather than assumed (see §5, including the pressure-
  sensitivity investigation).

Note on methodology: FNO and DeepONet are established neural-operator
architectures (Anandkumar et al. 2020; Lu et al. 2019), not introduced here.
What this repository contributes is their careful, verified application to
this specific coupled-physics problem — including an explicit accounting of
where and why the surrogate departs from the full-fidelity reference.

## 2. Architecture

| Component | Details |
|---|---|
| Spatial fields (vm, ξ, σ_xx) | FNO — 24 modes, 64 channels, 6 layers, ~1.2M params |
| Scalars (V_cell, capacity) | DeepONet — 128 hidden, 4 layers, ~50K params |
| Uncertainty | Monte Carlo Dropout (p=0.15, N=200 at inference) |
| Framework | JAX + Flax, trained on Google Colab T4 GPU |
| Deployed size | < 5 MB, < 1 s inference |

## 3. Physical validation

| Test | Expected | Result | Status |
|---|---|---|---|
| ξ at τ=0 | ≈ 0 | 0.004 | Pass |
| ξ at τ=1 | > 0.5 | 0.892 | Pass |
| Voltage at cutoff | 4.2 V | 4.200 V | Pass |
| Capacity ordering | 0.02C > 0.1C > 0.5C | 248 > 188 > 80 mAh/g | Pass |
| Pressure reduces damage | P↑ → ξ↓ | Confirmed (weak magnitude, see §5) | Pass |
| Fracture location | NMC–LPSC interface | Confirmed spatially | Pass |
| Damage monotonicity in τ | Non-decreasing | Confirmed | Pass |

R² against held-out FEM data: vm 0.944, ξ 0.965, σ_xx 0.935, V_cell 0.9999,
capacity 0.9998.

## 4. Comparison with the reference paper

| Quantity | Paper | This work | Gap |
|---|---|---|---|
| Capacity at 0.1C | ~180 mAh/g | 188 mAh/g | 4% |
| Cutoff voltage | 4.2 V | 4.200 V | Exact |
| Starting voltage at 0.1C | ~3.55 V | ~3.00 V | 0.55 V offset |
| ξ_max at end of charge | 0.7–1.0 | 0.892 | Consistent |
| Fracture location | NMC–LPSC interface | NMC–LPSC interface | Reproduced |
| Pressure effect on ξ | Strong (near-total suppression) | Present, weak | See §5 |

## 5. Documented simplifications and their consequences

Every departure from the paper's formulation was a deliberate choice made after
hitting a concrete tooling constraint, not an oversight. Each has a known
physical consequence.

| Choice | Consequence | What resolves it |
|---|---|---|
| Galvanostatic instead of Butler–Volmer | 0.55 V voltage offset; pressure does not couple into V_cell/capacity, since the stress term in the overpotential equation (η = Φ_ed − Φ_el − E^eq − β·σ_ij/F) is absent | Calibrated i₀ from EIS + coupled nonlinear solver |
| AT-2 phase-field instead of AT-1 | AT-1 requires solving a constrained minimization problem (irreversibility + threshold constraints) at every load step — infeasible with the available scikit-fem + JAX toolchain on a single T4 GPU, which is why AT-2's unconstrained formulation was used instead. Consequence: no nucleation threshold → damage grows smoothly instead of being gated. This is the primary reason pressure's effect on ξ_max and vm_max is weak (~1–1.5% over the full 0–45 MPa range) instead of the near-total suppression seen in the paper's Fig. 9 vs. Fig. 10, where pressure works mainly by preventing crack *nucleation* in the first place | Constrained minimization solver (COMSOL) or FEniCSx + PETSc on HPC |
| tanh ROM instead of history variable | Valid for a single charge event; cannot accumulate damage across cycles | Segregated solver tracking max_t(ψ₀⁺) per element |
| 5 circular particles instead of SEM geometry | Correct physics, no particle-size statistics or morphology | SEM image + HPC mesh generation |
| scikit-fem + JAX instead of COMSOL/FEniCSx | ~25K DOF vs ~3.5M DOF; coarser field contours | FEniCSx + PETSc + libCEED on HPC |

**On the weak pressure sensitivity specifically:** this was investigated in
depth, including a controlled test (fixed C-rate and τ, P varied from 5 to 35
MPa) and a direct check of the trained model's input normalization bounds
(`in_min`/`in_max` recovered from the saved model bundle). Both the input
scaling and the training pipeline were confirmed correct — the weak sensitivity
is a property of the AT-2 formulation itself (and its absence of a nucleation
threshold), inherited faithfully from the FEM training data, not an
implementation defect. AT-2 was chosen specifically because AT-1's constrained
minimization at every load step was not solvable with the available
scikit-fem + JAX toolchain on a single T4 GPU — the same category of
toolchain constraint documented for mesh resolution and geometry elsewhere in
this table.

## 6. Deployment profile

| | Current (laptop + Colab T4) | With HPC (COMSOL/FEniCSx) |
|---|---|---|
| DOF per solve | ~25,000 | ~500,000+ |
| Electrochemistry | Galvanostatic | Full Butler–Volmer + stress coupling |
| Phase-field | AT-2, tanh ROM | AT-1, full history variable |
| Geometry | 5 circular particles | SEM microstructure |
| Model size | < 5 MB | ~25–30 MB (estimated) |
| Inference time | < 1 s | < 1 s |
| BMS/EV deployable | Yes | Yes |

The deployment profile (size, speed) is a property of the network architecture,
not the training data moving to full-fidelity FEM data would close most of
the accuracy gap while keeping the same footprint and inference speed.

## 7. Running locally

```bash
pip install streamlit jax flax plotly scipy numpy
streamlit run app.py
```

Trained model weights are expected at `models/fno_bundle_final.pkl` and
`models/don_bundle_final.pkl`.

## 8. Citation

If referencing the underlying physics, please cite:

> Taghikhani, K., Huber, W., Weddle, P.J., Asle Zaeem, M., Berger, J.R., Kee, R.J.
> "Modeling coupled electro-chemo-mechanical phenomena within all-solid-state
> battery composite cathodes." *Journal of the Mechanics and Physics of Solids*,
> 198 (2025), 106060.

This repository is an independent surrogate-modeling reproduction of that work
and is not affiliated with the original authors.
