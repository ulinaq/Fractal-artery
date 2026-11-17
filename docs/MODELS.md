# Models & formulas

The mathematical formulation behind every module in `perfusion.core` (and the
service layers `optimize`, `uq`, `inverse`). Each entry gives the equation, where it
lives in code, and the thesis equation number cited in the docstring (U. N. A. Qohar,
PhD thesis, University of Bergen, 2023 — Papers A–D). Symbols: `p` pressure, `q`/`Q`
volumetric flux, `r` radius, `L` length, `μ` viscosity, `K` permeability, `τ` wall
shear / transit time, `H_D` discharge hematocrit.

> Conventions the whole codebase relies on live in **CLAUDE.md** (column-major
> `order="F"` grids, 0-based network arrays, incidence-matrix assembly, degree-based
> node classification). This file is the *physics*; that file is the *bookkeeping*.

---

## 1. Vascular network — 1-D Poiseuille flow

**Hydraulic resistance** (`geometry.Network.resistance`, Poiseuille):

```
R = 8 μ L / (π r⁴)
```

**Flow / mass balance** assembled from the signed segment–node incidence `S`
(`network._assemble`, thesis Eq. 5.5 / 2.20). With `(S p)[s] = p_start − p_end`:

```
Poiseuille (per segment):   (S p) − R q = 0        →   Δp = R q
Mass conservation (node):   (Sᵀ q)[node] = 0       (net outflow zero at interior nodes)
```

Arterial and venous trees are stacked into one sparse block system `A x = b` with
unknowns `[p_a, q_a, p_v, q_v, p_ref]`; Dirichlet `p_in` at arterial inlets, `p_out`
at venous outlets. Vein flow `q_v` is positive toward the venous root, so the venous
Poiseuille row is `−(S p_v) − R q = 0`.

**Junction (Bernoulli) pressure loss** — the only nonlinearity, `network._junction_drop`,
thesis Eq. 4.11. For a daughter segment with parent flow `q_p`, child flow `q_c`,
datum radius `r_dat`, child radius `r_n` and branch angle `φ`:

```
Δp_junction = (ρ q_p² / (2 π² r_dat⁴)) · [ 1 + q_c² r_dat⁴/(q_p² r_n⁴)
                                            − 2 r_dat² cosφ · q_c/(q_p r_n²) ]
```

Solved by a **fixed-point** iteration (`solve_nonlinear`) that reuses one LU
factorisation of the linear operator; it reduces exactly to `solve_linear` when the
loss is disabled.

**Fåhræus–Lindqvist apparent viscosity** (`viscosity.apparent_viscosity`, Eqs. 4.5–4.7)
— viscosity rises with diameter `d` (µm) and hematocrit `H_D`:

```
μ₀.₄₅ = 6 e^(−0.085 d) + 3.2 − 2.44 e^(−0.06 d^0.645)          (Eq. 4.6, in-vitro ref.)
C     = (0.8 + e^(−0.075 d))(−1 + 1/P) + 1/P,  P = 1 + 1e−11 d¹²  (Eq. 4.7, shape)
corr  = (d/(d−1.1))²                                            (Eq. 4.5, wall layer)
μ_rel = (1 + (μ₀.₄₅ − 1)·((1−H_D)^C − 1)/((1−0.45)^C − 1)·corr)·corr
```

**Pries–Secomb phase separation** (RBC skimming at a bifurcation,
`hematocrit.phase_separation_fraction`, Pries–Secomb–Gaehtgens 1990). The fraction of
RBCs entering daughter α given its blood-flow fraction `FQ_B` (`d` in µm):

```
x₀ = 0.4/d_parent,   A = −6.96 ln(d_α/d_β)/d_p,   B = 1 + 6.98(1−H_D)/d_p
logit(FQ_E) = A + B · logit( (FQ_B − x₀)/(1 − 2x₀) )
```

with `FQ_E = 0` below `x₀` and `1` above `1 − x₀`. Propagated through the tree by
`segment_hematocrit`, then fed back into the F–L viscosity as a fixed point
(`run_hybrid(phase_separation=True)`).

---

## 2. Darcy capillary bed — TPFA finite volume

Single-phase Darcy flow on a structured grid (`darcy`, thesis Eqs. 4.16, 5.1–5.4):

```
−∇·(K/μ ∇p) = q
```

**Interface transmissibility** (`darcy.transmissibilities`, Eq. 5.1) — the harmonic
mean of the two adjacent cells' permeabilities, per axis (so a diagonal-tensor
`Kx ≠ Ky` anisotropy is exact):

```
K_face = 2 Kᵢ Kⱼ / (Kᵢ + Kⱼ)          (harmonic mean; 0 if either cell inactive)
T_x = (K_face/μ)·(h_y·thickness)/h_x,   T_y = (K_face/μ)·(h_x·thickness)/h_y
```

The TPFA operator `A` (`darcy.tpfa_operator`) is a 5-point Laplacian with `diag =
Σ faces` and `−T` off-diagonals; inactive cells get identity rows (`p = 0`), which
hydraulically isolates the organ. `A` is singular (pure Neumann) → pin one active cell.

**Mollifier** — regularised point source at a terminal cell (`darcy.mollifier`, the
thesis power-1 bump, Eq. 4.20 / 2.11), support radius `ε` (cell units), restricted to
the active mask and normalised to a partition of unity:

```
w(r) ∝ exp( −1 / (1 − (r/ε)²) )  for r < ε,  Σ w = 1
```

---

## 3. Hybrid coupling — network ↔ Darcy

The terminals are where the 1-D network meets the porous bed. `darcy.terminal_response`
solves the pinned TPFA system once (batched LU back-substitution) for every mollified
unit terminal source and returns the **terminal-to-terminal response** `Φ` — the Schur
block eliminated from the coupled system (thesis Eq. 5.5):

```
Φ[i,j] = pressure at terminal i per unit flux injected at terminal j
```

The coordinate-free network solver consumes `Φ`; the field is rebuilt by superposition
(`reconstruct_field`), rescaling arterial inflow to match venous outflow so the pin
carries no net flux:

```
p(x) = Σ_j Φ_field[x, j] · q_terminal[j]        (balanced)
```

Face fluxes from the reconstructed pressure: `q_face = T_face (p_i − p_j)`
(`darcy.face_fluxes`). `pipeline.run_hybrid` chains geometry → response → solve →
reconstruct → fluxes → time-of-flight.

---

## 4. Multi-compartment Darcy (arteriole → capillary → venule)

`core/multicompartment.py` (Michler / Hyde–Smith). `N` stacked porous compartments on
one grid, each with permeability `K_c`, coupled by a linear pressure exchange `β`:

```
−∇·(K_c ∇p_c) + Σ_{d} β_{cd}(p_c − p_d) = s_c
```

Block operator `M` (size `N·n`): diagonal blocks `L_c + diag(Σ β·V)`, off-diagonal
`−diag(β·V)` on active cells. Arterial terminals source compartment 0, venous drain
compartment `N−1`; the generalised `terminal_response_multi` returns the same `Φ` the
network solver already consumes, so **N=1 reduces exactly to the single-bed hybrid**.
The **capillary perfusion** is the inter-compartment exchange field:

```
perfusion_c(x) = β_c · V · (p_c(x) − p_{c+1}(x))
```

At steady state `Σ perfusion₀ = arterial inflow = venous outflow` (conservation).

---

## 5. Direct-connection model (Paper C)

Unresolved microvessels are resistive links from each arterial terminal to nearby
venous terminals (`direct`). Connection resistance is Poiseuille scaled by a relative
capillary resistance from an `n`-generation auxiliary tree (thesis Eq. 4.34):

```
γ_n = Σᵢ 2^((3i−1)/3) / Σᵢ 2^((1−i)/3)     (n=1 → 2^(2/3) ≈ 1.587)
R_k = γ_n · 8 μ L_k / (π r_k⁴)
```

Radii follow Murray's law `r_k = r_terminal · N_c^(−1/3)`. Solved with the same
incidence-block assembly as the hybrid, closing the loop `arterial in = venous out`.

**Network-transit kinetics** (`network_transit`) times a bolus along the vasculature:
element transit `τ = V/|q|`, arrival time accumulated from the inlet by a
pressure-ordered (flow-directed, loop-robust) sweep; each element lights with a
gamma-variate bolus delayed by its arrival.

---

## 6. Time-of-flight & tracer transport

**Time-of-flight** (`timeofflight`, upwind finite volume, Eqs. 17–21): the steady
`A t = 1` on the net-inflow connectivity, `t` = time for blood to reach each cell.
**Flux magnitude** `|v| = √(fx² + fy²)` at cell centres.

**In 3-D** the same post-processing runs on the voxel bed (`run_hybrid_3d` returns
`pressure`, `flux`, `tof`): `darcy3d.face_fluxes_3d` gives interior-face fluxes
`T (p_owner − p_neighbour)` per axis — **zero on faces touching an inactive cell**, so
nothing leaks across an organ-shaped mask — `flux_magnitude_3d` averages them to cell
centres `|v| = √(qx² + qy² + qz²)`, and `timeofflight.time_of_flight_3d` solves the same
upwind system on `darcy3d.upwind_operator_3d` with pore volume `V = hx·hy·hz·φ`
(`Grid3D.porosity`). Ordering check: TOF is shortest at the arterial injection cells,
longer at the venous drains, longest in stagnant tissue.

**Tracer transport** (`transport`, θ-rule, Eq. 5.7) — advection of a contrast bolus,
diffusion neglected:

```
V_pore dC/dt = A C + inject·c_a(t) − extract·C
C^{n+1} = C^n + dt[θ L C^{n+1} + (1−θ) L C^n] + dt·source·c_a
```

`θ=0` explicit Euler (CFL-limited, `dt ≤ min V/outflux`), `θ=½` Crank–Nicolson
(default), `θ=1` implicit Euler. One LU of `I − θ dt L` is reused for all steps.

---

## 7. Perfusion metrics — transit-time distribution

`core/perfusion_metrics.py`, over the TOF field `h(τ)`:

```
MTT = ⟨τ⟩                            (mean transit time)
CTH = √⟨(τ − MTT)²⟩                   (capillary transit heterogeneity, σ)
RD  = CTH / MTT                       (relative dispersion)
OEF = ⟨1 − e^(−k τ)⟩                  (Kety single-capillary extraction over h(τ))
```

Because `Q(τ)=1−e^{−kτ}` is concave, **higher CTH lowers OEF** at fixed MTT — the
Jespersen–Østergaard heterogeneity penalty. `k` auto-scales to `1/MTT` for a unit-free
index (≈0.63 for homogeneous flow).

---

## 8. Tracer-kinetic models (DCE/DSC-MRI, Paper C)

`core/tkm.py` maps an AIF `c_a(t)` to tissue concentration `C(t)` via an impulse
response ⊛ `c_a`:

```
1CM  (Eq. 3.16):  imp = Fp e^(−t Fp/vp)
2CUM (Eq. 3.31):  imp = Fp[(1−E_F) e^(−t/Tp) + E_F],  E_F = PS/(PS+Fp)
2CXM (Eq. 3.3x):  imp = Fp[A e^{x₁t} + (1−A) e^{x₂t}]  (bi-exponential)
MS:               Fp = maxₜ dC/dt / max(c_a)
AIF  (Eq. 11):    c_a(t) = C₀ (t−t₀)^A e^(−(t−t₀)/B)   (gamma-variate)
```

**Voxel-wise maps** (`kinetic_maps`): `Fp = maxₜ dC/dt / max(c_a)` (max-slope),
`vp = ∫C/∫c_a` (AUC blood volume).

**Residue deconvolution** (`deconvolve`, Østergaard) solves `C = A k` for the
flow-scaled residue `k = Fp·R`:
- **sSVD**: `A` lower-triangular Toeplitz (causal); truncate singular values below
  `oi·max(s)`.
- **oSVD** (`circulant=True`): `A` block-circulant on a `2N` zero-padded grid →
  **delay-insensitive** (a bolus arrival lag no longer biases `Fp`).
- Central-volume theorem: `Fp = max k`, `CBV = ∫k`, `MTT = CBV/Fp`.

---

## 9. Oxygen / solute transport & extraction (Krogh/Kety)

`core/oxygen.py` — steady advection–reaction(–diffusion) of a blood-borne solute:

```
∇·(v C) − D ∇²C + k C = delivery
```

Discretised on the upwind connectivity: `(A − diag(extract) − k_eff·diag(V) − D·L) C =
−inject·c_art`. `k` is a **dimensionless Damköhler number**: `k_eff = k · (inflow/volume)
= k/MTT`, so the sink scales with flow and the global extraction spans `(0,1)`. With
`k>0` the operator is a non-singular M-matrix → `C ≥ 0`, mass-conservative:

```
delivery = consumption + venous washout
OEF = consumption / delivery ≈ 1 − e^{−k}   (well-mixed limit; lower with transit heterogeneity)
```

---

## 10. Structural adaptation (Pries–Secomb → Murray's law)

`core/adaptation.py` — wall-shear-driven radius remodelling:

```
τ_w = 4 μ |Q| / (π r³)                          (Poiseuille wall shear)
S   = log₁₀(τ_w / τ_ref) + k_m·log₁₀(1 + Q_ref/|Q|) − k_s     (growth stimulus)
r ← r · exp(rate · S)                            (clamped to [0.1, 10]×r₀)
```

Iterated with a flow solve in the loop, this drives the `|Q| ∝ r^k` scaling exponent
toward **Murray's law (k = 3)**; the pure-shear stimulus over-differentiates past it, so
`run_adaptation` early-stops at the closest approach to the target exponent.

---

## 11. Morphometry — Horton–Strahler

`core/morphometry.py`. Leaves = order 1; where two or more order-`ω` branches meet the
parent becomes `ω+1`, else it keeps the max child order. Horton geometric laws fitted
across orders (`Ω` = max):

```
N_ω = R_B^(Ω−ω)      →  R_B = e^(−slope of log N_ω vs ω)    (bifurcation ratio)
L_ω = L₁ R_L^(ω−1)   →  R_L = e^(+slope of log L_ω vs ω)    (length ratio)
D_ω = D₁ R_D^(ω−1)   →  R_D = e^(+slope of log D_ω vs ω)    (diameter ratio)
```

A complete binary tree gives `R_B = 2` exactly; the fractal arterial tree gives
`R_B ≈ 2.1`, `R_D ≈ 1.4` (physiological).

---

## 12. Growth — space colonisation & CCO

`perfusion/growth.py`. **Space colonisation** (Runions 2007): each attraction point
pulls the nearest tree node; a node with pulls grows a segment of fixed `step` in the
mean pull direction; points within `kill_radius` are consumed. **CCO** adds terminals
one at a time, each connected to the nearest point on the tree (min-length), with a
`d_min` spacing constraint. Radii follow **Murray's law** at each bifurcation:

```
r_parent^γ = Σ r_child^γ           (γ = exponent, default 3)
```

**Data-constrained growth**: attraction points sampled with probability ∝ a target
density map (`sample_points_in_mask(density=)`); **obstacles** deactivate no-grow cells.

**Dimension-agnostic.** Sampling, colonisation and CCO work on 2-D *or* 3-D trees: the
dimension follows the base network (`node_z`) and the mask (`(nx, ny)` with a 4-value
window, or `(nx, ny, nz)` with a 6-value window); distances use `‖·‖₂` and the CCO
spacing default is `d_min = ½ (size / n_terminals)^(1/dim)`.

**3-D fractal skeleton** (`geometry.grow_tree_3d`). Each bifurcation splits in a
*branching plane* through the parent direction `d`; that plane's normal `n` is rotated
about `d` by the **twist** `θ_t` (default 90°) relative to the parent's plane,

```
n_child = R_d(θ_t + ε) n_parent,      d_k = R_n(±φ_k (1 + ε')) d,
r_k = ρ_k r,                          L_k = ρ_k L (1 + ε'')
```

with `(φ_k, ρ_k)` from Murray's law (`bifurcation_geometry(γ, ξ)`, or a symmetric
`±θ, ratio`) and seeded jitter `ε` (`jitter ∈ [0, 1]`). A fixed global rotation axis
(the old rule) kept every generation in one plane; rotating relative to the parent
plane makes the tree volumetric from the second generation
(`tests/test_geometry.py::test_grow_tree_3d_is_volumetric_not_planar`).
`grow_tree(max_generations=)` caps the 2-D fractal depth the same way.

**Pruning** (`geometry.prune_terminal_branches`) — the inverse of growth: one step
removes every *terminal branch* (a free end, back through degree-2 nodes, to its
nearest junction); chains reaching an inlet/outlet are never removed, loops have no free
end and are untouched. Radii of what remains are unchanged.

---

## 13. Transient — 2-element Windkessel

`core/transient.py`. A compliant arterial compartment `P(t)` fed through proximal `Rp`,
draining through distal `Rd` (`Rp = f·R`, `Rd = (1−f)·R`):

```
C dP/dt = (p_in − P)/Rp − (P − p_out)/Rd,     q_tissue = (P − p_out)/Rd
```

Solved by implicit Euler (unconditionally stable). `C→0` gives the quasi-static limit
`q_tissue = ΔP/R`; compliance damps the inlet pulse before it reaches the tissue.
Total compliance `C_total = 2π·compliance·Σ(r² L)`.

---

## 14. Analysis service layers

**Uncertainty (`uq.py`)** — Latin-hypercube sampling; first-order sensitivity
`corr(xᵢ, y)` (Pearson). **Sobol** via a degree-2 orthonormal-Legendre polynomial-chaos
surrogate: variance `V = Σ_{α≠0} c_α²`, `S1_i = Σ_{α: supp={i}} c_α²/V`,
`ST_i = Σ_{α: i∈supp} c_α²/V` (so `ST ≥ S1`), with the fit `R²`.

**Inverse / calibration (`inverse.py`)** — bounded 1-D search (`minimize_scalar`) over
forward solves to recover an input (`K`, `p_in`, `μ`, …) so a measured QoI matches a
target.

**Optimisation (`optimize.py`)** — minimise flux-/TOF-heterogeneity over the tree's
branching asymmetries (Nelder–Mead / differential evolution), Paper D.

---

## 15. Organ domains & anatomical roots

`perfusion/organs.py` (service layer, pure NumPy). Stylised, **connected** organ tissue
domains in unit coordinates, placed into a physical window: organs keep their
proportions (uniform scale `min(half-extents)`), generic shapes stretch to fill.

- **2-D outlines** (closed polygons → `rasterize_mask`): ellipse, rectangle, wedge,
  heart (frontal silhouette), kidney (with a hilum notch), brain (gyri + cerebellum),
  liver, lung, tongue, tumour (seeded lobulated blob), placenta (disc).
- **3-D masks** (implicit inside-tests on cell centres): box, ellipsoid, cylinder,
  hemisphere, heart (the myocardial **wall** — a tilted egg-shaped ventricular mass
  minus a closed cavity), kidney (bean minus hilum), brain (two hemispheres joined
  below the longitudinal fissure), liver (two lobes), lung (dome, concave base),
  tumour.

Every shape must stay **one connected region** at every resolution (a split mask leaves
a singular Darcy block) — `tests/test_organs.py`. `organs.roots(name, dim, domain)`
gives each organ's **anatomical** arterial inlet / venous outlet and trunk aim (renal
hilum, aortic root → apex, porta hepatis / hepatic veins, circle of Willis / sagittal
sinus, lung hilum, tongue root, cord insertion), tested so every trunk leads into the
tissue. `treegen` (explorer service) builds: skeleton from those roots →
optional fill → solve, in 2-D or 3-D, with `extent` scaling the whole window.

---

*See `docs/ROADMAP.md` for delivery status and `CLAUDE.md` for the architecture and
data-flow. Thesis equation numbers are preserved in the code docstrings — keep them in
sync when editing.*
