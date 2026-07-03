---
title: "perfusion — Model & Code Reference"
subtitle: "Multiscale blood circulation and perfusion (Python port)"
author: "Based on the PhD thesis of U. N. A. Qohar (University of Bergen, 2023)"
---

# 1. Introduction

This document explains the mathematical models, numerical methods and software
structure of the `perfusion` package. It is meant as a standalone reference: a
reader can understand *what* each equation models, *how* it is discretised, and
*where* in the code it lives, without reading the source line by line.

The package is a clean Python re-implementation of the multiscale blood-flow and
perfusion models from the thesis *Modeling and simulation of blood circulation
and perfusion*. It couples a **1-D vascular network** (arteries and veins,
Poiseuille flow) to a **microcirculation model** — either a **Darcy porous
medium** (the *hybrid model*) or a **direct-connection network** — and adds
tracer transport and tracer-kinetic analysis for DCE-MRI.

## 1.1 The multiscale idea

Blood vessels span scales from large arteries down to capillaries a few microns
wide. Resolving every vessel is intractable, and imaging cannot see below
~250 µm. The strategy is therefore to model **observable vessels explicitly** as
a 1-D network and represent the **unresolved microvasculature** by an upscaled
continuum (Darcy) or by effective resistive links (direct connections). The two
scales are coupled so blood conserves mass across the interface.

```
        Arteries (1-D network)                Veins (1-D network)
                 │  Poiseuille + junction losses            ▲
                 ▼                                          │
        ┌──────────────────  Microcirculation  ──────────────────┐
        │  Darcy porous bed (hybrid)   OR   direct connections    │
        └─────────────────────────────────────────────────────────┘
```

## 1.2 Notation

| Symbol | Meaning | Units |
|---|---|---|
| $P,\ p$ | pressure | Pa (model units) |
| $Q,\ q$ | volumetric flow rate | mL·s⁻¹ |
| $R$ | hydraulic resistance | Pa·s·mL⁻¹ |
| $\mu$ | (blood/plasma) dynamic viscosity | Pa·s |
| $r,\ L$ | vessel radius, length | mm |
| $K$ | permeability of the capillary bed | mm² |
| $\phi$ | porosity | – |
| $H_D$ | discharge hematocrit | – |
| $\mathbf v$ | Darcy flux (volume rate per area) | mm·s⁻¹ |
| $c,\ C$ | tracer concentration (blood, tissue) | mM |

---

# 2. The vascular network model

Each vessel segment is a rigid cylinder in which blood flows as a steady,
laminar, incompressible Newtonian fluid (Reynolds number $\lesssim 200$,
Womersley number $\lesssim 0.1$, so inertia and pulsatility are negligible).

## 2.1 Poiseuille's law

Integrating the axial Navier–Stokes equation over a cylinder of length $L$ and
radius $r$ gives the **hydraulic resistance**

$$ R = \frac{8\,\mu\,L}{\pi\,r^{4}}, \qquad \Delta P = R\,Q . $$

This is the electrical-circuit analogue $\Delta P \leftrightarrow V$,
$Q \leftrightarrow I$. The $r^{4}$ dependence means small vessels dominate the
resistance.

> **Code.** `Network.resistance(viscosity)` returns $R$ per segment
> (`geometry.py`). The pressure–flow relation is assembled by the solvers in
> `network.py`.

## 2.2 Blood viscosity — the Fåhræus–Lindqvist effect

In vessels comparable in size to a red blood cell (~8 µm) the *apparent*
viscosity is no longer constant. The Secomb–Pries relation makes it a function
of vessel diameter $d = 2r$ (in µm):

$$
\mu(d) = \mu_0\!\left[1 + (\mu_{0.45}-1)\,
      \frac{(1-H_D)^{C}-1}{(1-0.45)^{C}-1}
      \left(\frac{d}{d-1.1}\right)^{2}\right]
      \left(\frac{d}{d-1.1}\right)^{2},
$$

$$
\mu_{0.45} = 6\,e^{-0.085 d} + 3.2 - 2.44\,e^{-0.06\,d^{0.645}}, \qquad
C = (0.8 + e^{-0.075 d})\!\left(-1 + \tfrac{1}{1+10^{-11}d^{12}}\right) + \tfrac{1}{1+10^{-11}d^{12}} .
$$

The curve is **U-shaped**: viscosity rises steeply for capillary-size vessels
and gently again for large vessels, with a minimum near ~50 µm.

> **Code.** `viscosity.apparent_viscosity(d, H_D, mu_0)`. Enabled per run with
> `Parameters.fahraeus_lindqvist=True` and `radius_to_um` (the model-radius →
> micron scale); `pipeline.run_hybrid` then feeds a *per-segment* viscosity to
> the resistance of every vessel.

## 2.3 Mass conservation and boundary conditions

At every internal (bifurcation) node, flow is conserved:

$$ \sum_{\text{in}} Q = \sum_{\text{out}} Q . $$

The trees are driven by **Dirichlet pressures** at the roots: $p = p_{\text{in}}$
at the arterial root and $p = p_{\text{out}}$ at the venous root.

## 2.4 Incidence-matrix form (how the code assembles the system)

Let $S$ be the signed segment–node **incidence matrix**,
$S_{s,\text{start}(s)}=+1$, $S_{s,\text{end}(s)}=-1$. Then for the pressure
vector $\mathbf p$ and flow vector $\mathbf q$:

$$ (S\mathbf p)_s = p_{\text{start}} - p_{\text{end}}
  \quad\Longrightarrow\quad S\mathbf p - R\,\mathbf q = 0 \ \ (\text{Poiseuille}), $$
$$ (S^{\top}\mathbf q)_{\text{node}} = \text{net outflow}
  \quad\Longrightarrow\quad (S^{\top}\mathbf q)_{\text{bifurcation}} = 0 \ \ (\text{conservation}). $$

Veins carry flow *toward* the root, so their Poiseuille row uses $-S$:
$p_{\text{end}} - p_{\text{start}} = R\,q$.

> **Code.** `Network.incidence()` builds $S$ as a sparse matrix; `network._assemble`
> stacks the Poiseuille, conservation, coupling and balance blocks. This replaces
> the per-node loops of the original MATLAB and scales linearly with mesh size.

---

# 3. Fractal-tree generation

Real vascular trees follow optimality principles. The generator builds a
self-similar binary tree from the root outward.

## 3.1 Murray's law and bifurcation geometry

At a bifurcation with asymmetry ratio $\gamma = (r_1/r_2)^2$ and **bifurcation
exponent** $\xi$ (Murray: $r_0^{\xi} = r_1^{\xi} + r_2^{\xi}$, with
$2 \le \xi \le 3$, $\xi=3$ for laminar flow), the child/parent radius ratios are

$$ \delta = \big(1 + \gamma^{\xi/2}\big)^{-1/\xi}, \qquad \beta = \delta\sqrt{\gamma}, $$

so $r_1 = \delta\, r_0$, $r_2 = \beta\, r_0$. The optimal branch angles are

$$ \phi_1 = \arccos\frac{1+\delta^4-\beta^4}{2\delta^2}, \qquad
   \phi_2 = \arccos\frac{1-\delta^4+\beta^4}{2\beta^2}. $$

Segment length is taken proportional to radius, $L = (\text{length-to-radius})\cdot r$.
Branching stops when a segment radius falls below $r_{\min}$.

> **Code.** `parameters.bifurcation_geometry(gamma, exponent)` returns
> $(\phi_1,\phi_2)$ and $(\delta,\beta)$; `geometry.grow_tree` grows the tree and
> `arterial_tree` / `venous_tree` are the convenience wrappers.

---

# 4. The capillary bed — Darcy porous medium

The unresolved microvasculature is homogenised into a **porous medium** on a
structured 2-D grid.

## 4.1 Governing equation

Darcy's law with mass conservation (incompressible, gravity neglected):

$$ \mathbf v = -\frac{K}{\mu}\nabla P, \qquad
   -\nabla\!\cdot\!\Big(\frac{K}{\mu}\nabla P\Big) = q . $$

## 4.2 Two-point flux approximation (TPFA)

Cells are numbered column-major. The flux across the face between adjacent cells
$i,j$ is $\tau_{ij}(P_i-P_j)$ with **transmissibility**

$$ \tau_{ij} = \frac{K}{\mu}\,\frac{A_{\text{face}}}{h}
   \quad\text{(harmonic mean; constant for a uniform homogeneous grid).} $$

Summing fluxes over each cell gives the sparse linear system

$$ \sum_{j\in\mathcal N_i}\tau_{ij}\,(P_i - P_j) = q_i, $$

i.e. $A_D\,\mathbf P = \mathbf q$ with $A_D$ symmetric. The operator is singular
(pure Neumann); one cell is pinned to fix the datum.

> **Code.** `darcy.transmissibilities`, `darcy.tpfa_operator`. The scheme is
> consistent on the quadrilateral grids used here.

## 4.3 Domain mask (organ shape)

Only cells inside the organ are *active*. A face is conductive **only if both
adjacent cells are active**, and inactive cells are given identity rows, so the
active region is hydraulically isolated (no flow leaks through the surrounding
box). Flux and time-of-flight are undefined (NaN) on inactive cells.

> **Code.** `Grid.mask`, applied in `darcy.transmissibilities` / `tpfa_operator`.

## 4.4 Point-source regularisation (the mollifier)

A 1-D vessel terminal injects flow at a *point*, whose Darcy solution is singular.
It is regularised by a compactly supported bump (thesis Eq. 4.20)

$$ f(\mathbf x) = C\,\exp\!\Big(\!-\frac{1}{1-|\mathbf x|^2}\Big)\ \text{for}\ |\mathbf x|<1,\quad 0 \text{ otherwise}, $$

rescaled to a finite radius $\varepsilon$ and normalised so its weights sum to 1
over the (masked) support. Each terminal $l$ thus contributes a distributed
source $q_l^{\varepsilon}(\mathbf x)=\int q_l\,f^{\varepsilon}(\mathbf x-\mathbf x_l)$.

> **Code.** `darcy.mollifier`.

---

# 5. The hybrid model — coupling network and Darcy bed

## 5.1 Terminal response (Schur complement)

Rather than solving the large Darcy system inside every network iteration, the
Darcy block is pre-eliminated. For unit flux injected at terminal $j$ (mollified,
extracted at the pin) the bed pressure field is $\mathbf{pvec}_j = A_D^{-1}\mathbf s_j$.
The **terminal-response matrix** collects the pressure felt at terminal $i$ per
unit flux at terminal $j$:

$$ \mathcal P_{ij} = \mathbf w_i^{\top}\,\mathbf{pvec}_j . $$

The full field for a set of terminal fluxes $\mathbf f$ is recovered by
superposition, $P(\mathbf x)=\sum_j f_j\,\mathbf{pvec}_j$.

> **Code.** `darcy.terminal_response` (batched LU back-substitution over all
> terminals) and `darcy.reconstruct_field`.

## 5.2 Coupling equations and the closed loop

Pressure continuity between the 1-D terminals and the 3-D bed reads

$$ p_{\text{terminal }i} - p_{\text{ref}} - \sum_j \mathcal P_{ij}\, f_j = 0, $$

where $p_{\text{ref}}$ is a floating Darcy datum. Arterial terminals inject
($f>0$), venous terminals extract ($f<0$). Because the pinned Green's function
would otherwise leak flux through the pin, a **global mass-balance row** closes
the loop:

$$ \sum_j f_j = 0 \quad\Longleftrightarrow\quad
   \text{(arterial inflow)} = \text{(venous outflow)} . $$

The monolithic linear system (unknowns
$[\,\mathbf p_a\ \mathbf q_a\ \mathbf p_v\ \mathbf q_v\ p_{\text{ref}}\,]$) is

$$
\begin{bmatrix}
 \text{root BCs}\\ \text{Poiseuille } (S)\\ \text{conservation } (S^\top)\\
 \text{terminal coupling } (\mathcal P)\\ \text{mass balance}
\end{bmatrix}\mathbf x = \mathbf b .
$$

> **Code.** `network.solve_linear` (assembles and solves with sparse LU).

## 5.3 Nonlinearity — bifurcation pressure loss

0-D/1-D models lose accuracy at junctions. A Bernoulli-based correction (thesis
Eq. 4.11) is added to each child branch $n$ of a bifurcation fed by the *datum*
vessel:

$$
\Delta P^{b}_{n} = \frac{\rho\,q_{dat}^{2}}{2\pi^{2} r_{dat}^{4}}
\left(1 + \frac{q_{n}^{2} r_{dat}^{4}}{q_{dat}^{2} r_{n}^{4}}
        - \frac{2\,q_{n}\,r_{dat}^{2}}{q_{dat}\,r_{n}^{2}}\cos\!\big(\tfrac34\theta_{dat,n}\big)\right).
$$

The total drop across a segment is $\Delta P_n = \Delta P^{h}_n + \Delta P^{b}_n$
(hydraulic + bifurcation). Vessel-wall elasticity is an optional further
nonlinearity (thesis Eq. 4.9).

## 5.4 Solving the nonlinear system — factorisation-reusing fixed point

The junction loss is a **mild, flow-dependent correction** to the Poiseuille
rows. Writing the linear system as $A\mathbf x = \mathbf b$, the nonlinear
problem is $A\mathbf x = \mathbf b + \mathbf c(\mathbf x)$, where
$\mathbf c$ places the junction drops on the right-hand side. It is solved by the
fixed-point iteration

$$ \mathbf x^{(k+1)} = A^{-1}\big(\mathbf b + \mathbf c(\mathbf x^{(k)})\big), $$

which **reuses a single LU factorisation of $A$** (the matrix never changes). This
is far cheaper than a dense finite-difference Newton and reduces exactly to the
linear solve when losses are disabled.

> **Code.** `network.solve_nonlinear`. The original thesis used Powell's dogleg /
> Levenberg–Marquardt; the fixed point is a numerically cheaper equivalent here.

---

# 6. The direct-connection microcirculation (alternative to Darcy)

Instead of a continuum bed, each arterial terminal is joined to nearby venous
terminals (within an effective radius $\rho_c$) by **direct connections** — extra
resistive links that emulate the unresolved arteriole–capillary–venule path.

## 6.1 Relative capillary resistance

A connection's resistance is the Poiseuille resistance scaled by a factor
$\gamma$ that accounts analytically for the auxiliary branching it represents
(thesis Eqs. 4.23–4.34):

$$ P_a - P_v = \gamma\,\frac{8\mu L_k}{\pi r_k^4}, \qquad
   \gamma_n = \frac{\sum_{i=1}^{n} 2^{(3i-1)/3}}{\sum_{i=1}^{n} 2^{(1-i)/3}} . $$

For $n=1$ generation, $\gamma = 2^{2/3}\approx1.587$. Connection radii follow
Murray's law, $r_k = r_t\,N_c^{-1/3}$, with $N_c$ the number of connections
leaving arterial terminal $t$.

> **Code.** `direct.relative_capillary_resistance`, `direct.connect_terminals`
> (radius search via `scipy.spatial.cKDTree`), `direct.solve_direct` (same
> incidence-matrix assembly as the hybrid model, closed by the connection links).

---

# 7. Post-processing: flux and time-of-flight

## 7.1 Darcy flux magnitude

Face fluxes $F_x,F_y$ are averaged to cell centres, giving
$|\mathbf v| = \sqrt{q_x^2 + q_y^2}$.

## 7.2 Time-of-flight

The **time-of-flight** $T(\mathbf x)$ — the time for a passive particle to travel
from an arterial terminal to $\mathbf x$ — satisfies the steady advection equation

$$ \mathbf v\cdot\nabla T = \phi, $$

discretised with a **first-order upwind** finite-volume scheme. This yields a
matrix that, after flow-direction reordering, is lower-triangular and solved
directly. A homogeneous TOF field indicates evenly perfused tissue (used as an
optimisation target in §9).

> **Code.** `darcy.upwind_operator` (shared connectivity),
> `timeofflight.flux_magnitude`, `timeofflight.time_of_flight`.

---

# 8. Tracer transport and tracer-kinetic models

## 8.1 Transport equation

A contrast agent (tracer) is advected passively with the blood (diffusion
negligible):

$$ \frac{\partial(\phi\, c)}{\partial t} + \nabla\!\cdot(c\,\mathbf v) = q_c . $$

On the grid this becomes an ODE $\phi\,\dot{\mathbf C} = A_{\text{adv}}\mathbf C +
\mathbf s_{\text{in}}\,c_a(t) - \mathbf s_{\text{out}}\mathbf C$, integrated with
the **theta rule** (thesis Eq. 5.7):

$$ \mathbf C^{k+1} = \mathbf C^{k} + \Delta t\big[\theta\, S(\mathbf C^{k+1}) + (1-\theta)\,S(\mathbf C^{k})\big]. $$

$\theta = 0$ is explicit Euler (conditionally stable — a CFL limit
$\Delta t \le \min_i \phi V_i / \text{outflux}_i$ is checked); $\theta=\tfrac12$
Crank–Nicolson; $\theta=1$ implicit Euler (unconditionally stable). For
$\theta>0$ a single LU factorisation of $I-\theta\Delta t\,L$ is reused each step.

> **Code.** `transport.build_advection`, `transport.transport_forward(theta=…)`,
> `transport.cfl_limit`.

## 8.2 Tracer-kinetic models (DCE-MRI)

Tissue concentration relates to the arterial input function (AIF) $c_a$ through an
impulse response, $C(t) = I(t) \ast c_a(t)$. The package implements the four
classic models compared in the thesis:

| Model | Impulse / estimate | Parameters |
|---|---|---|
| **1CM** (one-compartment) | $C = F_p\,e^{-t F_p/v_p} \ast c_a$ | $F_p, v_p$ |
| **MS** (maximum slope) | $F_0 = \dfrac{\max_t C'(t)}{\max_t c_a(t)}$ | $F_p$ |
| **2CXM** (exchange) | $I = A e^{x_1 t} + (1-A) e^{x_2 t}$ | $F_p, v_p, PS, v_e$ |
| **2CUM** (uptake) | $C = F_p\big[(1-E_F)e^{-t/T_p} + E_F\big]\ast c_a$ | $F_p, v_p, PS$ |

with $T_p = v_p/(F_p+PS)$, $E_F = PS/(F_p+PS)$, and the 2CXM exponents the roots
of the associated quadratic. The AIF is modelled as a gamma-variate
$c_a(t) = C_0\,(t-t_0)^A\,e^{-(t-t_0)/B}$.

Fitting is done by non-linear least squares. A network-flow simulation provides a
*ground-truth* tissue curve, so the models can be validated independently of the
data-generating assumption — the central idea of Paper C.

> **Code.** `tkm.one_compartment`, `tkm.two_compartment_uptake`,
> `tkm.two_compartment_exchange`, `tkm.maximum_slope`, `tkm.fit`,
> `tkm.gamma_variate_aif`.

---

# 9. Vascular-structure optimisation

The vascular tree is believed to be an optimal delivery network. The optimiser
varies the tree configuration (bifurcation exponent, asymmetry, angles, lengths,
root positions) to make delivery to the capillary bed as uniform as possible,
under two families of objective:

- **Flux homogeneity** — minimise range / standard deviation / $L_2$-norm / mean
  of the Darcy flux magnitude.
- **Time-of-flight homogeneity** — minimise the spread (std / max) of $T$, which
  reduces stagnation and favours even oxygen delivery.

> **Code.** `optimize.params_from_alpha` (maps a parameter vector to
> `Parameters`), `optimize.objective(kind=…)`. Trees that do not fit the grid
> return $\infty$ under `strict=True`.

---

# 10. Numerical conventions and design notes

- **Column-major (Fortran) cell ordering** throughout the grid, matching the
  MATLAB `reshape`/`sub2ind` convention: `linear = ix + iy·nx`.
- **0-based network indexing**; `incoming_segment[node]` is the feeding segment
  (−1 at the root); `node_role ∈ {ROOT, BIFURCATION, TERMINAL}`.
- **Sparse block assembly** via the incidence matrix (both hybrid and direct
  solvers); no per-node Python loops.
- **Coincident terminals** are nudged to distinct active cells so the coupling
  matrix stays non-singular.
- **Datum & mass balance**: the singular Darcy operator is pinned at one active
  cell, and terminal fluxes are rebalanced so the pin carries no net flow.

---

# 11. Concept → equation → code map

| Concept | Equation(s) | Module · function |
|---|---|---|
| Poiseuille resistance | $R=8\mu L/\pi r^4$ | `geometry.Network.resistance` |
| Fåhræus–Lindqvist viscosity | §2.2 | `viscosity.apparent_viscosity` |
| Bifurcation geometry (Murray) | §3.1 | `parameters.bifurcation_geometry` |
| Fractal tree growth | §3 | `geometry.grow_tree` |
| Darcy + TPFA | §4.1–4.2 | `darcy.transmissibilities`, `darcy.tpfa_operator` |
| Domain mask | §4.3 | `darcy.transmissibilities`, `darcy.tpfa_operator` |
| Mollifier point source | $f=Ce^{-1/(1-|x|^2)}$ | `darcy.mollifier` |
| Terminal response (Schur) | §5.1 | `darcy.terminal_response`, `reconstruct_field` |
| Hybrid coupling + balance | §5.2 | `network.solve_linear` |
| Junction pressure loss | §5.3 | `network.solve_nonlinear` (`_junction_drop`) |
| Fixed-point nonlinear solve | §5.4 | `network.solve_nonlinear` |
| Direct connections + $\gamma$ | §6 | `direct.*` |
| Flux & time-of-flight | §7 | `timeofflight.*`, `darcy.upwind_operator` |
| Tracer transport (theta rule) | §8.1 | `transport.transport_forward` |
| Tracer-kinetic models | §8.2 | `tkm.*` |
| Optimisation objectives | §9 | `optimize.objective` |
| End-to-end pipeline | §1–7 | `pipeline.run_hybrid` |

---

# 12. Reproducing the results

```bash
uv sync --extra image                          # install
uv run pytest                                  # physics + regression tests
uv run perfusion-tree                          # hybrid on a synthetic fractal tree
uv run perfusion-frog                          # hybrid on the frog-tongue organ domain
uv run python -m perfusion.examples.tkm_demo   # tracer-kinetic model comparison
uv run python scripts/generate_data.py         # regenerate bundled datasets
```

For deeper implementation notes and invariants, see `CLAUDE.md`; for the original
derivations, see the thesis (Papers A–D) and the references therein.
