"""Pure 7-bit ASCII terminal visualizer for cheminformatics, PK, docking, and Pareto traces.

All outputs strictly use printable 7-bit ASCII characters (codes 32-126 plus
newlines) so headless CI runners and Windows cp1252 consoles never fail with
``UnicodeEncodeError``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from chemi_cascade.types import (
    ADMEPrediction,
    DescriptorProfile,
    DockingPose,
    MoleculeGraph,
    OptimizationTrace,
    PKProfile,
    ProtacCurve,
)


def assert_ascii(text: str) -> str:
    """Verify ``text`` contains strictly 7-bit ASCII characters (0-127)."""
    for idx, ch in enumerate(text):
        if ord(ch) > 127:
            raise ValueError(f"Non-ASCII character {ch!r} (ord={ord(ch)}) at index {idx}")
    return text


def _ascii_bar(val: float, max_val: float, width: int = 24, fill: str = "#") -> str:
    frac = max(0.0, min(1.0, val / max(max_val, 1e-9)))
    filled = int(round(frac * width))
    return "[" + (fill * filled) + ("-" * (width - filled)) + "]"


def render_descriptor_panel(
    mol: MoleculeGraph,
    desc: DescriptorProfile,
    murcko_smiles: str = "",
) -> str:
    """Render physicochemical, topological, and Gasteiger partial-charge summary."""
    lines: List[str] = []
    title = mol.name or desc.canonical_smiles
    lines.append("=" * 74)
    lines.append(f"MOLECULE PROFILE: {title}")
    lines.append("=" * 74)
    lines.append(f"  Canonical SMILES : {desc.canonical_smiles}")
    lines.append(f"  Murcko Scaffold  : {murcko_smiles or '<acyclic>'}")
    lines.append(f"  Formula / Heavy  : {desc.molecular_formula} ({desc.heavy_atom_count} heavy atoms)")
    lines.append(f"  Rings / Aromatic : {desc.ring_count} total / {desc.aromatic_ring_count} aromatic")
    ro5_str = "PASS" if desc.veber_pass else "FAIL"
    lines.append(f"  Lipinski / Veber : {desc.lipinski_violations} violations | Veber={ro5_str}")
    lines.append(f"  Alerts (PAINS)   : {', '.join(desc.alerts) if desc.alerts else 'NONE'}")
    lines.append("-" * 74)
    lines.append(f"  MW (g/mol)   {desc.molecular_weight:8.2f}  {_ascii_bar(desc.molecular_weight, 600.0)}")
    lines.append(f"  cLogP        {desc.clogp:8.3f}  {_ascii_bar(max(0.0, desc.clogp + 1.0), 7.0)}")
    lines.append(f"  TPSA (A^2)   {desc.tpsa:8.2f}  {_ascii_bar(desc.tpsa, 160.0)}")
    lines.append(f"  QED          {desc.qed:8.4f}  {_ascii_bar(desc.qed, 1.0, fill='*')}")
    lines.append(f"  SA Score     {desc.sa_score:8.3f}  {_ascii_bar(10.0 - desc.sa_score, 9.0)}")
    lines.append(f"  CNS-MPO      {desc.cns_mpo:8.3f}  {_ascii_bar(desc.cns_mpo, 6.0)}")
    lines.append("-" * 74)
    lines.append(
        f"  Topo Indices: Wiener={desc.wiener_index} | M1={desc.zagreb_m1} | "
        f"M2={desc.zagreb_m2} | Chi={desc.randic_chi:.3f} | J={desc.balaban_j:.3f}"
    )
    if desc.gasteiger_charges:
        q_min = min(desc.gasteiger_charges)
        q_max = max(desc.gasteiger_charges)
        q_sum = sum(desc.gasteiger_charges)
        lines.append(f"  Gasteiger Q : sum={q_sum:+.6f} e | min={q_min:+.4f} e | max={q_max:+.4f} e")
    return assert_ascii("\n".join(lines))


def render_similarity_matrix(
    names: Sequence[str],
    sim_matrix: NDArray[np.float64],
) -> str:
    """Render a compact 7-bit ASCII pairwise Tanimoto similarity heatmap."""
    ramp = " .:-=+*#%@"
    n = len(names)
    lines: List[str] = ["=" * 74, "PAIRWISE ECFP4 TANIMOTO SIMILARITY HEATMAP", "=" * 74]
    header = " " * 18 + " ".join(f"{i:2d}" for i in range(n))
    lines.append(header)
    for i in range(n):
        label = f"[{i:2d}] {names[i][:12]:<12}"
        cells = []
        for j in range(n):
            v = float(np.clip(sim_matrix[i, j], 0.0, 1.0))
            ch = ramp[min(len(ramp) - 1, int(v * (len(ramp) - 1)))]
            cells.append(f"{ch}{ch}")
        lines.append(f"{label} " + " ".join(cells))
    lines.append("  Legend: ' '=0.0  '.'=0.1  ':'=0.2  '-'=0.3  '='=0.4  '+'=0.6  '*'=0.7  '#'=0.8  '@'=1.0")
    return assert_ascii("\n".join(lines))


def render_pk_curve(
    pk: PKProfile,
    adme: ADMEPrediction | None = None,
    width: int = 58,
    height: int = 11,
) -> str:
    """Render a 2-compartment plasma concentration-time profile in 7-bit ASCII."""
    lines: List[str] = ["=" * 74, "2-COMPARTMENT PBPK PLASMA CONCENTRATION PROFILE", "=" * 74]
    if adme is not None:
        lines.append(
            f"  ADME: logS={adme.log_s:.2f} | logD7.4={adme.log_d_74:.2f} | "
            f"CLint={adme.cl_int_ul_min_mg:.1f} uL/min/mg | Papp={adme.papp_caco2:.1f} | "
            f"hERG pIC50={adme.herg_pic50:.2f}"
        )
    lines.append(
        f"  PK (Dose={pk.dose_mg:.0f} mg): Cmax={pk.cmax_ug_l:.1f} ug/L @ Tmax={pk.tmax_h:.2f} h | "
        f"t1/2={pk.half_life_beta_h:.2f} h | F={pk.bioavailability_f * 100:.1f}% | "
        f"Margin={pk.therapeutic_margin:.1f}x"
    )
    lines.append("-" * 74)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    t = pk.times_h
    c = pk.c_plasma_ug_l
    c_max = max(float(np.max(c)), 1e-6)
    t_max = max(float(t[-1]), 1e-6)

    for col in range(width):
        target_t = (col / max(1, width - 1)) * t_max
        val = float(np.interp(target_t, t, c))
        row = height - 1 - int(round((val / c_max) * (height - 1)))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"
        for r_below in range(row + 1, height):
            if grid[r_below][col] == " ":
                grid[r_below][col] = ":"

    for r in range(height):
        y_val = c_max * (height - 1 - r) / max(1, height - 1)
        lines.append(f"{y_val:7.1f} |{''.join(grid[r])}|")
    lines.append("        +" + "-" * width + "+")
    lines.append(f"        0.0 h{' ' * (width - 14)}{t_max:5.1f} h")
    return assert_ascii("\n".join(lines))


def render_protac_hook_curve(
    curve: ProtacCurve,
    width: int = 58,
    height: int = 11,
) -> str:
    """Render the bell-shaped PROTAC ternary complex hook-effect dose-response curve."""
    lines: List[str] = ["=" * 74, "PROTAC TERNARY COMPLEX HOOK-EFFECT DOSE-RESPONSE", "=" * 74]
    lines.append(
        f"  Peak [ABP]={curve.max_ternary_nm:.2f} nM | EC50={curve.ec50_formation_nm:.2f} nM | "
        f"Hook IC50={curve.hook_ic50_nm:.1f} nM | eff_alpha={curve.effective_alpha:.3f} | "
        f"dG_WLC={curve.wlc_entropy_penalty_kcal:.2f} kcal/mol"
    )
    lines.append("-" * 74)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    x_log = np.log10(np.maximum(curve.concentrations_nm, 1e-6))
    x_lo, x_hi = float(x_log[0]), float(x_log[-1])
    y_max = max(float(np.max(curve.ternary_complex_nm)), 1e-6)

    for col in range(width):
        lx = x_lo + (col / max(1, width - 1)) * (x_hi - x_lo)
        val = float(np.interp(lx, x_log, curve.ternary_complex_nm))
        row = height - 1 - int(round((val / y_max) * (height - 1)))
        row = max(0, min(height - 1, row))
        grid[row][col] = "#"

    for r in range(height):
        y_val = y_max * (height - 1 - r) / max(1, height - 1)
        lines.append(f"{y_val:6.2f}nM|{''.join(grid[r])}|")
    lines.append("        +" + "-" * width + "+")
    lines.append(f"   10^{x_lo:+.1f} nM{' ' * (width - 21)}10^{x_hi:+.1f} nM")
    return assert_ascii("\n".join(lines))


def render_docking_breakdown(name: str, pose: DockingPose) -> str:
    """Render empirical Vina-like binding energy decomposition bar chart."""
    lines: List[str] = ["=" * 74, f"EMPIRICAL DOCKING SCORE BREAKDOWN: {name}", "=" * 74]
    lines.append(
        f"  Rank #{pose.pose_rank} | DeltaG={pose.affinity_kcal:+.3f} kcal/mol | "
        f"Kd={pose.predicted_kd_um:.3f} uM | pKi={pose.predicted_pki:.2f} | "
        f"HBonds={pose.hbonds_formed} | RMSD={pose.rmsd_from_best:.2f} A"
    )
    lines.append("-" * 74)
    terms = [
        ("Gauss1 (steric)", pose.gauss1),
        ("Gauss2 (long)", pose.gauss2),
        ("Hydrophobic", pose.hydrophobic),
        ("Hydrogen Bond", pose.hydrogen_bond),
        ("Electrostatic", pose.electrostatic),
        ("Repulsion (+)", pose.repulsion),
        ("Torsion Ent.", pose.torsion_penalty),
    ]
    scale = max(1.0, max(abs(v) for _, v in terms))
    for label, val in terms:
        bar_len = int(round(min(1.0, abs(val) / scale) * 24))
        ch = "+" if val >= 0 else "-"
        lines.append(f"  {label:<16} {val:+8.3f}  [{ch * bar_len}{' ' * (24 - bar_len)}]")
    return assert_ascii("\n".join(lines))


def render_pareto_frontier(
    trace: OptimizationTrace,
    width: int = 58,
    height: int = 11,
) -> str:
    """Render 2D (-DeltaG_bind vs QED) Pareto scatter and top candidate scorecard."""
    lines: List[str] = ["=" * 74, f"DE NOVO PARETO OPTIMIZATION ({trace.method.upper()})", "=" * 74]
    lines.append(
        f"  Evaluated={trace.total_evaluated} | Pareto Front={len(trace.pareto_front)} | "
        f"Best MPO: {trace.history_best_mpo[0]:.3f} -> {trace.history_best_mpo[-1]:.3f} | "
        f"HV={trace.history_hypervolume[-1]:.3f}"
    )
    lines.append("-" * 74)

    all_c = trace.all_candidates
    xs = [-c.docking_kcal for c in all_c]
    ys = [c.qed for c in all_c]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = max(x_hi - x_lo, 0.5)
    y_span = max(y_hi - y_lo, 0.1)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for c in all_c:
        px = int(round(((-c.docking_kcal - x_lo) / x_span) * (width - 1)))
        py = height - 1 - int(round(((c.qed - y_lo) / y_span) * (height - 1)))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        if c.canonical_smiles == trace.seed_smiles:
            grid[py][px] = "S"
        elif c.pareto_rank == 0:
            grid[py][px] = "*"
        elif grid[py][px] == " ":
            grid[py][px] = "."

    for r in range(height):
        q_lbl = y_lo + ((height - 1 - r) / max(1, height - 1)) * y_span
        lines.append(f"QED={q_lbl:4.2f}|{''.join(grid[r])}|")
    lines.append("        +" + "-" * width + "+")
    lines.append(f"  -dG={x_lo:4.2f}{' ' * (width - 13)}-dG={x_hi:4.2f}  (*=Pareto, S=Seed, .=Dominated)")
    lines.append("-" * 74)
    lines.append("  Rank  MPO    dG(kcal)  QED    SA    logS   hERG  Op             SMILES")
    for idx, c in enumerate(trace.pareto_front[:6], start=1):
        lines.append(
            f"  #{idx:<3}  {c.mpo_score:.3f}  {c.docking_kcal:+7.3f}  {c.qed:.3f}  "
            f"{c.sa_score:4.2f}  {c.log_s:+5.2f}  {c.herg_pic50:4.2f}  {c.operation:<13}  {c.canonical_smiles[:36]}"
        )
    return assert_ascii("\n".join(lines))
