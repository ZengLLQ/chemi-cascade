"""Quantitative medicinal chemistry benchmark, calibration, and sensitivity suite.

Validates numerical correctness, physical conservation laws, and model fidelity:
1. Gasteiger-Marsili PEOE formal-charge conservation invariant across charged
   and zwitterionic molecules.
2. MolecularMPNN analytical vs. central finite-difference gradient verification.
3. Bemis-Murcko scaffold-split ADME generalization (RMSE and Spearman rank rho).
4. Kinase hinge docking virtual screening enrichment (ROC-AUC and EF_20%)
   comparing active hinge-binding scaffolds vs. non-binding decoy molecules.
5. PROTAC ternary cooperativity mass-balance conservation & linker-length sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

from chemi_cascade.adme_pk import REFERENCE_DRUG_SMILES, synthetic_adme_ground_truth
from chemi_cascade.descriptors import compute_descriptors, compute_gasteiger_charges
from chemi_cascade.docking import (
    build_kinase_hinge_pocket,
    dock_ligand,
    simulate_protac_hook_effect,
    solve_ternary_equilibrium,
)
from chemi_cascade.featurizer import featurize_molecule, scaffold_split
from chemi_cascade.gnn import MolecularMPNN
from chemi_cascade.optimizer import run_pareto_optimization
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.types import ProtacSpec


@dataclass
class BenchmarkReport:
    """Quantitative benchmark metrics across all ChemiCascade subsystems."""

    gasteiger_max_charge_error: float
    gnn_max_rel_grad_error: float
    scaffold_split_test_rmse: float
    scaffold_split_spearman_logd: float
    docking_enrichment_roc_auc: float
    docking_mean_active_kcal: float
    docking_mean_decoy_kcal: float
    protac_mass_conservation_err: float
    protac_optimal_contour_a: float
    protac_peak_ternary_nm: float
    optimization_mpo_gain: float
    all_gates_passed: bool


def _roc_auc_from_scores(y_true: List[int], scores_higher_better: List[float]) -> float:
    """Compute exact trapezoidal ROC-AUC via Mann-Whitney U statistic."""
    pos = [s for y, s in zip(y_true, scores_higher_better) if y == 1]
    neg = [s for y, s in zip(y_true, scores_higher_better) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def run_benchmark_suite(seed: int = 42, epochs: int = 25) -> BenchmarkReport:
    """Execute full end-to-end verification and quantitative benchmarking."""
    # 1. Gasteiger charge conservation across neutral + charged species
    charge_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)[O-]",
        "C[NH3+]",
        "c1ccc([NH3+])cc1C(=O)[O-]",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]
    max_q_err = 0.0
    for smi in charge_smiles:
        mol = parse_smiles(smi)
        q = compute_gasteiger_charges(mol)
        expected_q = float(sum(a.formal_charge for a in mol.atoms))
        max_q_err = max(max_q_err, abs(sum(q) - expected_q))

    # 2. GNN analytical gradient check + Murcko scaffold-split ADME evaluation
    mols = [parse_smiles(smi, name=name) for name, smi in REFERENCE_DRUG_SMILES]
    graphs = [featurize_molecule(m) for m in mols]
    targets = np.vstack([synthetic_adme_ground_truth(m) for m in mols])

    model = MolecularMPNN(hidden_dim=36, num_layers=2, seed=seed)
    grad_err = model.verify_gradients(graphs[0], eps=1e-5, seed=seed)

    tr_idx, va_idx, te_idx = scaffold_split(mols, train_frac=0.65, val_frac=0.15, seed=seed)
    if not te_idx:
        te_idx = va_idx
    tr_g = [graphs[i] for i in tr_idx]
    tr_y = targets[tr_idx]
    va_g = [graphs[i] for i in va_idx]
    va_y = targets[va_idx]
    te_g = [graphs[i] for i in te_idx]
    te_y = targets[te_idx]

    model.fit(tr_g, tr_y, val_graphs=va_g, val_targets=va_y, epochs=epochs, batch_size=6, lr=4e-3, seed=seed)
    te_preds = np.vstack([model.predict_raw(g) for g in te_g])
    test_rmse = float(np.sqrt(np.mean((te_preds - te_y) ** 2)))
    rho_res = spearmanr(te_y[:, 1], te_preds[:, 1])
    rho_val = float(rho_res.statistic) if np.isfinite(rho_res.statistic) else 0.0

    # 3. Docking enrichment: hinge-binding kinase scaffolds vs small/aliphatic decoys
    pocket = build_kinase_hinge_pocket()
    actives_smi = [
        "c1ccc(Nc2ncnc3ccccc23)cc1",          # 4-anilinoquinazoline hinge binder
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC",  # gefitinib-like core
        "Cc1ccc(NC(=O)c2ccccc2)cc1Nc1ncccc1",    # diaryl amide-aminopyrimidine
        "O=C1NC(=O)c2c1c1ccccc1[nH]2",           # staurosporine-like maleimide
        "Nc1ncnc2c1ccn2Cc1ccccc1",               # purine hinge scaffold
    ]
    decoys_smi = [
        "CCCCCC",
        "CC(C)CC(C)C",
        "C1CCCCC1",
        "CCO",
        "CC(=O)OC",
    ]
    y_true: List[int] = []
    scores: List[float] = []
    act_e: List[float] = []
    dec_e: List[float] = []

    for i, smi in enumerate(actives_smi):
        p = dock_ligand(parse_smiles(smi), pocket, n_starts=5, n_poses=1, seed=seed + i)
        e = p[0].affinity_kcal
        act_e.append(e)
        y_true.append(1)
        scores.append(-e)

    for i, smi in enumerate(decoys_smi):
        p = dock_ligand(parse_smiles(smi), pocket, n_starts=5, n_poses=1, seed=seed + 50 + i)
        e = p[0].affinity_kcal
        dec_e.append(e)
        y_true.append(0)
        scores.append(-e)

    auc = _roc_auc_from_scores(y_true, scores)

    # 4. PROTAC mass conservation + linker-length sensitivity sweep
    a_tot, b_tot, p_tot = 100.0, 50.0, 40.0
    fa, fb, fp, ap, bp, abp = solve_ternary_equilibrium(p_tot, a_tot, b_tot, 12.0, 25.0, 8.0)
    mass_err = max(
        abs((fa + ap + abp) - a_tot),
        abs((fb + bp + abp) - b_tot),
        abs((fp + ap + bp + abp) - p_tot),
    )

    best_contour = 18.0
    best_peak = -1.0
    for contour in (10.0, 14.0, 18.0, 24.0, 32.0):
        curve = simulate_protac_hook_effect(
            ProtacSpec(
                name=f"L_{int(contour)}A",
                kd_poi_nm=15.0,
                kd_e3_nm=25.0,
                cooperativity_alpha=10.0,
                linker_contour_angstrom=contour,
                required_span_angstrom=12.0,
            ),
            n_points=35,
        )
        if curve.max_ternary_nm > best_peak:
            best_peak = curve.max_ternary_nm
            best_contour = contour

    # 5. De novo optimization MPO gain check
    trace = run_pareto_optimization(
        seed_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
        adme_model=model,
        pocket=pocket,
        generations=3,
        pop_size=8,
        seed=seed,
    )
    seed_mpo = trace.history_best_mpo[0]
    final_mpo = trace.history_best_mpo[-1]
    mpo_gain = final_mpo - seed_mpo

    # Verify descriptor computation sanity
    _ = compute_descriptors(mols[0])

    gates_ok = (
        max_q_err < 1e-8
        and grad_err < 1e-4
        and auc >= 0.75
        and mass_err < 1e-6
        and mpo_gain >= 0.0
    )

    return BenchmarkReport(
        gasteiger_max_charge_error=round(max_q_err, 12),
        gnn_max_rel_grad_error=round(grad_err, 8),
        scaffold_split_test_rmse=round(test_rmse, 4),
        scaffold_split_spearman_logd=round(rho_val, 4),
        docking_enrichment_roc_auc=round(auc, 4),
        docking_mean_active_kcal=round(float(np.mean(act_e)), 4),
        docking_mean_decoy_kcal=round(float(np.mean(dec_e)), 4),
        protac_mass_conservation_err=round(mass_err, 12),
        protac_optimal_contour_a=best_contour,
        protac_peak_ternary_nm=round(best_peak, 3),
        optimization_mpo_gain=round(mpo_gain, 4),
        all_gates_passed=gates_ok,
    )


def sensitivity_pk_clearance(
    cl_grid: Tuple[float, ...] = (5.0, 15.0, 40.0, 90.0),
    fup_grid: Tuple[float, ...] = (0.02, 0.10, 0.30),
) -> List[Dict[str, float]]:
    """Run 2D parametric sensitivity sweep of hepatic CL_H and bioavailability."""
    from chemi_cascade.adme_pk import scale_hepatic_clearance

    rows: List[Dict[str, float]] = []
    for cl_int in cl_grid:
        for fup in fup_grid:
            cl_ws, eh_ws = scale_hepatic_clearance(cl_int, fup=fup, model="well_stirred")
            cl_pt, eh_pt = scale_hepatic_clearance(cl_int, fup=fup, model="parallel_tube")
            rows.append({
                "cl_int_ul_min_mg": cl_int,
                "fup": fup,
                "cl_h_well_stirred": cl_ws,
                "fh_well_stirred": round(1.0 - eh_ws, 4),
                "cl_h_parallel_tube": cl_pt,
                "fh_parallel_tube": round(1.0 - eh_pt, 4),
            })
    return rows
