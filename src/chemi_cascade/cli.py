"""Command-line interface for ChemiCascade computational drug discovery platform."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from chemi_cascade.adme_pk import build_calibrated_adme_model, simulate_two_compartment_pk
from chemi_cascade.benchmark import run_benchmark_suite, sensitivity_pk_clearance
from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.docking import (
    build_kinase_hinge_pocket,
    dock_ligand,
    simulate_protac_hook_effect,
)
from chemi_cascade.fingerprints import (
    butina_clustering,
    compute_morgan_fingerprint,
    find_matched_molecular_pairs,
    maxmin_diverse_selection,
    pairwise_similarity_matrix,
)
from chemi_cascade.geometry import embed_conformer
from chemi_cascade.optimizer import run_pareto_optimization
from chemi_cascade.pipeline import DEFAULT_SERIES_SMILES, DiscoveryCascade, export_cascade_json
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.topology import murcko_scaffold_smiles
from chemi_cascade.types import ProtacSpec
from chemi_cascade.visualizer import (
    render_descriptor_panel,
    render_docking_breakdown,
    render_pareto_frontier,
    render_pk_curve,
    render_protac_hook_curve,
    render_similarity_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct top-level ``chemi-cascade`` argument parser with 7 subcommands."""
    parser = argparse.ArgumentParser(
        prog="chemi-cascade",
        description="ChemiCascade: Pure-Python Computational Cheminformatics, GNN ADME, Docking & De Novo Design",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ana = sub.add_parser("analyze", help="Compute canonical SMILES, Murcko scaffold, Gasteiger Q & descriptors")
    p_ana.add_argument("--smiles", type=str, default="CC(=O)Oc1ccccc1C(=O)O", help="Input SMILES string")
    p_ana.add_argument("--name", type=str, default="QueryCompound", help="Compound identifier")

    p_clu = sub.add_parser("cluster", help="ECFP4 Tanimoto similarity, Butina clustering & MMP analysis")
    p_clu.add_argument("--threshold", type=float, default=0.55, help="Butina Tanimoto sphere threshold")
    p_clu.add_argument("--radius", type=int, default=2, help="Morgan circular radius (2=ECFP4)")

    p_adme = sub.add_parser("adme", help="Train/run MPNN-GAT ADME prediction & 2-compartment oral PBPK")
    p_adme.add_argument("--smiles", type=str, default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C", help="Target SMILES")
    p_adme.add_argument("--epochs", type=int, default=20, help="GNN training epochs")
    p_adme.add_argument("--dose-mg", type=float, default=100.0, help="Oral dose in mg")
    p_adme.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")

    p_dock = sub.add_parser("dock", help="3D DG/UFF conformer embedding, kinase docking & PROTAC hook curve")
    p_dock.add_argument("--smiles", type=str, default="Cc1ccc(NC(=O)c2ccccc2)cc1", help="Ligand SMILES")
    p_dock.add_argument("--n-starts", type=int, default=6, help="Number of MC+Powell docking starts")
    p_dock.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")

    p_opt = sub.add_parser("optimize", help="De novo MCTS + NSGA-II multi-objective lead optimization")
    p_opt.add_argument("--seed-smiles", type=str, default="c1ccc(Nc2ncnc3ccccc23)cc1", help="Seed SMILES")
    p_opt.add_argument("--generations", type=int, default=4, help="Evolutionary generations")
    p_opt.add_argument("--pop-size", type=int, default=10, help="Population size per generation")
    p_opt.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")

    p_bench = sub.add_parser("benchmark", help="Run invariant checks, scaffold-split GNN & docking enrichment")
    p_bench.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    p_bench.add_argument("--epochs", type=int, default=20, help="GNN training epochs")

    p_demo = sub.add_parser("demo", help="Run complete end-to-end discovery cascade showcase")
    p_demo.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    p_demo.add_argument("--out-json", type=str, default="", help="Optional path to save JSON report")

    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    mol = parse_smiles(args.smiles, name=args.name)
    scaf = murcko_scaffold_smiles(mol)
    desc = compute_descriptors(mol)
    print(render_descriptor_panel(mol, desc, murcko_smiles=scaf))
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    mols = [parse_smiles(smi, name=name) for name, smi in DEFAULT_SERIES_SMILES]
    descs = [compute_descriptors(m) for m in mols]
    fps = [compute_morgan_fingerprint(m, radius=args.radius, nbits=2048) for m in mols]
    sim = pairwise_similarity_matrix(fps, metric="tanimoto")
    res = butina_clustering(fps, similarity_threshold=args.threshold)
    diverse = maxmin_diverse_selection(fps, k=4, seed_index=0)
    props = [{"clogp": d.clogp, "qed": d.qed, "tpsa": d.tpsa} for d in descs]
    mmps = find_matched_molecular_pairs(mols, props)

    print(render_similarity_matrix([m.name for m in mols], sim))
    print(f"Butina Clusters (threshold={res.threshold:.2f}, Gini={res.gini_diversity:.3f}):")
    for cid, members in enumerate(res.clusters):
        names = [mols[i].name for i in members]
        print(f"  Cluster {cid}: {names}")
    print(f"MaxMin Diverse 4: {[mols[i].name for i in diverse]}")
    print(f"Matched Molecular Pairs discovered: {len(mmps)}")
    for pair in mmps[:5]:
        print(
            f"  MMP {mols[pair.from_idx].name} -> {mols[pair.to_idx].name} | "
            f"core={pair.core_smiles} | dLogP={pair.delta_properties.get('clogp', 0.0):+.3f} "
            f"dQED={pair.delta_properties.get('qed', 0.0):+.3f}"
        )
    return 0


def _cmd_adme(args: argparse.Namespace) -> int:
    mol = parse_smiles(args.smiles, name="ADME_Target")
    desc = compute_descriptors(mol)
    model, metrics = build_calibrated_adme_model(epochs=args.epochs, hidden_dim=36, seed=args.seed)
    adme = model.predict_with_uncertainty(mol, mc_samples=10, seed=args.seed)
    pk = simulate_two_compartment_pk(adme, mw=desc.molecular_weight, dose_mg=args.dose_mg)

    print(f"Calibrated MPNN-GAT (overall_rmse={metrics['overall_rmse']:.4f}):")
    print(
        f"  Uncertainties: logS+/-{adme.uncertainties['log_s']:.3f}, "
        f"logD+/-{adme.uncertainties['log_d_74']:.3f}, hERG+/-{adme.uncertainties['herg_pic50']:.3f}"
    )
    top_som = sorted(enumerate(adme.som_scores), key=lambda x: x[1], reverse=True)[:4]
    print("  Top CYP Soft-Spots (atom_idx, symbol, score): " + ", ".join(
        f"({idx}:{mol.atoms[idx].symbol}={sc:.2f})" for idx, sc in top_som
    ))
    print(render_pk_curve(pk, adme=adme))
    return 0


def _cmd_dock(args: argparse.Namespace) -> int:
    mol = parse_smiles(args.smiles, name="Docking_Ligand")
    conf = embed_conformer(mol, seed=args.seed)
    pocket = build_kinase_hinge_pocket()
    poses = dock_ligand(mol, pocket, conformer=conf, n_starts=args.n_starts, n_poses=3, seed=args.seed)

    print(f"3D UFF Conformer Energy={conf.uff_energy:.3f} kcal/mol (converged={conf.converged})")
    if poses:
        print(render_docking_breakdown(mol.name, poses[0]))
    curve = simulate_protac_hook_effect(
        ProtacSpec(
            name="Kinase_PROTAC",
            kd_poi_nm=15.0,
            kd_e3_nm=25.0,
            cooperativity_alpha=10.0,
            linker_contour_angstrom=18.0,
            required_span_angstrom=12.0,
        )
    )
    print(render_protac_hook_curve(curve))
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    trace = run_pareto_optimization(
        seed_smiles=args.seed_smiles,
        generations=args.generations,
        pop_size=args.pop_size,
        seed=args.seed,
    )
    print(render_pareto_frontier(trace))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    rep = run_benchmark_suite(seed=args.seed, epochs=args.epochs)
    print("=" * 74)
    print("CHEMICASCADE QUANTITATIVE BENCHMARK & INVARIANT REPORT")
    print("=" * 74)
    print(f"  Gasteiger Charge Conservation Max Err : {rep.gasteiger_max_charge_error:.3e}")
    print(f"  GNN Analytical vs Finite-Diff Grad Err: {rep.gnn_max_rel_grad_error:.3e}")
    print(f"  Murcko Scaffold-Split ADME Test RMSE  : {rep.scaffold_split_test_rmse:.4f}")
    print(f"  Murcko Scaffold-Split Spearman logD   : {rep.scaffold_split_spearman_logd:.4f}")
    print(f"  Kinase Docking Enrichment ROC-AUC     : {rep.docking_enrichment_roc_auc:.4f}")
    print(
        f"  Mean Active vs Decoy Affinity (kcal)  : "
        f"{rep.docking_mean_active_kcal:.3f} vs {rep.docking_mean_decoy_kcal:.3f}"
    )
    print(f"  PROTAC Mass Conservation Max Err      : {rep.protac_mass_conservation_err:.3e}")
    print(
        f"  PROTAC Optimal Linker / Peak [ABP]    : "
        f"{rep.protac_optimal_contour_a:.1f} A -> {rep.protac_peak_ternary_nm:.2f} nM"
    )
    print(f"  De Novo Optimization MPO Gain         : {rep.optimization_mpo_gain:+.4f}")
    print(f"  ALL VERIFICATION GATES PASSED         : {rep.all_gates_passed}")
    sens = sensitivity_pk_clearance()
    print("-" * 74)
    print(f"  PK Clearance Sensitivity Sweep rows   : {len(sens)} (WS vs Parallel-Tube)")
    return 0 if rep.all_gates_passed else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    cascade = DiscoveryCascade(seed=args.seed)
    res = cascade.run_series(opt_generations=3, opt_pop_size=8, adme_epochs=20)
    print(render_descriptor_panel(res.molecules[0], res.descriptors[0], res.scaffolds[0]))
    if res.docking_results and res.docking_results[0]:
        print(render_docking_breakdown(res.molecules[0].name, res.docking_results[0][0]))
    print(render_pk_curve(res.pk_profiles[0], adme=res.adme_predictions[0]))
    print(render_protac_hook_curve(res.protac_curve))
    print(render_pareto_frontier(res.optimization_trace))
    if args.out_json:
        saved = export_cascade_json(res, args.out_json)
        print(f"Exported JSON report to {saved}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point returning POSIX exit status code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    dispatch = {
        "analyze": _cmd_analyze,
        "cluster": _cmd_cluster,
        "adme": _cmd_adme,
        "dock": _cmd_dock,
        "optimize": _cmd_optimize,
        "benchmark": _cmd_benchmark,
        "demo": _cmd_demo,
    }
    handler = dispatch[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
