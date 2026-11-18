"""Tests for de novo graph mutations, NSGA-II Pareto optimization & benchmarks."""

import numpy as np

from chemi_cascade.benchmark import run_benchmark_suite, sensitivity_pk_clearance
from chemi_cascade.optimizer import (
    compute_pareto_hypervolume_2d,
    generate_neighbors,
    non_dominated_sort,
    run_pareto_optimization,
)
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.types import OptimizationCandidate


def test_valence_safe_graph_mutations() -> None:
    seed_mol = parse_smiles("c1ccc(Nc2ncnc3ccccc23)cc1")
    rng = np.random.default_rng(42)
    neighbors = generate_neighbors(seed_mol, rng, max_neighbors=8)
    assert len(neighbors) >= 4
    canons = {m.canonical_smiles for _, m in neighbors}
    assert seed_mol.canonical_smiles not in canons
    # Ensure every generated mutant re-parses cleanly
    for _, mut in neighbors:
        reparsed = parse_smiles(mut.canonical_smiles)
        assert reparsed.num_atoms == mut.num_atoms


def test_nsga2_pareto_sorting_and_hypervolume() -> None:
    cands = [
        OptimizationCandidate("a", "a", 0, "", "seed", 0.5, -6.0, 0.70, 2.5, -2.5, 20.0, 4.5),
        OptimizationCandidate("b", "b", 1, "a", "add_F", 0.6, -7.2, 0.75, 2.4, -2.0, 15.0, 4.4),
        OptimizationCandidate("c", "c", 1, "a", "add_Cl", 0.4, -5.0, 0.60, 3.0, -3.5, 30.0, 5.0),
    ]
    fronts = non_dominated_sort(cands)
    assert len(fronts) >= 2
    assert cands[1].pareto_rank == 0
    assert cands[2].pareto_rank >= 1
    hv = compute_pareto_hypervolume_2d(fronts[0])
    assert hv > 0.0


def test_end_to_end_pareto_optimization_and_benchmark_gates() -> None:
    trace = run_pareto_optimization(
        seed_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
        generations=3,
        pop_size=8,
        seed=42,
    )
    assert trace.total_evaluated >= 5
    assert len(trace.pareto_front) >= 1
    assert trace.history_best_mpo[-1] >= trace.history_best_mpo[0]

    rep = run_benchmark_suite(seed=42, epochs=20)
    assert rep.all_gates_passed is True
    assert rep.gasteiger_max_charge_error < 1e-8
    assert rep.gnn_max_rel_grad_error < 1e-4
    assert rep.docking_enrichment_roc_auc >= 0.75
    assert rep.protac_mass_conservation_err < 1e-6

    sens = sensitivity_pk_clearance()
    assert len(sens) == 12
    assert all(r["fh_well_stirred"] >= 0.0 for r in sens)
