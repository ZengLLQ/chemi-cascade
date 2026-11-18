"""Example 05: End-to-End De Novo MCTS + NSGA-II Pareto Discovery Cascade."""

from __future__ import annotations

from chemi_cascade.benchmark import run_benchmark_suite
from chemi_cascade.pipeline import DiscoveryCascade
from chemi_cascade.visualizer import render_pareto_frontier


def main() -> None:
    """Run full DiscoveryCascade pipeline and quantitative benchmark gates."""
    cascade = DiscoveryCascade(seed=42)
    result = cascade.run_series(opt_generations=3, opt_pop_size=8, adme_epochs=20)
    print(render_pareto_frontier(result.optimization_trace))

    bench = run_benchmark_suite(seed=42, epochs=20)
    print(
        f"\nBenchmark Gates Passed={bench.all_gates_passed} | "
        f"Docking AUC={bench.docking_enrichment_roc_auc:.3f} | "
        f"MPO Gain={bench.optimization_mpo_gain:+.4f}"
    )


if __name__ == "__main__":
    main()
