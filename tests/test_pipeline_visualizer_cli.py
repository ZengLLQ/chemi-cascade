"""Tests for unified DiscoveryCascade pipeline, 7-bit ASCII visualizer & CLI."""

import json
from pathlib import Path

from chemi_cascade.cli import main
from chemi_cascade.fingerprints import compute_morgan_fingerprint, pairwise_similarity_matrix
from chemi_cascade.pipeline import DiscoveryCascade, export_cascade_json
from chemi_cascade.visualizer import (
    assert_ascii,
    render_descriptor_panel,
    render_docking_breakdown,
    render_pareto_frontier,
    render_pk_curve,
    render_protac_hook_curve,
    render_similarity_matrix,
)


def test_pipeline_json_export_and_strict_ascii_visualizers(tmp_path: Path) -> None:
    cascade = DiscoveryCascade(seed=42)
    res = cascade.run_series(opt_generations=2, opt_pop_size=6, adme_epochs=15)
    assert len(res.molecules) == 8
    assert len(res.adme_predictions) == 8
    assert len(res.pk_profiles) == 8

    out_file = tmp_path / "cascade_report.json"
    saved = export_cascade_json(res, out_file)
    assert saved.exists()
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert len(data["compounds"]) == 8
    assert len(data["pareto_front"]) >= 1

    # Strict 7-bit ASCII verification across all renderers
    fps = [compute_morgan_fingerprint(m, radius=2, nbits=1024) for m in res.molecules]
    sim = pairwise_similarity_matrix(fps)
    panels = [
        render_descriptor_panel(res.molecules[0], res.descriptors[0], res.scaffolds[0]),
        render_similarity_matrix([m.name for m in res.molecules], sim),
        render_pk_curve(res.pk_profiles[0], adme=res.adme_predictions[0]),
        render_protac_hook_curve(res.protac_curve),
        render_docking_breakdown(res.molecules[0].name, res.docking_results[0][0]),
        render_pareto_frontier(res.optimization_trace),
    ]
    for txt in panels:
        assert_ascii(txt)
        assert all(ord(ch) < 128 for ch in txt)


def test_cli_all_seven_subcommands(tmp_path: Path) -> None:
    json_out = tmp_path / "demo_out.json"
    invocations = [
        ["analyze", "--smiles", "CC(=O)Oc1ccccc1C(=O)O"],
        ["cluster", "--threshold", "0.55"],
        ["adme", "--smiles", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "--epochs", "12", "--seed", "42"],
        ["dock", "--smiles", "Cc1ccc(NC(=O)c2ccccc2)cc1", "--n-starts", "4", "--seed", "42"],
        ["optimize", "--seed-smiles", "c1ccc(Nc2ncnc3ccccc23)cc1", "--generations", "2", "--pop-size", "6"],
        ["benchmark", "--seed", "42", "--epochs", "15"],
        ["demo", "--seed", "42", "--out-json", str(json_out)],
    ]
    for argv in invocations:
        rc = main(argv)
        assert rc == 0
    assert json_out.exists()
