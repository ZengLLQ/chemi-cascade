"""Tests for Morgan circular fingerprints, Butina clustering, diversity & MMPs."""

import numpy as np

from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.fingerprints import (
    butina_clustering,
    compute_gini_diversity,
    compute_morgan_fingerprint,
    compute_topological_torsion,
    find_matched_molecular_pairs,
    maxmin_diverse_selection,
    pairwise_similarity_matrix,
    tanimoto_similarity,
    tversky_similarity,
)
from chemi_cascade.smiles import parse_smiles


def test_morgan_ecfp_fcfp_and_similarity_metrics() -> None:
    m1 = parse_smiles("c1ccc(Nc2ncnc3ccccc23)cc1")
    m2 = parse_smiles("Fc1ccc(Nc2ncnc3ccccc23)cc1")
    m3 = parse_smiles("CCCCCCCCCC")

    fp1 = compute_morgan_fingerprint(m1, radius=2, nbits=2048)
    fp2 = compute_morgan_fingerprint(m2, radius=2, nbits=2048)
    fp3 = compute_morgan_fingerprint(m3, radius=2, nbits=2048)

    assert len(fp1.bit_info) > 0
    assert tanimoto_similarity(fp1, fp1) == 1.0
    assert tanimoto_similarity(fp1, fp2) == tanimoto_similarity(fp2, fp1)
    assert tanimoto_similarity(fp1, fp2) > tanimoto_similarity(fp1, fp3)

    # Tversky substructure similarity
    tv = tversky_similarity(fp1, fp2, alpha=0.8, beta=0.2)
    assert 0.0 < tv <= 1.0

    # FCFP4 pharmacophore fingerprint & Topological Torsion
    fcfp1 = compute_morgan_fingerprint(m1, radius=2, nbits=1024, use_features=True)
    tor1 = compute_topological_torsion(m1, nbits=1024)
    assert fcfp1.kind == "FCFP4"
    assert int(np.sum(tor1.bits)) > 0


def test_similarity_matrix_butina_and_maxmin() -> None:
    smiles_list = [
        "c1ccc(Nc2ncnc3ccccc23)cc1",
        "Fc1ccc(Nc2ncnc3ccccc23)cc1",
        "Clc1ccc(Nc2ncnc3ccccc23)cc1",
        "CCCCCCCCCC",
        "CCCCCCCCCCO",
    ]
    mols = [parse_smiles(s) for s in smiles_list]
    fps = [compute_morgan_fingerprint(m, radius=2, nbits=2048) for m in mols]

    for metric in ("tanimoto", "dice", "cosine"):
        mat = pairwise_similarity_matrix(fps, metric=metric)
        assert mat.shape == (5, 5)
        assert np.allclose(np.diag(mat), 1.0)
        assert np.allclose(mat, mat.T)

    cl = butina_clustering(fps, similarity_threshold=0.55)
    assert len(cl.assignments) == 5
    assert sum(len(c) for c in cl.clusters) == 5
    # Two distinct chemical series (quinazolines vs decyl chains)
    assert cl.assignments[0] != cl.assignments[3]
    assert 0.0 <= cl.gini_diversity <= 1.0
    assert compute_gini_diversity([5, 5, 5]) == 0.0

    sel = maxmin_diverse_selection(fps, k=3, seed_index=0)
    assert len(sel) == 3
    assert len(set(sel)) == 3


def test_matched_molecular_pairs_discovery() -> None:
    mols = [
        parse_smiles("Fc1ccc(Nc2ncnc3ccccc23)cc1"),
        parse_smiles("Clc1ccc(Nc2ncnc3ccccc23)cc1"),
        parse_smiles("N#Cc1ccc(Nc2ncnc3ccccc23)cc1"),
    ]
    props = [{"clogp": compute_descriptors(m).clogp} for m in mols]
    mmps = find_matched_molecular_pairs(mols, props, min_core_atoms=5)
    assert len(mmps) >= 1
    # Every MMP delta matches exact property difference
    for mmp in mmps:
        expected = round(props[mmp.to_idx]["clogp"] - props[mmp.from_idx]["clogp"], 4)
        assert abs(mmp.delta_properties["clogp"] - expected) < 1e-6


def test_morgan_ecfp6_and_tanimoto_triangle_inequality() -> None:
    m1 = parse_smiles("c1ccccc1C")
    m2 = parse_smiles("c1ccccc1CC")
    m3 = parse_smiles("c1ccccc1O")
    f1 = compute_morgan_fingerprint(m1, radius=3, nbits=2048)
    f2 = compute_morgan_fingerprint(m2, radius=3, nbits=2048)
    f3 = compute_morgan_fingerprint(m3, radius=3, nbits=2048)
    assert f1.kind == "ECFP6"
    d12 = 1.0 - tanimoto_similarity(f1, f2)
    d23 = 1.0 - tanimoto_similarity(f2, f3)
    d13 = 1.0 - tanimoto_similarity(f1, f3)
    # Jaccard / Tanimoto distance is a true metric satisfying triangle inequality
    assert d13 <= d12 + d23 + 1e-12
    assert d12 <= d13 + d23 + 1e-12


def test_butina_extreme_thresholds() -> None:
    mols = [parse_smiles(s) for s in ("c1ccccc1", "c1ccccc1C", "CCCCCC")]
    fps = [compute_morgan_fingerprint(m, radius=2, nbits=1024) for m in mols]
    cl_hi = butina_clustering(fps, similarity_threshold=0.99)
    assert len(cl_hi.clusters) == 3
    cl_lo = butina_clustering(fps, similarity_threshold=0.0)
    assert len(cl_lo.clusters) == 1
