"""Tests for 3D distance geometry, UFF minimization, empirical docking & PROTACs."""

import numpy as np

from chemi_cascade.docking import (
    build_kinase_hinge_pocket,
    dock_ligand,
    score_ligand_pose,
    simulate_protac_hook_effect,
    solve_ternary_equilibrium,
    wlc_linker_entropy_penalty,
)
from chemi_cascade.geometry import (
    build_distance_bounds,
    embed_conformer,
    generate_conformer_ensemble,
    kabsch_rmsd,
    triangle_smooth,
)
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.types import ProtacSpec


def test_distance_bounds_and_triangle_smoothing() -> None:
    mol = parse_smiles("c1ccccc1CC(=O)O")
    L, U = build_distance_bounds(mol)
    lo, hi = triangle_smooth(L, U)
    n = mol.num_atoms
    assert np.all(lo <= hi + 1e-9)
    # Verify triangle inequality U[i,j] <= U[i,k] + U[k,j]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert hi[i, j] <= hi[i, k] + hi[k, j] + 1e-7


def test_3d_conformer_uff_minimization_and_kabsch_invariance() -> None:
    benz = parse_smiles("c1ccccc1")
    conf = embed_conformer(benz, seed=42)
    assert conf.coords.shape == (6, 3)
    assert conf.converged
    # Benzene must relax to a planar ring (smallest SVD singular value near 0)
    centered = conf.coords - np.mean(conf.coords, axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(centered)
    assert svals[-1] < 0.15

    # Kabsch RMSD must be invariant under arbitrary SO(3) rotation + translation
    theta = 0.75
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transformed = conf.coords @ rot.T + np.array([4.2, -1.7, 9.1])
    assert kabsch_rmsd(conf.coords, transformed) < 1e-8

    ensemble = generate_conformer_ensemble(parse_smiles("CCCCc1ccccc1"), n_confs=3, seed=7)
    assert len(ensemble) >= 1


def test_empirical_docking_and_active_vs_decoy_ranking() -> None:
    pocket = build_kinase_hinge_pocket()
    active = parse_smiles("c1ccc(Nc2ncnc3ccccc23)cc1", name="quinazoline")
    decoy = parse_smiles("CCO", name="ethanol")

    poses_act = dock_ligand(active, pocket, n_starts=6, n_poses=3, seed=42)
    poses_dec = dock_ligand(decoy, pocket, n_starts=6, n_poses=2, seed=42)
    assert len(poses_act) >= 1
    assert poses_act[0].affinity_kcal < poses_dec[0].affinity_kcal
    assert poses_act[0].predicted_pki > poses_dec[0].predicted_pki

    # Score consistency check
    re_score = score_ligand_pose(active, poses_act[0].coords, pocket, rank=1)
    assert abs(re_score.affinity_kcal - poses_act[0].affinity_kcal) < 1e-6


def test_protac_mass_conservation_hook_effect_and_wlc() -> None:
    a_tot, b_tot, p_tot = 120.0, 60.0, 50.0
    fa, fb, fp, ap, bp, abp = solve_ternary_equilibrium(p_tot, a_tot, b_tot, 10.0, 20.0, 15.0)
    assert abs((fa + ap + abp) - a_tot) < 1e-9
    assert abs((fb + bp + abp) - b_tot) < 1e-9
    assert abs((fp + ap + bp + abp) - p_tot) < 1e-9
    assert min(fa, fb, fp, ap, bp, abp) >= 0.0

    # Hook effect: peak exceeds both extreme-low and extreme-high concentration tails
    curve = simulate_protac_hook_effect(
        ProtacSpec(
            name="Test_PROTAC",
            kd_poi_nm=10.0,
            kd_e3_nm=20.0,
            cooperativity_alpha=12.0,
            linker_contour_angstrom=18.0,
            required_span_angstrom=12.0,
        ),
        n_points=51,
    )
    assert curve.max_ternary_nm > curve.ternary_complex_nm[0]
    assert curve.max_ternary_nm > curve.ternary_complex_nm[-1] * 2.0
    assert curve.ec50_formation_nm < curve.hook_ic50_nm

    # Over-stretched linker incurs sharp WLC entropy penalty
    pen_ok, mod_ok = wlc_linker_entropy_penalty(20.0, 4.5, 12.0)
    pen_bad, mod_bad = wlc_linker_entropy_penalty(10.0, 4.5, 12.0)
    assert pen_bad > pen_ok and mod_bad < mod_ok


def test_protac_cooperativity_boosts_peak_and_edge_cases() -> None:
    # Positive cooperativity (alpha=20) produces higher peak ternary than negative (alpha=0.5)
    c_pos = simulate_protac_hook_effect(
        ProtacSpec("pos", kd_poi_nm=15.0, kd_e3_nm=25.0, cooperativity_alpha=20.0),
        n_points=41,
    )
    c_neg = simulate_protac_hook_effect(
        ProtacSpec("neg", kd_poi_nm=15.0, kd_e3_nm=25.0, cooperativity_alpha=0.5),
        n_points=41,
    )
    assert c_pos.max_ternary_nm > c_neg.max_ternary_nm

    # Zero PROTAC concentration yields zero complexes
    fa, fb, fp, ap, bp, abp = solve_ternary_equilibrium(0.0, 100.0, 50.0, 10.0, 20.0, 5.0)
    assert fa == 100.0 and fb == 50.0 and fp == 0.0 and ap == 0.0 and bp == 0.0 and abp == 0.0


def test_docking_single_atom_and_torsion_penalty() -> None:
    pocket = build_kinase_hinge_pocket()
    rigid = parse_smiles("c1ccc2ccccc2c1")
    flex = parse_smiles("CCCCCCCCc1ccccc1")
    p_rig = dock_ligand(rigid, pocket, n_starts=4, n_poses=1, seed=42)[0]
    p_flx = dock_ligand(flex, pocket, n_starts=4, n_poses=1, seed=42)[0]
    assert p_rig.torsion_penalty == 0.0
    assert p_flx.torsion_penalty >= 0.0
