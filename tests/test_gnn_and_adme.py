"""Tests for GNN featurization, backpropagation gradients, ADME & 2-compartment PBPK."""

import numpy as np

from chemi_cascade.adme_pk import (
    HUMAN_QH_L_H,
    REFERENCE_DRUG_SMILES,
    estimate_dominant_pka,
    henderson_hasselbalch_logd,
    scale_hepatic_clearance,
    simulate_two_compartment_pk,
    synthetic_adme_ground_truth,
)
from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.featurizer import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, featurize_molecule, scaffold_split
from chemi_cascade.gnn import MolecularMPNN, OptionalPyTorchBridge
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.topology import murcko_scaffold_smiles


def test_featurizer_shapes_and_scaffold_split_disjointness() -> None:
    mols = [parse_smiles(smi, name=name) for name, smi in REFERENCE_DRUG_SMILES[:16]]
    g = featurize_molecule(mols[0])
    assert g.node_features.shape == (mols[0].num_atoms, NODE_FEATURE_DIM)
    assert g.edge_features.shape == (2 * mols[0].num_bonds, EDGE_FEATURE_DIM)
    assert g.shortest_paths.shape == (mols[0].num_atoms, mols[0].num_atoms)

    tr, va, te = scaffold_split(mols, train_frac=0.6, val_frac=0.2, seed=42)
    assert len(tr) + len(va) + len(te) == len(mols)
    assert set(tr).isdisjoint(set(te))

    tr_scafs = {murcko_scaffold_smiles(mols[i]) for i in tr if murcko_scaffold_smiles(mols[i])}
    te_scafs = {murcko_scaffold_smiles(mols[i]) for i in te if murcko_scaffold_smiles(mols[i])}
    assert tr_scafs.isdisjoint(te_scafs)


def test_gnn_analytical_gradients_match_finite_differences() -> None:
    mol = parse_smiles("Cc1ccc(NC(=O)c2ccccc2)cc1")
    graph = featurize_molecule(mol)
    model = MolecularMPNN(hidden_dim=16, num_layers=2, dropout_rate=0.0, seed=11)
    rel_err = model.verify_gradients(graph, eps=1e-5, seed=11)
    assert rel_err < 1e-4


def test_gnn_training_uncertainty_and_atom_attribution() -> None:
    mols = [parse_smiles(smi, name=name) for name, smi in REFERENCE_DRUG_SMILES[:18]]
    graphs = [featurize_molecule(m) for m in mols]
    targets = np.vstack([synthetic_adme_ground_truth(m) for m in mols])

    model = MolecularMPNN(hidden_dim=28, num_layers=2, seed=42)
    hist = model.fit(graphs, targets, epochs=22, batch_size=6, lr=5e-3, seed=42)
    assert hist.train_losses[-1] < hist.train_losses[0]

    pred = model.predict_with_uncertainty(mols[0], mc_samples=8, seed=7)
    assert all(v >= 0.0 for v in pred.uncertainties.values())
    assert len(pred.atom_attributions["log_s"]) == mols[0].num_atoms
    assert len(pred.som_scores) == mols[0].num_atoms
    assert all(0.0 <= s <= 1.0 for s in pred.som_scores)

    state = OptionalPyTorchBridge.export_state_dict(model)
    assert "W_in" in state


def test_henderson_hasselbalch_and_pbpk_simulation() -> None:
    ibu = parse_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    prop = parse_smiles("CC(C)NCC(O)COc1cccc2ccccc12")

    acid_pka, _ = estimate_dominant_pka(ibu)
    _, basic_pka = estimate_dominant_pka(prop)
    assert acid_pka is not None and acid_pka < 5.5
    assert basic_pka is not None and basic_pka > 8.0

    logd_acid, fu_acid = henderson_hasselbalch_logd(3.5, acid_pka, None, ph=7.4)
    assert logd_acid < 3.5 and 0.0 < fu_acid < 0.05

    cl_ws, eh_ws = scale_hepatic_clearance(45.0, fup=0.2, model="well_stirred")
    cl_pt, eh_pt = scale_hepatic_clearance(45.0, fup=0.2, model="parallel_tube")
    assert 0.0 < cl_ws <= cl_pt <= HUMAN_QH_L_H
    assert 0.0 < eh_ws <= eh_pt < 1.0

    model = MolecularMPNN(hidden_dim=24, num_layers=2, seed=5)
    adme = model.predict_with_uncertainty(prop, mc_samples=4, seed=5)
    desc = compute_descriptors(prop)
    pk = simulate_two_compartment_pk(adme, mw=desc.molecular_weight, dose_mg=100.0)
    assert pk.cmax_ug_l > 0.0
    assert 0.0 < pk.tmax_h < pk.tau_hours
    assert pk.half_life_beta_h > 0.0
    assert 0.0 < pk.bioavailability_f < 1.0
    assert np.all(pk.c_plasma_ug_l >= 0.0)


def test_pbpk_dose_linearity_and_clearance_monotonicity() -> None:
    mol = parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    desc = compute_descriptors(mol)
    model = MolecularMPNN(hidden_dim=24, num_layers=2, seed=9)
    adme = model.predict_with_uncertainty(mol, mc_samples=4, seed=9)

    pk_100 = simulate_two_compartment_pk(adme, mw=desc.molecular_weight, dose_mg=100.0)
    pk_200 = simulate_two_compartment_pk(adme, mw=desc.molecular_weight, dose_mg=200.0)
    # Linear 2-compartment ODE: doubling dose doubles Cmax and AUC identically
    assert abs((pk_200.cmax_ug_l / pk_100.cmax_ug_l) - 2.0) < 1e-3
    assert abs((pk_200.auc_0_inf / pk_100.auc_0_inf) - 2.0) < 1e-3

    # Higher intrinsic clearance must strictly yield higher hepatic CL_H
    cl_lo, _ = scale_hepatic_clearance(5.0, fup=0.2)
    cl_hi, _ = scale_hepatic_clearance(150.0, fup=0.2)
    assert cl_hi > cl_lo


def test_gnn_single_atom_and_edge_less_molecule_safety() -> None:
    methane = parse_smiles("C")
    g = featurize_molecule(methane)
    assert g.num_atoms == 1 and g.num_directed_edges == 0
    model = MolecularMPNN(hidden_dim=16, num_layers=2, seed=3)
    out = model.predict_raw(g)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))
