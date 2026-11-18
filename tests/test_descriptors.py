"""Tests for Gasteiger partial charges, topological indices, and drug-likeness."""

import math

from chemi_cascade.descriptors import (
    compute_descriptors,
    compute_gasteiger_charges,
    compute_topological_indices,
    screen_structural_alerts,
)
from chemi_cascade.smiles import parse_smiles


def test_gasteiger_charge_conservation_and_symmetry() -> None:
    for smi, expected_q in [
        ("CC(=O)Oc1ccccc1C(=O)O", 0.0),
        ("CC(=O)[O-]", -1.0),
        ("C[NH3+]", 1.0),
        ("[NH3+]CC(=O)[O-]", 0.0),
        ("c1ccccc1", 0.0),
    ]:
        mol = parse_smiles(smi)
        charges = compute_gasteiger_charges(mol)
        assert len(charges) == mol.num_atoms
        assert abs(sum(charges) - expected_q) < 1e-10

    # Symmetry: all 6 carbons in benzene must carry identical partial charges
    benz = parse_smiles("c1ccccc1")
    q_benz = compute_gasteiger_charges(benz)
    assert max(q_benz) - min(q_benz) < 1e-10


def test_topological_indices_exact_values() -> None:
    # n-butane path P4: pairwise distances = 1,2,3 (from end), 1,2, 1 -> sum = 10
    butane = parse_smiles("CCCC")
    t_but = compute_topological_indices(butane)
    assert t_but["wiener"] == 10.0
    # Degrees: 1, 2, 2, 1 -> M1 = 1 + 4 + 4 + 1 = 10; M2 = 1*2 + 2*2 + 2*1 = 8
    assert t_but["zagreb_m1"] == 10.0
    assert t_but["zagreb_m2"] == 8.0
    assert abs(t_but["randic_chi"] - (2.0 / math.sqrt(2.0) + 0.5)) < 1e-6

    # Cyclobutane C4: pairwise distances = 1,2,1, 1,2, 1 -> sum = 8
    c4 = parse_smiles("C1CCC1")
    t_c4 = compute_topological_indices(c4)
    assert t_c4["wiener"] == 8.0


def test_physicochemical_descriptors_and_alerts() -> None:
    aspirin = compute_descriptors(parse_smiles("CC(=O)Oc1ccccc1C(=O)O"))
    hexane = compute_descriptors(parse_smiles("CCCCCC"))
    caffeine = compute_descriptors(parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C"))

    assert 179.0 < aspirin.molecular_weight < 181.5
    assert aspirin.lipinski_violations == 0
    assert aspirin.veber_pass is True
    assert 0.0 < aspirin.qed <= 1.0
    assert 1.0 <= aspirin.sa_score <= 10.0
    assert 0.0 <= aspirin.cns_mpo <= 6.0

    # Polarity and lipophilicity physical ordering
    assert hexane.tpsa == 0.0
    assert aspirin.tpsa > 55.0 and caffeine.tpsa > 55.0
    assert hexane.clogp > caffeine.clogp

    # Structural alerts
    nitro = parse_smiles("O=[N+]([O-])c1ccccc1")
    assert "nitro_group" in screen_structural_alerts(nitro)
    michael = parse_smiles("C=CC=O")
    assert "michael_acceptor_enal" in screen_structural_alerts(michael)


def test_rotatable_bonds_amide_exclusion_and_sa_bounds() -> None:
    # Amide C(=O)-N bond must NOT be counted as rotatable
    acetamide = parse_smiles("CC(=O)NC")
    prof_am = compute_descriptors(acetamide)
    assert prof_am.rotatable_bonds == 0

    # Flexible alkane vs rigid fused aromatic SA & Fsp3
    decane = compute_descriptors(parse_smiles("CCCCCCCCCC"))
    pyrene = compute_descriptors(parse_smiles("c1cc2ccc3cccc4ccc(c1)c2c34"))
    assert decane.fraction_csp3 == 1.0
    assert pyrene.fraction_csp3 == 0.0
    assert decane.rotatable_bonds == 7
    assert pyrene.rotatable_bonds == 0
    assert 1.0 <= decane.sa_score <= 10.0
    assert 1.0 <= pyrene.sa_score <= 10.0


def test_cns_mpo_and_lipinski_rule_of_five() -> None:
    # Small lipophilic polar-balanced CNS drug-like scaffold scores higher than huge polar peptide
    diazepam = compute_descriptors(parse_smiles("CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"))
    polar_macro = compute_descriptors(parse_smiles("OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"))
    assert diazepam.cns_mpo > polar_macro.cns_mpo
    assert diazepam.lipinski_violations == 0
    assert polar_macro.veber_pass is False or polar_macro.rotatable_bonds > 10
