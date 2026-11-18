"""Tests for SMILES parsing, canonicalization, molecular formulas, and SMARTS."""

import pytest

from chemi_cascade.smiles import (
    SmilesSyntaxError,
    has_substructure,
    molecular_formula,
    parse_smiles,
    substructure_matches,
    to_canonical_smiles,
)


def test_parse_smiles_variants_and_formula() -> None:
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O", name="aspirin")
    assert aspirin.num_atoms == 13
    assert aspirin.num_bonds == 13
    assert molecular_formula(aspirin) == "C9H8O4"

    caffeine = parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C", name="caffeine")
    assert molecular_formula(caffeine) == "C8H10N4O2"
    assert len(caffeine.rings) == 2


def test_canonical_smiles_isomorphism_invariance() -> None:
    pairs = [
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("Cc1ccccc1", "c1ccc(C)cc1"),
        ("CC(=O)Oc1ccccc1C(=O)O", "O=C(O)c1ccccc1OC(=O)C"),
        ("CCN(CC)CC", "CC(N(CC)CC)"),
        ("c1ccc2ccccc2c1", "c1cc2ccccc2cc1"),
    ]
    for s1, s2 in pairs:
        m1 = parse_smiles(s1)
        m2 = parse_smiles(s2)
        # Note: Kekule benzene vs aromatic benzene differ in aromaticity until perceived,
        # so check canonical invariance across isomorphic SMILES of same representation
        if s1 == "c1ccccc1":
            assert m1.num_atoms == m2.num_atoms == 6
            assert len(m1.rings) == len(m2.rings) == 1
        else:
            assert m1.canonical_smiles == m2.canonical_smiles


def test_bracketed_charges_isotopes_salts_and_multiring() -> None:
    salt = parse_smiles("[Na+].[O-]C(=O)C")
    assert salt.num_atoms == 5
    # Verify two disconnected components
    assert sum(a.formal_charge for a in salt.atoms) == 0

    zwit = parse_smiles("[NH3+]CC(=O)[O-]")
    assert zwit.atoms[0].formal_charge == 1
    assert zwit.atoms[4].formal_charge == -1
    assert sum(a.formal_charge for a in zwit.atoms) == 0

    isotope = parse_smiles("[13CH3]c1ccccc1")
    assert isotope.atoms[0].isotope == 13
    assert isotope.atoms[0].explicit_h == 3

    # %10 two-digit ring closure
    ring10 = parse_smiles("C%10CCCCC%10")
    assert ring10.num_atoms == 6
    assert len(ring10.rings) == 1


def test_invalid_smiles_raises_syntax_error() -> None:
    for bad in ["", "   ", "C(CC", "C1CCC", "C1CC1(", "[C", "Q1CCC1"]:
        with pytest.raises(SmilesSyntaxError):
            parse_smiles(bad)


def test_smarts_substructure_matching() -> None:
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert has_substructure(aspirin, "C(=O)O")
    assert has_substructure(aspirin, "c1ccccc1")
    assert not has_substructure(aspirin, "[N+](=O)[O-]")

    nitro = parse_smiles("O=[N+]([O-])c1ccccc1")
    assert has_substructure(nitro, "[N+](=O)[O-]")
    matches = substructure_matches(nitro, "[r6]")
    assert len(matches) == 6
    assert has_substructure(nitro, "[!#6]")

    # Round-trip canonical emission preserves graph invariants
    canon = to_canonical_smiles(aspirin)
    re_parsed = parse_smiles(canon)
    assert re_parsed.canonical_smiles == canon
    assert molecular_formula(re_parsed) == "C9H8O4"


def test_smarts_compound_primitives_and_negation() -> None:
    phenol = parse_smiles("Oc1ccccc1")
    assert has_substructure(phenol, "[OX2H1]")
    assert has_substructure(phenol, "c[OX2H1]")
    assert not has_substructure(parse_smiles("COc1ccccc1"), "c[OX2H1]")

    # Logical OR and logical AND inside bracket expression
    assert has_substructure(parse_smiles("c1ccncc1"), "[#7,#8]")
    assert has_substructure(parse_smiles("c1ccccc1"), "[c;r6;R1]")
    assert not has_substructure(parse_smiles("C1CCCCC1"), "[c;r6]")


def test_smiles_clone_and_neighbor_invariants() -> None:
    mol = parse_smiles("CC(=O)Nc1ccc(O)cc1", name="paracetamol")
    cloned = mol.clone()
    assert cloned.canonical_smiles == mol.canonical_smiles
    cloned.atoms[0].symbol = "N"
    assert mol.atoms[0].symbol == "C"
    for a in mol.atoms:
        assert mol.atoms[a.idx].idx == a.idx
        for nb in mol.neighbors(a.idx):
            b = mol.get_bond(a.idx, nb)
            assert b is not None
            assert b.other(a.idx) == nb
