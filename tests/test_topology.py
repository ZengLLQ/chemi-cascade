"""Tests for ring perception, Huckel 4n+2 aromaticity, Kekulization, and Murcko scaffolds."""

import pytest

from chemi_cascade.smiles import parse_smiles
from chemi_cascade.topology import (
    connected_ring_systems,
    extract_murcko_scaffold,
    kekulize,
    murcko_scaffold_smiles,
    perceive_huckel_aromaticity,
)


def test_ring_systems_fused_vs_biphenyl() -> None:
    naphthalene = parse_smiles("c1ccc2ccccc2c1")
    sys_naph = connected_ring_systems(naphthalene)
    assert len(sys_naph) == 1
    assert len(sys_naph[0]) == 10

    biphenyl = parse_smiles("c1ccccc1-c1ccccc1")
    sys_biph = connected_ring_systems(biphenyl)
    assert len(sys_biph) == 2
    assert all(len(s) == 6 for s in sys_biph)


def test_huckel_aromaticity_perception() -> None:
    # Kekule benzene (6 pi) -> perceived aromatic
    benz_kek = parse_smiles("C1=CC=CC=C1")
    benz_arom = perceive_huckel_aromaticity(benz_kek)
    assert all(a.is_aromatic for a in benz_arom.atoms)
    assert all(b.is_aromatic for b in benz_arom.bonds)

    # Furan (4 C + 1 O lone pair = 6 pi) -> perceived aromatic
    furan = perceive_huckel_aromaticity(parse_smiles("C1=COC=C1"))
    assert all(a.is_aromatic for a in furan.atoms)

    # Pyrrole (4 C + 1 NH lone pair = 6 pi) -> perceived aromatic
    pyrrole = perceive_huckel_aromaticity(parse_smiles("C1=CNC=C1"))
    assert all(a.is_aromatic for a in pyrrole.atoms)

    # Cyclooctatetraene (8 pi = 4n) -> NOT aromatic
    cot = perceive_huckel_aromaticity(parse_smiles("C1=CC=CC=CC=C1"))
    assert not any(a.is_aromatic for a in cot.atoms)


def test_kekulization_conserves_valence_and_bond_orders() -> None:
    for smi in ["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"]:
        mol = parse_smiles(smi)
        kek = kekulize(mol)
        assert not any(b.is_aromatic for b in kek.bonds)
        # Verify at least one localized double bond was placed in aromatic rings
        assert any(abs(b.order - 2.0) < 1e-6 for b in kek.bonds)
        # Verify carbon valence conservation: bond_order_sum + total_h == 4
        for a in kek.atoms:
            if a.symbol == "C":
                assert int(round(kek.bond_order_sum(a.idx))) + a.total_h == 4

    # Invalid radical/impossible aromatic ring raises ValueError cleanly
    bad_ring = parse_smiles("c1cccc1")
    with pytest.raises(ValueError):
        kekulize(bad_ring)


def test_bemis_murcko_scaffold_and_generic_framework() -> None:
    # Ibuprofen has one benzene ring + aliphatic isobutyl & propionic acid tails
    ibu = parse_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    scaf = murcko_scaffold_smiles(ibu)
    assert scaf == parse_smiles("c1ccccc1").canonical_smiles

    # Biphenyl with terminal methyls and carbonyl linker retains inter-ring linker + =O
    linked = parse_smiles("Cc1ccc(C(=O)Nc2ccc(Cl)cc2)cc1")
    scaf_linked = extract_murcko_scaffold(linked, generic=False)
    assert scaf_linked.num_atoms >= 14  # 2 phenyl rings (12) + C(=O)N linker (3)

    # Generic framework turns all atoms into C and all bonds into single
    gen = extract_murcko_scaffold(linked, generic=True)
    assert all(a.symbol == "C" and not a.is_aromatic for a in gen.atoms)
    assert all(abs(b.order - 1.0) < 1e-6 for b in gen.bonds)

    # Acyclic molecule yields empty Murcko scaffold
    assert murcko_scaffold_smiles(parse_smiles("CCCC(=O)O")) == ""


def test_heteroaromatic_five_and_six_membered_rings() -> None:
    for smi in ["C1=CSC=C1", "C1=NC=CC=C1", "C1=CC=C2C=CC=CC2=C1"]:
        mol = perceive_huckel_aromaticity(parse_smiles(smi))
        assert all(a.is_aromatic for a in mol.atoms)
        kek = kekulize(mol)
        assert not any(b.is_aromatic for b in kek.bonds)


def test_spiro_and_bridged_ring_systems() -> None:
    spiro = parse_smiles("C1CCC2(CC1)CCCC2")
    sys_spiro = connected_ring_systems(spiro)
    assert len(sys_spiro) == 1
    assert len(sys_spiro[0]) == 10

    norbornane = parse_smiles("C1CC2CCC1C2")
    sys_nor = connected_ring_systems(norbornane)
    assert len(sys_nor) == 1
    assert len(norbornane.rings) == 2
