"""Example 01: SMILES Parsing, Huckel Aromaticity, Kekulization & Physicochemical Descriptors."""

from __future__ import annotations

from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.smiles import has_substructure, parse_smiles
from chemi_cascade.topology import kekulize, murcko_scaffold_smiles, perceive_huckel_aromaticity
from chemi_cascade.visualizer import render_descriptor_panel


def main() -> None:
    """Demonstrate SMILES canonicalization, aromaticity/Kekule cycles, and descriptors."""
    samples = [
        ("Gefitinib_Scaffold", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Cyclooctatetraene_NonAromatic", "C1=CC=CC=CC=C1"),
    ]
    for name, smi in samples:
        mol = parse_smiles(smi, name=name)
        mol_arom = perceive_huckel_aromaticity(mol)
        scaf = murcko_scaffold_smiles(mol_arom)
        desc = compute_descriptors(mol_arom)
        print(render_descriptor_panel(mol_arom, desc, murcko_smiles=scaf))
        if any(a.is_aromatic for a in mol_arom.atoms):
            kek = kekulize(mol_arom)
            print(f"  Kekulized SMILES : {kek.canonical_smiles}")
        print(f"  Contains benzene : {has_substructure(mol_arom, 'c1ccccc1')}\n")


if __name__ == "__main__":
    main()
