"""Example 04: 3D Distance Geometry, Kinase Docking & PROTAC Ternary Hook Effect."""

from __future__ import annotations

from chemi_cascade.docking import (
    build_kinase_hinge_pocket,
    dock_ligand,
    simulate_protac_hook_effect,
)
from chemi_cascade.geometry import generate_conformer_ensemble
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.types import ProtacSpec
from chemi_cascade.visualizer import render_docking_breakdown, render_protac_hook_curve


def main() -> None:
    """Generate 3D conformers, dock into kinase hinge, and sweep PROTAC cooperativity."""
    ligand = parse_smiles("c1ccc(Nc2ncnc3ccccc23)cc1", name="Anilinoquinazoline")
    ensemble = generate_conformer_ensemble(ligand, n_confs=3, seed=42)
    print(f"Generated {len(ensemble)} conformers; lowest UFF={ensemble[0].uff_energy:.3f} kcal/mol")

    pocket = build_kinase_hinge_pocket()
    poses = dock_ligand(ligand, pocket, conformer=ensemble[0], n_starts=6, n_poses=2, seed=42)
    if poses:
        print(render_docking_breakdown(ligand.name, poses[0]))

    curve = simulate_protac_hook_effect(
        ProtacSpec(
            name="Kinase_VHL_PROTAC",
            kd_poi_nm=12.0,
            kd_e3_nm=25.0,
            cooperativity_alpha=12.0,
            linker_contour_angstrom=18.0,
            required_span_angstrom=12.0,
        )
    )
    print(render_protac_hook_curve(curve))


if __name__ == "__main__":
    main()
