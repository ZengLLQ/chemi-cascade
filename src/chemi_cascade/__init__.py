"""ChemiCascade: Computational cheminformatics, GNN ADME, docking & de novo design."""

from chemi_cascade.adme_pk import (
    build_calibrated_adme_model,
    estimate_dominant_pka,
    henderson_hasselbalch_logd,
    scale_hepatic_clearance,
    simulate_two_compartment_pk,
)
from chemi_cascade.benchmark import BenchmarkReport, run_benchmark_suite
from chemi_cascade.descriptors import compute_descriptors, compute_gasteiger_charges
from chemi_cascade.docking import (
    build_kinase_hinge_pocket,
    dock_ligand,
    score_ligand_pose,
    simulate_protac_hook_effect,
    solve_ternary_equilibrium,
)
from chemi_cascade.featurizer import featurize_molecule, scaffold_split
from chemi_cascade.fingerprints import (
    butina_clustering,
    compute_morgan_fingerprint,
    compute_topological_torsion,
    find_matched_molecular_pairs,
    maxmin_diverse_selection,
    pairwise_similarity_matrix,
    tanimoto_similarity,
)
from chemi_cascade.geometry import embed_conformer, generate_conformer_ensemble, kabsch_rmsd
from chemi_cascade.gnn import MolecularMPNN, OptionalPyTorchBridge
from chemi_cascade.optimizer import run_pareto_optimization
from chemi_cascade.pipeline import CascadeResult, DiscoveryCascade, export_cascade_json
from chemi_cascade.smiles import (
    SmilesSyntaxError,
    has_substructure,
    molecular_formula,
    parse_smiles,
    substructure_matches,
    to_canonical_smiles,
)
from chemi_cascade.topology import (
    connected_ring_systems,
    extract_murcko_scaffold,
    kekulize,
    murcko_scaffold_smiles,
    perceive_huckel_aromaticity,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkReport",
    "CascadeResult",
    "DiscoveryCascade",
    "MolecularMPNN",
    "OptionalPyTorchBridge",
    "SmilesSyntaxError",
    "build_calibrated_adme_model",
    "build_kinase_hinge_pocket",
    "butina_clustering",
    "compute_descriptors",
    "compute_gasteiger_charges",
    "compute_morgan_fingerprint",
    "compute_topological_torsion",
    "connected_ring_systems",
    "dock_ligand",
    "embed_conformer",
    "estimate_dominant_pka",
    "export_cascade_json",
    "extract_murcko_scaffold",
    "featurize_molecule",
    "find_matched_molecular_pairs",
    "generate_conformer_ensemble",
    "has_substructure",
    "henderson_hasselbalch_logd",
    "kabsch_rmsd",
    "kekulize",
    "maxmin_diverse_selection",
    "molecular_formula",
    "murcko_scaffold_smiles",
    "pairwise_similarity_matrix",
    "parse_smiles",
    "perceive_huckel_aromaticity",
    "run_benchmark_suite",
    "run_pareto_optimization",
    "scaffold_split",
    "scale_hepatic_clearance",
    "score_ligand_pose",
    "simulate_protac_hook_effect",
    "simulate_two_compartment_pk",
    "solve_ternary_equilibrium",
    "substructure_matches",
    "tanimoto_similarity",
    "to_canonical_smiles",
]
