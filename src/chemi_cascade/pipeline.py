"""Unified end-to-end drug discovery cascade orchestrator and JSON report exporter.

Coordinates molecular ingestion, canonicalization, Murcko scaffold extraction,
physicochemical profiling, ECFP4 Butina clustering, Matched Molecular Pair (MMP)
analysis, multitask GNN ADME + 2-compartment PBPK simulation, 3D conformer
embedding & kinase docking, PROTAC ternary cooperativity, and de novo Pareto
lead optimization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chemi_cascade.adme_pk import build_calibrated_adme_model, simulate_two_compartment_pk
from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.docking import (
    build_kinase_hinge_pocket,
    dock_ligand,
    simulate_protac_hook_effect,
)
from chemi_cascade.fingerprints import (
    butina_clustering,
    compute_morgan_fingerprint,
    find_matched_molecular_pairs,
    maxmin_diverse_selection,
    pairwise_similarity_matrix,
)
from chemi_cascade.geometry import embed_conformer
from chemi_cascade.gnn import MolecularMPNN
from chemi_cascade.optimizer import run_pareto_optimization
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.topology import murcko_scaffold_smiles
from chemi_cascade.types import (
    ADMEPrediction,
    ClusterResult,
    Conformer3D,
    DescriptorProfile,
    DockingPose,
    MatchedMolecularPair,
    MoleculeGraph,
    OptimizationTrace,
    PKProfile,
    ProtacCurve,
    ProtacSpec,
)

DEFAULT_SERIES_SMILES: Tuple[Tuple[str, str], ...] = (
    ("Quinazoline_seed", "c1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_F", "Fc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_Cl", "Clc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_OMe", "COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OC"),
    ("Quinazoline_CN", "N#Cc1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Quinazoline_amide", "NC(=O)c1ccc(Nc2ncnc3ccccc23)cc1"),
    ("Aminopyrimidine_lead", "Cc1ccc(NC(=O)c2ccccc2)cc1Nc1ncccc1"),
    ("Indolocarbazole_ref", "O=C1NC(=O)c2c1c1ccccc1[nH]2"),
)


@dataclass
class CascadeResult:
    """Complete structured output from a ``DiscoveryCascade`` run."""

    molecules: List[MoleculeGraph]
    scaffolds: List[str]
    descriptors: List[DescriptorProfile]
    clusters: ClusterResult
    diverse_indices: List[int]
    mmps: List[MatchedMolecularPair]
    adme_predictions: List[ADMEPrediction]
    pk_profiles: List[PKProfile]
    conformers: List[Conformer3D]
    docking_results: List[List[DockingPose]]
    protac_curve: ProtacCurve
    optimization_trace: OptimizationTrace
    calibration_metrics: Dict[str, float]


class DiscoveryCascade:
    """High-level orchestrator for multi-stage computational drug discovery."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.adme_model: Optional[MolecularMPNN] = None
        self.calibration_metrics: Dict[str, float] = {}
        self.pocket = build_kinase_hinge_pocket()

    def ensure_calibrated_adme(self, epochs: int = 25) -> MolecularMPNN:
        """Lazily calibrate the multitask GNN ADME model if not already trained."""
        if self.adme_model is None:
            self.adme_model, self.calibration_metrics = build_calibrated_adme_model(
                epochs=epochs, hidden_dim=36, seed=self.seed
            )
        return self.adme_model

    def run_series(
        self,
        smiles_entries: Sequence[Tuple[str, str]] = DEFAULT_SERIES_SMILES,
        cluster_threshold: float = 0.55,
        opt_generations: int = 4,
        opt_pop_size: int = 10,
        adme_epochs: int = 25,
    ) -> CascadeResult:
        """Execute full cheminformatics -> ADME/PK -> 3D Docking -> De Novo cascade."""
        mols = [parse_smiles(smi, name=name) for name, smi in smiles_entries]
        scaffolds = [murcko_scaffold_smiles(m) for m in mols]
        descs = [compute_descriptors(m) for m in mols]

        fps = [compute_morgan_fingerprint(m, radius=2, nbits=2048) for m in mols]
        _ = pairwise_similarity_matrix(fps, metric="tanimoto")
        clusters = butina_clustering(fps, similarity_threshold=cluster_threshold)
        diverse = maxmin_diverse_selection(fps, k=min(4, len(mols)), seed_index=0)

        prop_dicts = [
            {"clogp": d.clogp, "qed": d.qed, "tpsa": d.tpsa, "mw": d.molecular_weight}
            for d in descs
        ]
        mmps = find_matched_molecular_pairs(mols, prop_dicts, min_core_atoms=5)

        model = self.ensure_calibrated_adme(epochs=adme_epochs)
        adme_preds = [
            model.predict_with_uncertainty(m, mc_samples=8, seed=self.seed + i)
            for i, m in enumerate(mols)
        ]
        pk_profiles = [
            simulate_two_compartment_pk(adme, mw=d.molecular_weight, dose_mg=100.0)
            for adme, d in zip(adme_preds, descs)
        ]

        conformers = [embed_conformer(m, seed=self.seed + i) for i, m in enumerate(mols)]
        docking_results = [
            dock_ligand(m, self.pocket, conformer=c, n_starts=5, n_poses=2, seed=self.seed + i)
            for i, (m, c) in enumerate(zip(mols, conformers))
        ]

        protac_curve = simulate_protac_hook_effect(
            ProtacSpec(
                name="Kinase_CRBN_PROTAC",
                kd_poi_nm=12.0,
                kd_e3_nm=28.0,
                cooperativity_alpha=12.0,
                linker_contour_angstrom=18.0,
                required_span_angstrom=12.0,
            ),
            n_points=45,
        )

        # Pick best-docking seed for de novo Pareto expansion
        best_seed_idx = min(
            range(len(mols)),
            key=lambda i: docking_results[i][0].affinity_kcal if docking_results[i] else 0.0,
        )
        opt_trace = run_pareto_optimization(
            seed_smiles=mols[best_seed_idx].canonical_smiles,
            adme_model=model,
            pocket=self.pocket,
            generations=opt_generations,
            pop_size=opt_pop_size,
            seed=self.seed,
        )

        return CascadeResult(
            molecules=mols,
            scaffolds=scaffolds,
            descriptors=descs,
            clusters=clusters,
            diverse_indices=diverse,
            mmps=mmps,
            adme_predictions=adme_preds,
            pk_profiles=pk_profiles,
            conformers=conformers,
            docking_results=docking_results,
            protac_curve=protac_curve,
            optimization_trace=opt_trace,
            calibration_metrics=dict(self.calibration_metrics),
        )


def export_cascade_json(result: CascadeResult, output_path: str | Path) -> Path:
    """Serialize key cascade tables and optimization Pareto front to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "calibration_metrics": result.calibration_metrics,
        "compounds": [
            {
                "name": m.name,
                "canonical_smiles": m.canonical_smiles,
                "murcko_scaffold": scaf,
                "descriptors": asdict(desc),
                "adme": {
                    "log_s": adme.log_s,
                    "log_d_74": adme.log_d_74,
                    "cl_int_ul_min_mg": adme.cl_int_ul_min_mg,
                    "papp_caco2": adme.papp_caco2,
                    "log_bb": adme.log_bb,
                    "herg_pic50": adme.herg_pic50,
                },
                "pk_summary": {
                    "cmax_ug_l": pk.cmax_ug_l,
                    "tmax_h": pk.tmax_h,
                    "auc_0_inf": pk.auc_0_inf,
                    "bioavailability_f": pk.bioavailability_f,
                    "therapeutic_margin": pk.therapeutic_margin,
                },
                "top_docking_kcal": dock[0].affinity_kcal if dock else None,
            }
            for m, scaf, desc, adme, pk, dock in zip(
                result.molecules,
                result.scaffolds,
                result.descriptors,
                result.adme_predictions,
                result.pk_profiles,
                result.docking_results,
            )
        ],
        "clusters": {
            "threshold": result.clusters.threshold,
            "num_clusters": len(result.clusters.clusters),
            "gini_diversity": result.clusters.gini_diversity,
            "clusters": result.clusters.clusters,
        },
        "pareto_front": [asdict(c) for c in result.optimization_trace.pareto_front],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
