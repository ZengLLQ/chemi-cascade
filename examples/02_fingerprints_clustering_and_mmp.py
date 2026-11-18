"""Example 02: Morgan ECFP4/FCFP4 Fingerprints, Butina Clustering & Matched Molecular Pairs."""

from __future__ import annotations

from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.fingerprints import (
    butina_clustering,
    compute_morgan_fingerprint,
    find_matched_molecular_pairs,
    maxmin_diverse_selection,
    pairwise_similarity_matrix,
)
from chemi_cascade.pipeline import DEFAULT_SERIES_SMILES
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.visualizer import render_similarity_matrix


def main() -> None:
    """Cluster a medicinal-chemistry series and extract Matched Molecular Pairs."""
    mols = [parse_smiles(smi, name=name) for name, smi in DEFAULT_SERIES_SMILES]
    fps = [compute_morgan_fingerprint(m, radius=2, nbits=2048) for m in mols]
    sim = pairwise_similarity_matrix(fps, metric="tanimoto")
    print(render_similarity_matrix([m.name for m in mols], sim))

    clusters = butina_clustering(fps, similarity_threshold=0.55)
    print(f"\nButina clusters (Gini diversity={clusters.gini_diversity:.3f}):")
    for cid, mem in enumerate(clusters.clusters):
        print(f"  Cluster {cid}: {[mols[i].name for i in mem]}")

    diverse = maxmin_diverse_selection(fps, k=4, seed_index=0)
    print(f"MaxMin Diverse 4: {[mols[i].name for i in diverse]}")

    props = [{"clogp": compute_descriptors(m).clogp, "qed": compute_descriptors(m).qed} for m in mols]
    mmps = find_matched_molecular_pairs(mols, props)
    print(f"\nDiscovered {len(mmps)} Matched Molecular Pairs:")
    for pair in mmps[:6]:
        print(
            f"  {mols[pair.from_idx].name} -> {mols[pair.to_idx].name} | "
            f"core={pair.core_smiles} | dLogP={pair.delta_properties['clogp']:+.3f}"
        )


if __name__ == "__main__":
    main()
