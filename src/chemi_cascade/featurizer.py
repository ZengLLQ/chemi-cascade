"""Molecular graph tensor featurizer and Bemis-Murcko scaffold dataset splitter.

Converts ``MoleculeGraph`` instances into dense NumPy node/edge/distance tensors
suitable for message-passing neural networks (MPNN), graph attention (GAT), and
spatial graph transformers, and provides deterministic Bemis-Murcko scaffold
partitioning to prevent analog leakage during ADME QSAR evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from chemi_cascade.descriptors import all_pairs_shortest_paths, compute_gasteiger_charges
from chemi_cascade.topology import murcko_scaffold_smiles
from chemi_cascade.types import MoleculeGraph, get_element

ATOM_ELEMENTS: Tuple[str, ...] = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B")
HYBRIDIZATIONS: Tuple[str, ...] = ("sp3", "sp2", "sp")

NODE_FEATURE_DIM = 32
EDGE_FEATURE_DIM = 10


@dataclass
class GraphTensor:
    """Tensorized single molecular graph for differentiable MPNN/GAT computation."""

    node_features: NDArray[np.float64]
    edge_src: NDArray[np.int32]
    edge_dst: NDArray[np.int32]
    edge_features: NDArray[np.float64]
    shortest_paths: NDArray[np.int32]
    num_atoms: int
    num_directed_edges: int


def featurize_atom(atom_idx: int, mol: MoleculeGraph) -> NDArray[np.float64]:
    """Encode a heavy atom into a 32-dimensional continuous & one-hot feature vector."""
    atom = mol.atoms[atom_idx]
    vec = np.zeros(NODE_FEATURE_DIM, dtype=np.float64)

    # 0..10: Element one-hot (10 known + 1 other)
    if atom.symbol in ATOM_ELEMENTS:
        vec[ATOM_ELEMENTS.index(atom.symbol)] = 1.0
    else:
        vec[10] = 1.0

    # 11..16: Heavy degree one-hot (0..5+)
    deg = min(5, mol.degree(atom_idx))
    vec[11 + deg] = 1.0

    # 17..19: Hybridization one-hot
    if atom.hybridization in HYBRIDIZATIONS:
        vec[17 + HYBRIDIZATIONS.index(atom.hybridization)] = 1.0

    # 20..24: Total hydrogens one-hot (0..4+)
    th = min(4, atom.total_h)
    vec[20 + th] = 1.0

    # 25..31: Electronic & topological scalars
    ed = get_element(atom.symbol)
    vec[25] = 1.0 if atom.is_aromatic else 0.0
    vec[26] = 1.0 if atom.in_ring else 0.0
    vec[27] = float(np.clip(atom.formal_charge / 2.0, -1.0, 1.0))
    vec[28] = float(np.clip(atom.partial_charge, -1.0, 1.0))
    vec[29] = (ed.electronegativity - 2.2) / 1.8
    vec[30] = ed.atomic_mass / 127.0
    vec[31] = 1.0 if (atom.symbol in {"N", "O"} and atom.total_h > 0) else 0.0
    return vec


def featurize_bond(
    mol: MoleculeGraph,
    u: int,
    v: int,
    dist_mat: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Encode a directed bond ``u -> v`` into a 10-dimensional edge feature vector."""
    bond = mol.get_bond(u, v)
    vec = np.zeros(EDGE_FEATURE_DIM, dtype=np.float64)
    if bond is None:
        return vec

    # 0..3: Bond type one-hot (single, double, triple, aromatic)
    if bond.is_aromatic:
        vec[3] = 1.0
    elif bond.order >= 2.9:
        vec[2] = 1.0
    elif bond.order >= 1.9:
        vec[1] = 1.0
    else:
        vec[0] = 1.0

    vec[4] = 1.0 if bond.is_conjugated else 0.0
    vec[5] = 1.0 if bond.in_ring else 0.0
    vec[6] = bond.order / 3.0
    vec[7] = float(abs(mol.atoms[u].partial_charge - mol.atoms[v].partial_charge))
    vec[8] = 1.0 if mol.atoms[u].symbol != mol.atoms[v].symbol else 0.0
    # Ring membership size hint
    shared_rings = set(mol.atoms[u].ring_sizes) & set(mol.atoms[v].ring_sizes)
    vec[9] = (min(shared_rings) / 8.0) if shared_rings else float(dist_mat[u, v]) / 10.0
    return vec


def featurize_molecule(mol: MoleculeGraph) -> GraphTensor:
    """Convert ``MoleculeGraph`` into a ``GraphTensor`` for GNN forward/backward passes."""
    if all(abs(a.partial_charge) < 1e-12 for a in mol.atoms):
        compute_gasteiger_charges(mol)

    n = mol.num_atoms
    dist_mat = all_pairs_shortest_paths(mol)
    node_feats = np.vstack([featurize_atom(i, mol) for i in range(n)]) if n > 0 else np.zeros((0, NODE_FEATURE_DIM))

    src_list: List[int] = []
    dst_list: List[int] = []
    edge_feats_list: List[NDArray[np.float64]] = []
    for b in mol.bonds:
        for u, v in ((b.atom_a, b.atom_b), (b.atom_b, b.atom_a)):
            src_list.append(u)
            dst_list.append(v)
            edge_feats_list.append(featurize_bond(mol, u, v, dist_mat))

    if edge_feats_list:
        edge_src = np.asarray(src_list, dtype=np.int32)
        edge_dst = np.asarray(dst_list, dtype=np.int32)
        edge_feats = np.vstack(edge_feats_list)
    else:
        edge_src = np.zeros(0, dtype=np.int32)
        edge_dst = np.zeros(0, dtype=np.int32)
        edge_feats = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float64)

    return GraphTensor(
        node_features=node_feats,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_features=edge_feats,
        shortest_paths=dist_mat,
        num_atoms=n,
        num_directed_edges=len(src_list),
    )


def scaffold_split(
    mols: Sequence[MoleculeGraph],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Partition ``mols`` into (train, val, test) grouped by Bemis-Murcko scaffolds.

    Largest scaffold groups are placed into training first (standard DeepChem
    balanced-scaffold protocol) with deterministic tie-breaking so structurally
    novel scaffolds land cleanly in validation and test splits.
    """
    n = len(mols)
    if n == 0:
        return [], [], []

    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = {}
    for idx, mol in enumerate(mols):
        scaf = murcko_scaffold_smiles(mol, generic=False)
        key = scaf if scaf else f"__acyclic_{mol.canonical_smiles}"
        groups.setdefault(key, []).append(idx)

    # Sort scaffold buckets by descending size, breaking ties deterministically
    buckets = list(groups.values())
    tie_noise = rng.random(len(buckets))
    ordered_buckets = [
        b for _, b in sorted(
            zip(tie_noise, buckets),
            key=lambda pair: (-len(pair[1]), pair[0]),
        )
    ]

    train_cut = int(round(train_frac * n))
    val_cut = train_cut + int(round(val_frac * n))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for bucket in ordered_buckets:
        if len(train_idx) + len(bucket) <= train_cut or len(train_idx) == 0:
            train_idx.extend(bucket)
        elif len(val_idx) + len(bucket) <= (val_cut - train_cut):
            val_idx.extend(bucket)
        else:
            test_idx.extend(bucket)

    return train_idx, val_idx, test_idx
