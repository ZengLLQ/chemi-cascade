"""Circular Morgan ECFP/FCFP fingerprints, similarity metrics, Butina clustering & MMPs.

Provides:
- Morgan circular fingerprints (ECFP4 / ECFP6 / FCFP4) with deterministic 64-bit
  hash invariants, duplicate-neighborhood pruning, bit-level provenance mapping,
  and topological torsion fingerprints.
- Vectorized pairwise similarity matrices (Tanimoto, Dice, Cosine, Tversky).
- Taylor-Butina sphere-exclusion clustering and MaxMin diverse subset selection.
- Scaffold Gini diversity index.
- Single-cut Matched Molecular Pair (MMP) fragmentation and property-delta analysis.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from chemi_cascade.smiles import to_canonical_smiles
from chemi_cascade.types import (
    HALOGENS,
    Atom,
    Bond,
    ClusterResult,
    FingerprintVector,
    MatchedMolecularPair,
    MoleculeGraph,
)


def _hash_tuple(values: Sequence[int]) -> int:
    """Deterministic 64-bit integer hash over an integer tuple."""
    payload = ",".join(str(v) for v in values).encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _atom_morgan_invariant(atom: Atom, mol: MoleculeGraph, feature_mode: bool) -> int:
    """Compute radius-0 atomic invariant for ECFP (connectivity) or FCFP (pharmacophore)."""
    if feature_mode:
        is_hbd = 1 if (atom.symbol in {"N", "O"} and atom.total_h > 0) else 0
        is_hba = 1 if (atom.symbol in {"N", "O"} and atom.formal_charge <= 0) else 0
        is_pos = 1 if atom.formal_charge > 0 else 0
        is_neg = 1 if atom.formal_charge < 0 else 0
        is_arom = 1 if atom.is_aromatic else 0
        is_hal = 1 if atom.symbol in HALOGENS else 0
        return _hash_tuple((is_hbd, is_hba, is_pos, is_neg, is_arom, is_hal, mol.degree(atom.idx)))

    deg = mol.degree(atom.idx)
    non_h_deg = deg
    total_conn = deg + atom.total_h
    return _hash_tuple((
        atom.atomic_number,
        non_h_deg,
        total_conn,
        atom.formal_charge + 8,
        atom.isotope or 0,
        1 if atom.in_ring else 0,
        1 if atom.is_aromatic else 0,
    ))


def compute_morgan_fingerprint(
    mol: MoleculeGraph,
    radius: int = 2,
    nbits: int = 2048,
    use_features: bool = False,
) -> FingerprintVector:
    """Compute Morgan circular fingerprint (ECFP4 when radius=2, ECFP6 when radius=3).

    Maintains bit-to-``(atom_idx, radius)`` provenance for substructure attribution.
    """
    n = mol.num_atoms
    bits = np.zeros(nbits, dtype=np.int32)
    counts = np.zeros(nbits, dtype=np.int32)
    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    if n == 0:
        kind = f"FCFP{2 * radius}" if use_features else f"ECFP{2 * radius}"
        return FingerprintVector(kind=kind, radius=radius, nbits=nbits, bits=bits, counts=counts)

    invariants = [_atom_morgan_invariant(a, mol, use_features) for a in mol.atoms]
    seen_env_hashes: Set[Tuple[int, FrozenBondSet]] = set()  # type: ignore[name-defined]
    atom_bonds: List[Set[int]] = [{b_idx for b_idx in mol.adjacency.get(i, {}).values()} for i in range(n)]

    # Radius 0 contributions
    for i, inv in enumerate(invariants):
        b = int(inv % nbits)
        bits[b] = 1
        counts[b] += 1
        bit_info.setdefault(b, []).append((i, 0))
        seen_env_hashes.add((inv, frozenset()))

    # Iterative circular expansion r = 1..radius
    for r in range(1, radius + 1):
        next_inv: List[int] = []
        next_bonds: List[Set[int]] = []
        for i in range(n):
            nb_pairs: List[Tuple[int, int]] = []
            env_bonds = set(atom_bonds[i])
            for nb in mol.neighbors(i):
                bond = mol.get_bond(i, nb)
                bo_code = 4 if (bond and bond.is_aromatic) else int(round((bond.order if bond else 1.0)))
                nb_pairs.append((bo_code, invariants[nb]))
                env_bonds |= atom_bonds[nb]
            nb_pairs.sort()
            flat: List[int] = [r, invariants[i]]
            for bo_c, nb_inv in nb_pairs:
                flat.append(bo_c)
                flat.append(nb_inv)
            new_h = _hash_tuple(flat)
            next_inv.append(new_h)
            next_bonds.append(env_bonds)

            env_key = (new_h, frozenset(env_bonds))
            if env_key not in seen_env_hashes:
                seen_env_hashes.add(env_key)
                bit_idx = int(new_h % nbits)
                bits[bit_idx] = 1
                counts[bit_idx] += 1
                bit_info.setdefault(bit_idx, []).append((i, r))
        invariants = next_inv
        atom_bonds = next_bonds

    kind = f"FCFP{2 * radius}" if use_features else f"ECFP{2 * radius}"
    return FingerprintVector(
        kind=kind,
        radius=radius,
        nbits=nbits,
        bits=bits,
        counts=counts,
        bit_info=bit_info,
    )


FrozenBondSet = frozenset[int]


def compute_topological_torsion(mol: MoleculeGraph, nbits: int = 1024) -> FingerprintVector:
    """Compute 4-atom linear path Topological Torsion fingerprint."""
    n = mol.num_atoms
    bits = np.zeros(nbits, dtype=np.int32)
    counts = np.zeros(nbits, dtype=np.int32)
    if n < 4:
        return FingerprintVector(kind="Torsion", radius=3, nbits=nbits, bits=bits, counts=counts)

    def atom_code(i: int) -> int:
        a = mol.atoms[i]
        pi_e = 1 if (a.is_aromatic or a.hybridization == "sp2") else (2 if a.hybridization == "sp" else 0)
        return (a.atomic_number * 31 + mol.degree(i) * 7 + pi_e)

    seen_paths: Set[Tuple[int, int, int, int]] = set()
    for a in range(n):
        for b in mol.neighbors(a):
            for c in mol.neighbors(b):
                if c == a:
                    continue
                for d in mol.neighbors(c):
                    if d in (a, b):
                        continue
                    path = (a, b, c, d)
                    rev = (d, c, b, a)
                    canon = min(path, rev)
                    if canon in seen_paths:
                        continue
                    seen_paths.add(canon)
                    codes = [atom_code(x) for x in canon]
                    h = _hash_tuple(codes)
                    bit_idx = int(h % nbits)
                    bits[bit_idx] = 1
                    counts[bit_idx] += 1

    return FingerprintVector(kind="Torsion", radius=3, nbits=nbits, bits=bits, counts=counts)


def tanimoto_similarity(fp_a: FingerprintVector, fp_b: FingerprintVector) -> float:
    """Compute exact binary Tanimoto similarity between two fingerprints."""
    a = fp_a.bits > 0
    b = fp_b.bits > 0
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    if union == 0:
        return 1.0 if int(np.sum(a)) == 0 and int(np.sum(b)) == 0 else 0.0
    return float(inter / union)


def tversky_similarity(
    fp_a: FingerprintVector,
    fp_b: FingerprintVector,
    alpha: float = 0.8,
    beta: float = 0.2,
) -> float:
    """Compute asymmetric Tversky index (substructure-biased when alpha > beta)."""
    a = fp_a.bits > 0
    b = fp_b.bits > 0
    inter = float(np.sum(a & b))
    only_a = float(np.sum(a & (~b)))
    only_b = float(np.sum((~a) & b))
    denom = inter + alpha * only_a + beta * only_b
    return float(inter / denom) if denom > 0.0 else 0.0


def pairwise_similarity_matrix(
    fps: Sequence[FingerprintVector],
    metric: str = "tanimoto",
) -> NDArray[np.float64]:
    """Compute vectorized N x N similarity matrix over a fingerprint library."""
    m = len(fps)
    if m == 0:
        return np.zeros((0, 0), dtype=np.float64)
    mat = np.vstack([(fp.bits > 0).astype(np.float64) for fp in fps])
    inter = mat @ mat.T
    norms = np.sum(mat, axis=1)

    if metric == "tanimoto":
        union = norms[:, None] + norms[None, :] - inter
        sim = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    elif metric == "dice":
        denom = norms[:, None] + norms[None, :]
        sim = np.divide(2.0 * inter, denom, out=np.ones_like(inter), where=denom > 0)
    elif metric == "cosine":
        denom = np.sqrt(np.outer(norms, norms))
        sim = np.divide(inter, denom, out=np.ones_like(inter), where=denom > 0)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric!r}")

    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0.0, 1.0)


def butina_clustering(
    fps: Sequence[FingerprintVector],
    similarity_threshold: float = 0.65,
) -> ClusterResult:
    """Cluster molecules using Taylor-Butina sphere-exclusion on Tanimoto distance.

    Centroid candidates are ordered by descending neighbor count within the
    similarity sphere ``sim >= similarity_threshold`` so dense chemical series
    nucleate cleanly.
    """
    n = len(fps)
    if n == 0:
        return ClusterResult(
            threshold=similarity_threshold,
            clusters=[],
            centroids=[],
            assignments=[],
            gini_diversity=0.0,
        )

    sim_mat = pairwise_similarity_matrix(fps, metric="tanimoto")
    neighbor_lists: List[List[int]] = []
    for i in range(n):
        nbs = [j for j in range(n) if j != i and sim_mat[i, j] >= similarity_threshold]
        neighbor_lists.append(nbs)

    order = sorted(range(n), key=lambda i: (-len(neighbor_lists[i]), i))
    assigned = [False] * n
    clusters: List[List[int]] = []
    centroids: List[int] = []
    assignments = [-1] * n

    for idx in order:
        if assigned[idx]:
            continue
        cluster = [idx]
        assigned[idx] = True
        for nb in neighbor_lists[idx]:
            if not assigned[nb]:
                assigned[nb] = True
                cluster.append(nb)
        cid = len(clusters)
        centroids.append(idx)
        clusters.append(cluster)
        for mem in cluster:
            assignments[mem] = cid

    gini = compute_gini_diversity([len(c) for c in clusters])
    return ClusterResult(
        threshold=similarity_threshold,
        clusters=clusters,
        centroids=centroids,
        assignments=assignments,
        gini_diversity=gini,
    )


def compute_gini_diversity(cluster_sizes: Sequence[int]) -> float:
    """Compute Gini diversity coefficient in [0, 1] (0 = uniform, ->1 = dominated)."""
    if not cluster_sizes:
        return 0.0
    arr = np.sort(np.asarray(cluster_sizes, dtype=np.float64))
    total = float(np.sum(arr))
    k = len(arr)
    if total <= 0.0 or k <= 1:
        return 0.0
    idx = np.arange(1, k + 1, dtype=np.float64)
    gini = float(np.sum((2.0 * idx - k - 1.0) * arr) / (k * total))
    return max(0.0, min(1.0, gini))


def maxmin_diverse_selection(
    fps: Sequence[FingerprintVector],
    k: int,
    seed_index: int = 0,
) -> List[int]:
    """Select ``k`` maximally diverse molecule indices via greedy MaxMin Tanimoto."""
    n = len(fps)
    if k <= 0 or n == 0:
        return []
    k = min(k, n)
    sim = pairwise_similarity_matrix(fps, metric="tanimoto")
    dist = 1.0 - sim

    selected = [max(0, min(seed_index, n - 1))]
    min_dist = dist[selected[0]].copy()
    while len(selected) < k:
        for s in selected:
            min_dist[s] = -1.0
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dist[nxt])
    return selected


def _fragment_single_cut(mol: MoleculeGraph, bond: Bond) -> Optional[Tuple[str, str]]:
    """Split ``mol`` across an acyclic single bond into two canonical fragment SMILES."""
    if bond.in_ring or bond.is_aromatic or abs(bond.order - 1.0) > 0.2:
        return None
    n = mol.num_atoms
    # BFS from atom_a without traversing bond
    comp_a: Set[int] = {bond.atom_a}
    stack = [bond.atom_a]
    while stack:
        cur = stack.pop()
        for nb in mol.neighbors(cur):
            if (cur == bond.atom_a and nb == bond.atom_b) or (cur == bond.atom_b and nb == bond.atom_a):
                continue
            if nb not in comp_a:
                comp_a.add(nb)
                stack.append(nb)
    comp_b = set(range(n)) - comp_a
    if not comp_a or not comp_b:
        return None

    def sub_smiles(keep: Set[int]) -> str:
        idx_map = {old: new for new, old in enumerate(sorted(keep))}
        atoms = [
            Atom(
                idx=idx_map[old],
                symbol=mol.atoms[old].symbol,
                atomic_number=mol.atoms[old].atomic_number,
                formal_charge=mol.atoms[old].formal_charge,
                is_aromatic=mol.atoms[old].is_aromatic,
                explicit_h=mol.atoms[old].explicit_h,
                implicit_h=mol.atoms[old].implicit_h,
                hybridization=mol.atoms[old].hybridization,
                in_ring=mol.atoms[old].in_ring,
                ring_sizes=mol.atoms[old].ring_sizes,
            )
            for old in sorted(keep)
        ]
        bonds: List[Bond] = []
        adj: Dict[int, Dict[int, int]] = {i: {} for i in range(len(atoms))}
        for b in mol.bonds:
            if b.atom_a in keep and b.atom_b in keep:
                u, v = idx_map[b.atom_a], idx_map[b.atom_b]
                bi = len(bonds)
                bonds.append(
                    Bond(
                        idx=bi,
                        atom_a=u,
                        atom_b=v,
                        order=b.order,
                        is_aromatic=b.is_aromatic,
                        is_conjugated=b.is_conjugated,
                        in_ring=b.in_ring,
                    )
                )
                adj[u][v] = bi
                adj[v][u] = bi
        sub = MoleculeGraph(smiles="", atoms=atoms, bonds=bonds, adjacency=adj)
        return to_canonical_smiles(sub)

    sa = sub_smiles(comp_a)
    sb = sub_smiles(comp_b)
    return (sa, sb)


def find_matched_molecular_pairs(
    mols: Sequence[MoleculeGraph],
    properties: Sequence[Dict[str, float]],
    min_core_atoms: int = 5,
) -> List[MatchedMolecularPair]:
    """Identify Hussain-Rea single-cut Matched Molecular Pairs and property deltas."""
    # Map core_smiles -> list of (mol_idx, r_group_smiles)
    index: Dict[str, List[Tuple[int, str]]] = {}
    for idx, mol in enumerate(mols):
        seen_cuts: Set[Tuple[str, str]] = set()
        for b in mol.bonds:
            cut = _fragment_single_cut(mol, b)
            if cut is None:
                continue
            frag_a, frag_b = cut
            # Consider larger or cyclic side as candidate core
            for core_s, r_s in ((frag_a, frag_b), (frag_b, frag_a)):
                # Rough heavy atom count check by counting alpha tokens
                core_heavy = sum(1 for ch in core_s if ch.isalpha() and ch not in {"H", "l", "r"})
                if core_heavy >= min_core_atoms and (core_s, r_s) not in seen_cuts:
                    seen_cuts.add((core_s, r_s))
                    index.setdefault(core_s, []).append((idx, r_s))

    mmps: List[MatchedMolecularPair] = []
    seen_pairs: Set[Tuple[str, int, int]] = set()
    for core_s, entries in index.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            idx_i, r_i = entries[i]
            for j in range(i + 1, len(entries)):
                idx_j, r_j = entries[j]
                if idx_i == idx_j or r_i == r_j:
                    continue
                key = (core_s, min(idx_i, idx_j), max(idx_i, idx_j))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pi = properties[idx_i]
                pj = properties[idx_j]
                deltas = {
                    k: round(pj[k] - pi[k], 4)
                    for k in pi
                    if k in pj
                }
                mmps.append(
                    MatchedMolecularPair(
                        core_smiles=core_s,
                        from_idx=idx_i,
                        to_idx=idx_j,
                        from_fragment=r_i,
                        to_fragment=r_j,
                        delta_properties=deltas,
                    )
                )
    return mmps
