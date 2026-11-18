"""De novo molecular optimization via valence-safe graph mutations, MCTS, and NSGA-II.

Combines:
1. Valence-safe chemical graph mutator (bioisosteric ring heteroatom swaps,
   substituent grafting, terminal group pruning, and halogen/polar capping)
   verified by round-trip SMILES canonicalization and structural alert checks.
2. NSGA-II non-dominated sorting, crowding distance, and Pareto hypervolume
   tracking across conflicting medicinal-chemistry objectives:
   - Docking binding affinity (minimize DeltaG_bind kcal/mol)
   - Drug-likeness (maximize Bickerton QED)
   - Aqueous solubility & synthetic accessibility (maximize logS, minimize SA)
   - Safety / metabolic stability (minimize CL_int, penalize hERG pIC50 > 5.8).
3. Monte Carlo Tree Search (UCB1) & Evolutionary NSGA-II optimization loops.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from chemi_cascade.adme_pk import build_calibrated_adme_model
from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.docking import build_kinase_hinge_pocket, dock_ligand
from chemi_cascade.gnn import MolecularMPNN
from chemi_cascade.smiles import parse_smiles, to_canonical_smiles
from chemi_cascade.types import (
    Atom,
    Bond,
    MoleculeGraph,
    OptimizationCandidate,
    OptimizationTrace,
    ReceptorPocket,
    get_element,
)

# Terminal substituent fragments grafted onto an available H-bearing atom:
# (label, SMILES fragment where atom 0 attaches to target atom)
_GRAFT_FRAGMENTS: Tuple[Tuple[str, str], ...] = (
    ("add_F", "F"),
    ("add_Cl", "Cl"),
    ("add_Me", "C"),
    ("add_OH", "O"),
    ("add_NH2", "N"),
    ("add_OMe", "OC"),
    ("add_CN", "C#N"),
    ("add_amide", "C(=O)N"),
    ("add_CF3", "C(F)(F)F"),
)


def _attach_fragment(mol: MoleculeGraph, target_idx: int, frag_smiles: str) -> Optional[MoleculeGraph]:
    """Attach a small fragment SMILES onto ``target_idx`` if valence permits."""
    target = mol.atoms[target_idx]
    if target.total_h < 1 or target.formal_charge != 0:
        return None
    if target.symbol not in {"C", "N"}:
        return None

    frag = parse_smiles(frag_smiles, canonicalize=False)
    if frag.num_atoms == 0:
        return None

    clone = mol.clone()
    offset = clone.num_atoms
    for fa in frag.atoms:
        new_idx = offset + fa.idx
        clone.atoms.append(
            Atom(
                idx=new_idx,
                symbol=fa.symbol,
                atomic_number=fa.atomic_number,
                formal_charge=fa.formal_charge,
                is_aromatic=fa.is_aromatic,
                explicit_h=fa.explicit_h,
                implicit_h=fa.implicit_h,
                hybridization=fa.hybridization,
                in_ring=fa.in_ring,
                ring_sizes=fa.ring_sizes,
            )
        )
        clone.adjacency[new_idx] = {}

    for fb in frag.bonds:
        u = offset + fb.atom_a
        v = offset + fb.atom_b
        b_idx = len(clone.bonds)
        clone.bonds.append(
            Bond(
                idx=b_idx,
                atom_a=u,
                atom_b=v,
                order=fb.order,
                is_aromatic=fb.is_aromatic,
                is_conjugated=fb.is_conjugated,
                in_ring=fb.in_ring,
            )
        )
        clone.adjacency[u][v] = b_idx
        clone.adjacency[v][u] = b_idx

    # Connect target_idx to offset (atom 0 of fragment)
    b_link = len(clone.bonds)
    clone.bonds.append(Bond(idx=b_link, atom_a=target_idx, atom_b=offset, order=1.0))
    clone.adjacency[target_idx][offset] = b_link
    clone.adjacency[offset][target_idx] = b_link

    # Decrement H on target and attachment atom
    if clone.atoms[target_idx].implicit_h > 0:
        clone.atoms[target_idx].implicit_h -= 1
    elif clone.atoms[target_idx].explicit_h > 0:
        clone.atoms[target_idx].explicit_h -= 1
    if clone.atoms[offset].implicit_h > 0:
        clone.atoms[offset].implicit_h -= 1

    smi = to_canonical_smiles(clone)
    return parse_smiles(smi, canonicalize=True)


def _prune_terminal_atom(mol: MoleculeGraph, leaf_idx: int) -> Optional[MoleculeGraph]:
    """Remove a non-ring degree-1 heavy atom (and update neighbor H count)."""
    if mol.num_atoms <= 6 or mol.degree(leaf_idx) != 1 or mol.atoms[leaf_idx].in_ring:
        return None
    nb_idx = mol.neighbors(leaf_idx)[0]
    keep = [i for i in range(mol.num_atoms) if i != leaf_idx]
    idx_map = {old: new for new, old in enumerate(keep)}

    atoms: List[Atom] = []
    for old in keep:
        a = mol.atoms[old]
        atoms.append(
            Atom(
                idx=idx_map[old],
                symbol=a.symbol,
                atomic_number=a.atomic_number,
                formal_charge=a.formal_charge,
                is_aromatic=a.is_aromatic,
                explicit_h=a.explicit_h,
                implicit_h=a.implicit_h + (1 if old == nb_idx else 0),
                hybridization=a.hybridization,
                in_ring=a.in_ring,
                ring_sizes=a.ring_sizes,
            )
        )
    bonds: List[Bond] = []
    adj: Dict[int, Dict[int, int]] = {i: {} for i in range(len(atoms))}
    for b in mol.bonds:
        if b.atom_a != leaf_idx and b.atom_b != leaf_idx:
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

    tmp = MoleculeGraph(smiles="", atoms=atoms, bonds=bonds, adjacency=adj)
    smi = to_canonical_smiles(tmp)
    return parse_smiles(smi, canonicalize=True)


def _bioisostere_swap(mol: MoleculeGraph, atom_idx: int) -> Optional[MoleculeGraph]:
    """Perform a valence-preserving bioisosteric atom swap (e.g. aromatic CH <-> N)."""
    a = mol.atoms[atom_idx]
    if a.formal_charge != 0:
        return None

    clone = mol.clone()
    ca = clone.atoms[atom_idx]
    # Aromatic CH <-> pyridine N
    if a.is_aromatic and a.symbol == "C" and a.total_h == 1 and mol.degree(atom_idx) == 2:
        ed = get_element("N")
        ca.symbol = "N"
        ca.atomic_number = ed.atomic_number
        ca.implicit_h = 0
        ca.explicit_h = 0
    elif a.is_aromatic and a.symbol == "N" and a.total_h == 0 and mol.degree(atom_idx) == 2:
        ed = get_element("C")
        ca.symbol = "C"
        ca.atomic_number = ed.atomic_number
        ca.implicit_h = 1
        ca.explicit_h = 0
    elif a.symbol == "F" and mol.degree(atom_idx) == 1:
        ed = get_element("Cl")
        ca.symbol = "Cl"
        ca.atomic_number = ed.atomic_number
    elif a.symbol == "Cl" and mol.degree(atom_idx) == 1:
        ed = get_element("F")
        ca.symbol = "F"
        ca.atomic_number = ed.atomic_number
    else:
        return None

    smi = to_canonical_smiles(clone)
    return parse_smiles(smi, canonicalize=True)


def generate_neighbors(
    mol: MoleculeGraph,
    rng: np.random.Generator,
    max_neighbors: int = 8,
) -> List[Tuple[str, MoleculeGraph]]:
    """Generate up to ``max_neighbors`` valid, unique chemical graph neighbors."""
    proposals: List[Tuple[str, MoleculeGraph]] = []
    seen_canon: Set[str] = {mol.canonical_smiles or to_canonical_smiles(mol)}

    # 1. Attachment sites (H-bearing C or N)
    attach_sites = [
        a.idx for a in mol.atoms if a.total_h >= 1 and a.symbol in {"C", "N"} and a.formal_charge == 0
    ]
    rng.shuffle(attach_sites)
    for site in attach_sites[:6]:
        frag_label, frag_smi = _GRAFT_FRAGMENTS[int(rng.integers(0, len(_GRAFT_FRAGMENTS)))]
        try:
            mut = _attach_fragment(mol, site, frag_smi)
        except Exception:
            mut = None
        if mut is not None and mut.canonical_smiles not in seen_canon and mut.num_atoms <= 42:
            seen_canon.add(mut.canonical_smiles)
            proposals.append((frag_label, mut))
            if len(proposals) >= max_neighbors:
                return proposals

    # 2. Bioisostere swaps
    swap_sites = [
        a.idx for a in mol.atoms
        if (a.is_aromatic and a.symbol in {"C", "N"}) or a.symbol in {"F", "Cl"}
    ]
    rng.shuffle(swap_sites)
    for site in swap_sites[:5]:
        try:
            mut = _bioisostere_swap(mol, site)
        except Exception:
            mut = None
        if mut is not None and mut.canonical_smiles not in seen_canon:
            seen_canon.add(mut.canonical_smiles)
            proposals.append(("bioisostere_swap", mut))
            if len(proposals) >= max_neighbors:
                return proposals

    # 3. Terminal leaf pruning
    leaf_sites = [a.idx for a in mol.atoms if mol.degree(a.idx) == 1 and not a.in_ring]
    rng.shuffle(leaf_sites)
    for site in leaf_sites[:4]:
        try:
            mut = _prune_terminal_atom(mol, site)
        except Exception:
            mut = None
        if mut is not None and mut.canonical_smiles not in seen_canon:
            seen_canon.add(mut.canonical_smiles)
            proposals.append(("prune_leaf", mut))
            if len(proposals) >= max_neighbors:
                return proposals

    return proposals


def evaluate_candidate(
    mol: MoleculeGraph,
    adme_model: MolecularMPNN,
    pocket: ReceptorPocket,
    generation: int = 0,
    parent_smiles: str = "",
    operation: str = "seed",
    seed: int = 42,
) -> OptimizationCandidate:
    """Score ``mol`` across descriptors, GNN ADME, and fast receptor docking."""
    desc = compute_descriptors(mol)
    adme = adme_model.predict_with_uncertainty(mol, mc_samples=6, seed=seed)
    poses = dock_ligand(mol, pocket, n_starts=3, n_poses=1, seed=seed)
    dock_kcal = poses[0].affinity_kcal if poses else 0.0

    # Composite Medicinal Chemistry MPO score in [0, 1]
    potency_term = min(1.0, max(0.0, (-dock_kcal - 2.5) / 6.5))
    qed_term = desc.qed
    sa_term = max(0.0, min(1.0, (7.0 - desc.sa_score) / 5.5))
    sol_term = max(0.0, min(1.0, (adme.log_s + 5.5) / 4.5))
    cl_term = max(0.0, min(1.0, (120.0 - adme.cl_int_ul_min_mg) / 110.0))
    herg_pen = 0.20 * max(0.0, adme.herg_pic50 - 5.8)
    alert_pen = 0.15 * len(desc.alerts)

    mpo = (
        0.34 * potency_term
        + 0.24 * qed_term
        + 0.16 * sa_term
        + 0.14 * sol_term
        + 0.12 * cl_term
        - herg_pen
        - alert_pen
    )
    mpo = float(max(0.0, min(1.0, mpo)))

    return OptimizationCandidate(
        smiles=mol.smiles,
        canonical_smiles=mol.canonical_smiles or to_canonical_smiles(mol),
        generation=generation,
        parent_smiles=parent_smiles,
        operation=operation,
        mpo_score=round(mpo, 4),
        docking_kcal=round(dock_kcal, 4),
        qed=round(desc.qed, 4),
        sa_score=round(desc.sa_score, 3),
        log_s=round(adme.log_s, 3),
        cl_int=round(adme.cl_int_ul_min_mg, 2),
        herg_pic50=round(adme.herg_pic50, 3),
        is_valid=(len(desc.alerts) == 0 and desc.lipinski_violations <= 1),
    )


def _dominates(a: OptimizationCandidate, b: OptimizationCandidate) -> bool:
    """Return True if candidate ``a`` Pareto-dominates ``b`` on (potency, QED, -SA, logS)."""
    va = (-a.docking_kcal, a.qed, -a.sa_score, a.log_s)
    vb = (-b.docking_kcal, b.qed, -b.sa_score, b.log_s)
    ge_all = all(x >= y for x, y in zip(va, vb))
    gt_one = any(x > y for x, y in zip(va, vb))
    return ge_all and gt_one


def non_dominated_sort(candidates: List[OptimizationCandidate]) -> List[List[OptimizationCandidate]]:
    """Assign NSGA-II Pareto ranks and crowding distances to ``candidates``."""
    n = len(candidates)
    if n == 0:
        return []

    dom_sets: List[List[int]] = [[] for _ in range(n)]
    dom_counts = [0] * n
    fronts_idx: List[List[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(candidates[i], candidates[j]):
                dom_sets[i].append(j)
                dom_counts[j] += 1
            elif _dominates(candidates[j], candidates[i]):
                dom_sets[j].append(i)
                dom_counts[i] += 1
        if dom_counts[i] == 0:
            candidates[i].pareto_rank = 0
            fronts_idx[0].append(i)

    rank = 0
    while fronts_idx[rank]:
        nxt_front: List[int] = []
        for i in fronts_idx[rank]:
            for j in dom_sets[i]:
                dom_counts[j] -= 1
                if dom_counts[j] == 0:
                    candidates[j].pareto_rank = rank + 1
                    nxt_front.append(j)
        rank += 1
        fronts_idx.append(nxt_front)

    fronts = [[candidates[i] for i in f] for f in fronts_idx if f]

    # Compute crowding distance per front
    for front in fronts:
        for c in front:
            c.crowding_distance = 0.0
        if len(front) <= 2:
            for c in front:
                c.crowding_distance = float("inf")
            continue
        objs = [
            lambda c: -c.docking_kcal,
            lambda c: c.qed,
            lambda c: -c.sa_score,
            lambda c: c.log_s,
        ]
        for obj_fn in objs:
            front.sort(key=obj_fn)
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            lo = obj_fn(front[0])
            hi = obj_fn(front[-1])
            span = max(hi - lo, 1e-9)
            for k in range(1, len(front) - 1):
                if math.isfinite(front[k].crowding_distance):
                    front[k].crowding_distance += (obj_fn(front[k + 1]) - obj_fn(front[k - 1])) / span

    return fronts


def compute_pareto_hypervolume_2d(front: Sequence[OptimizationCandidate]) -> float:
    """Compute exact 2D Pareto hypervolume in (-docking_kcal, QED) relative to (-2.0, 0.0)."""
    if not front:
        return 0.0
    ref_x, ref_y = -2.0, 0.0
    pts = sorted(
        [(max(ref_x, -c.docking_kcal), max(ref_y, c.qed)) for c in front],
        key=lambda p: p[0],
    )
    # Sweep from right to left taking running max y
    hv = 0.0
    best_y = ref_y
    prev_x = pts[-1][0]
    for x, y in reversed(pts):
        if x < prev_x:
            hv += (prev_x - x) * max(0.0, best_y - ref_y)
            prev_x = x
        best_y = max(best_y, y)
    hv += max(0.0, prev_x - ref_x) * max(0.0, best_y - ref_y)
    return round(float(hv), 4)


def run_pareto_optimization(
    seed_smiles: str,
    adme_model: Optional[MolecularMPNN] = None,
    pocket: Optional[ReceptorPocket] = None,
    generations: int = 5,
    pop_size: int = 12,
    method: str = "nsga2_mcts",
    seed: int = 42,
) -> OptimizationTrace:
    """Run hybrid MCTS + NSGA-II multi-objective de novo lead optimization."""
    rng = np.random.default_rng(seed)
    if adme_model is None:
        adme_model, _ = build_calibrated_adme_model(epochs=20, hidden_dim=32, seed=seed)
    if pocket is None:
        pocket = build_kinase_hinge_pocket()

    seed_mol = parse_smiles(seed_smiles, name="seed")
    seed_cand = evaluate_candidate(seed_mol, adme_model, pocket, generation=0, operation="seed", seed=seed)

    archive: Dict[str, OptimizationCandidate] = {seed_cand.canonical_smiles: seed_cand}
    mol_cache: Dict[str, MoleculeGraph] = {seed_cand.canonical_smiles: seed_mol}
    visit_counts: Dict[str, int] = {seed_cand.canonical_smiles: 1}
    total_visits = 1

    population: List[OptimizationCandidate] = [seed_cand]
    history_best_mpo: List[float] = [seed_cand.mpo_score]
    history_hv: List[float] = [compute_pareto_hypervolume_2d([seed_cand])]

    for gen in range(1, generations + 1):
        # UCB1 parent selection over top population members
        scored_parents = sorted(
            [
                (
                    c.mpo_score
                    + 0.35 * math.sqrt(math.log(total_visits + 1.0) / visit_counts.get(c.canonical_smiles, 1)),
                    i,
                    c,
                )
                for i, c in enumerate(population)
            ],
            reverse=True,
        )
        selected = [item[2] for item in scored_parents[:max(3, pop_size // 2)]]

        offspring: List[OptimizationCandidate] = []
        for parent in selected:
            visit_counts[parent.canonical_smiles] = visit_counts.get(parent.canonical_smiles, 0) + 1
            total_visits += 1
            pmol = mol_cache[parent.canonical_smiles]
            neighbors = generate_neighbors(pmol, rng, max_neighbors=4)
            for op, child_mol in neighbors:
                csmi = child_mol.canonical_smiles
                if csmi in archive:
                    continue
                cand = evaluate_candidate(
                    child_mol,
                    adme_model,
                    pocket,
                    generation=gen,
                    parent_smiles=parent.canonical_smiles,
                    operation=op,
                    seed=seed + gen * 31 + len(archive),
                )
                archive[csmi] = cand
                mol_cache[csmi] = child_mol
                offspring.append(cand)
                if len(offspring) >= pop_size:
                    break
            if len(offspring) >= pop_size:
                break

        combined = population + offspring
        fronts = non_dominated_sort(combined)
        next_pop: List[OptimizationCandidate] = []
        for f in fronts:
            f_sorted = sorted(f, key=lambda c: (c.crowding_distance, c.mpo_score), reverse=True)
            for c in f_sorted:
                if len(next_pop) < pop_size:
                    next_pop.append(c)
        population = next_pop
        front0 = fronts[0] if fronts else population
        history_best_mpo.append(max(c.mpo_score for c in archive.values()))
        history_hv.append(compute_pareto_hypervolume_2d(front0))

    all_cands = list(archive.values())
    final_fronts = non_dominated_sort(all_cands)
    pareto_front = sorted(final_fronts[0], key=lambda c: c.mpo_score, reverse=True) if final_fronts else []

    return OptimizationTrace(
        seed_smiles=seed_cand.canonical_smiles,
        method=method,
        total_evaluated=len(all_cands),
        pareto_front=pareto_front,
        history_best_mpo=history_best_mpo,
        history_hypervolume=history_hv,
        all_candidates=sorted(all_cands, key=lambda c: c.mpo_score, reverse=True),
    )
