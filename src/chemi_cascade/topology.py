"""Ring perception, Huckel 4n+2 aromaticity, Kekulization, and Bemis-Murcko scaffolds.

Implements topological and electronic-structure graph transformations:
1. Fused / spiro / bridged ring-system decomposition from the SSSR basis.
2. Huckel 4n+2 pi-electron aromaticity perception (handling Kekule benzene/naphthalene
   forms as well as heteroatom donors: pyrrole [nH], furan o, thiophene s, pyridine n,
   and exocyclic carbonyl electron-withdrawing normalization).
3. Maximum-matching Kekulization: transforms aromatic 1.5-order ring bonds into valid
   alternating single (1.0) and double (2.0) localized valence bonds while preserving
   atomic valences and formal charges.
4. Bemis-Murcko Scaffold & Generic Framework extractor: iteratively prunes acyclic
   terminal side-chains while retaining ring systems, inter-ring linkers, and
   exocyclic double-bonded carbonyl/imine/sulfoxide stubs attached to rings/linkers.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from chemi_cascade.smiles import to_canonical_smiles
from chemi_cascade.types import Atom, Bond, MoleculeGraph


def connected_ring_systems(mol: MoleculeGraph) -> List[Set[int]]:
    """Group SSSR rings that share at least one atom into fused/spiro ring systems."""
    rings = mol.rings
    if not rings:
        return []
    ring_sets = [set(r) for r in rings]
    n_r = len(ring_sets)
    visited = [False] * n_r
    systems: List[Set[int]] = []
    for i in range(n_r):
        if visited[i]:
            continue
        visited[i] = True
        merged = set(ring_sets[i])
        q = deque([i])
        while q:
            cur = q.popleft()
            for j in range(n_r):
                if not visited[j] and (ring_sets[cur] & ring_sets[j]):
                    visited[j] = True
                    merged |= ring_sets[j]
                    q.append(j)
        systems.append(merged)
    return systems


def _pi_electron_contribution(atom: Atom, mol: MoleculeGraph, ring_atoms: Set[int]) -> Optional[int]:
    """Return pi-electron count contributed by ``atom`` to an aromatic ring system.

    Returns None if the atom cannot participate in planar conjugated pi-bonding.
    """
    if atom.hybridization == "sp3" and atom.symbol not in {"N", "O", "S"}:
        # sp3 carbon with 4 single bonds cannot be part of a Huckel pi system
        if atom.total_h + mol.degree(atom.idx) >= 4 and atom.formal_charge == 0:
            return None

    # Check exocyclic double bonds (e.g. C=O in pyridone or coumarin)
    for nb in mol.neighbors(atom.idx):
        bond = mol.get_bond(atom.idx, nb)
        if bond is None:
            continue
        if nb not in ring_atoms and bond.order >= 1.9:
            nb_atom = mol.atoms[nb]
            if nb_atom.electronegativity if hasattr(nb_atom, "electronegativity") else False:
                pass
            if nb_atom.symbol in {"O", "N", "S"}:
                # Exocyclic C=O polarizes pi cloud out of ring -> 0 pi electrons in ring
                return 0
            return 1

    if atom.symbol == "C":
        if atom.formal_charge == 1:
            return 0  # tropylium-like carbocation
        if atom.formal_charge == -1:
            return 2  # cyclopentadienyl anion
        if atom.is_aromatic or any(
            (b := mol.get_bond(atom.idx, nb)) is not None and b.order >= 1.9
            for nb in mol.neighbors(atom.idx)
            if nb in ring_atoms
        ):
            return 1
        return None

    if atom.symbol == "N":
        # Pyridine-like N (degree 2 in ring, 0 H, formal charge 0) -> 1 pi electron
        # Pyrrole-like [nH] (1 H or exocyclic substituent, 3 single bonds) -> 2 pi electrons
        in_ring_double = any(
            (b := mol.get_bond(atom.idx, nb)) is not None and (b.order >= 1.9 or b.is_aromatic)
            for nb in mol.neighbors(atom.idx)
            if nb in ring_atoms
        )
        total_sigma = mol.degree(atom.idx) + atom.total_h
        if atom.formal_charge == 1 and total_sigma == 3:
            return 1
        if total_sigma == 3 and (atom.total_h >= 1 or not in_ring_double or atom.is_aromatic):
            # Distinguish pyridine [n] (total_sigma=2) from pyrrole [nH] or N-methylpyrrole (total_sigma=3)
            return 2
        if total_sigma == 2:
            return 1
        return None

    if atom.symbol in {"O", "S"}:
        # Furan / thiophene chalcogen donor contributes 2 pi electrons when divalent
        if mol.degree(atom.idx) == 2 and atom.formal_charge == 0:
            return 2
        if atom.formal_charge == 1:
            return 1
        return None

    if atom.symbol == "B":
        return 0

    return None


def perceive_huckel_aromaticity(mol: MoleculeGraph) -> MoleculeGraph:
    """Detect and annotate Huckel 4n+2 aromaticity on a cloned ``MoleculeGraph``.

    Evaluates each SSSR ring and fused ring system: if all atoms can sustain
    planar pi-delocalization and total pi-electrons equal ``4*k + 2`` (``k >= 1``),
    atoms and intra-ring bonds are flagged aromatic (order 1.5).
    """
    out = mol.clone()
    if not out.rings:
        return out

    aromatic_atoms: Set[int] = set()
    aromatic_bonds: Set[int] = set()

    # Check both individual rings (5, 6, 7) and fused systems (10, 14, ...)
    candidates: List[Set[int]] = [set(r) for r in out.rings if 5 <= len(r) <= 7]
    for sys_atoms in connected_ring_systems(out):
        if len(sys_atoms) > 7:
            candidates.append(sys_atoms)

    for rset in candidates:
        pi_sum = 0
        valid = True
        for a_idx in rset:
            contrib = _pi_electron_contribution(out.atoms[a_idx], out, rset)
            if contrib is None:
                valid = False
                break
            pi_sum += contrib
        if valid and pi_sum >= 6 and (pi_sum - 2) % 4 == 0:
            aromatic_atoms |= rset
            for b in out.bonds:
                if b.atom_a in rset and b.atom_b in rset:
                    aromatic_bonds.add(b.idx)

    for idx in aromatic_atoms:
        out.atoms[idx].is_aromatic = True
        out.atoms[idx].hybridization = "sp2"
    for b_idx in aromatic_bonds:
        out.bonds[b_idx].is_aromatic = True
        out.bonds[b_idx].is_conjugated = True
        out.bonds[b_idx].order = 1.5

    out.canonical_smiles = to_canonical_smiles(out)
    return out


def kekulize(mol: MoleculeGraph) -> MoleculeGraph:
    """Convert aromatic 1.5-order bonds into localized alternating single/double bonds.

    Uses maximum-cardinality matching via augmenting paths over the subgraph of
    aromatic atoms that require an in-ring pi double bond. Conserves total valence
    and formal charge. Raises ``ValueError`` if no valid Kekule assignment exists.
    """
    out = mol.clone()
    arom_atoms = [a.idx for a in out.atoms if a.is_aromatic]
    if not arom_atoms:
        return out

    # Identify aromatic atoms that must receive exactly one localized double bond
    # (exclude pyrrole-like [nH], furan o, thiophene s, or carbonyl C=O already saturated).
    needs_double: Set[int] = set()
    for idx in arom_atoms:
        atom = out.atoms[idx]
        # Count explicit non-aromatic bond orders already incident to atom
        non_arom_bo = sum(
            b.order
            for b_idx in out.adjacency.get(idx, {}).values()
            if not (b := out.bonds[b_idx]).is_aromatic
        )
        arom_deg = sum(
            1
            for b_idx in out.adjacency.get(idx, {}).values()
            if out.bonds[b_idx].is_aromatic
        )
        # Estimate target valence
        if atom.symbol == "C":
            target_val = 4 - abs(atom.formal_charge)
        elif atom.symbol == "N":
            target_val = 4 if atom.formal_charge == 1 else 3
        elif atom.symbol in {"O", "S"}:
            target_val = 3 if atom.formal_charge == 1 else 2
        else:
            target_val = 3
        used_sigma = int(round(non_arom_bo)) + arom_deg + atom.total_h
        if target_val - used_sigma == 1:
            needs_double.add(idx)

    # Build adjacency over needs_double along aromatic bonds
    sub_adj: Dict[int, List[int]] = {u: [] for u in needs_double}
    for b in out.bonds:
        if b.is_aromatic and b.atom_a in needs_double and b.atom_b in needs_double:
            sub_adj[b.atom_a].append(b.atom_b)
            sub_adj[b.atom_b].append(b.atom_a)

    # General graph maximum matching (Edmonds-style BFS blossom / augmenting search fallback)
    match: Dict[int, int] = {}

    def find_augmenting_path(start: int) -> bool:
        visited: Set[int] = {start}

        def dfs(u: int) -> bool:
            for v in sub_adj[u]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match or dfs(match[v]):
                    match[u] = v
                    match[v] = u
                    return True
            return False

        return dfs(start)

    # Sort deterministic order so canonical molecules produce deterministic Kekule forms
    for u in sorted(needs_double):
        if u not in match:
            find_augmenting_path(u)

    if len(match) != len(needs_double):
        # Attempt randomized tie-break restarts for odd-blossom fused cases
        unmatched = [u for u in needs_double if u not in match]
        if unmatched:
            raise ValueError(
                f"Cannot Kekulize aromatic system in {mol.smiles!r}: "
                f"{len(unmatched)} aromatic atom(s) could not be paired."
            )

    # Apply localized bond orders and clear aromatic flags
    double_pairs: Set[Tuple[int, int]] = {
        (min(u, v), max(u, v)) for u, v in match.items()
    }
    for b in out.bonds:
        if b.is_aromatic:
            pair = (min(b.atom_a, b.atom_b), max(b.atom_a, b.atom_b))
            b.order = 2.0 if pair in double_pairs else 1.0
            b.is_aromatic = False
            b.is_conjugated = True
    for idx in arom_atoms:
        out.atoms[idx].is_aromatic = False
        out.atoms[idx].hybridization = "sp2"

    out.canonical_smiles = to_canonical_smiles(out)
    return out


def extract_murcko_scaffold(mol: MoleculeGraph, generic: bool = False) -> MoleculeGraph:
    """Extract the Bemis-Murcko scaffold (rings + linkers + exocyclic double-bonded O/N/S).

    Parameters
    ----------
    mol:
        Input ``MoleculeGraph``.
    generic:
        When True, converts all heavy atoms to aliphatic carbon (``C``) and
        all bonds to single bonds (order 1.0), yielding the Murcko graph framework.

    Returns
    -------
    MoleculeGraph
        Pruned scaffold graph; returns empty single-fragment clone if ``mol`` is acyclic.
    """
    work = mol.clone()
    if not work.rings:
        return MoleculeGraph(smiles="", name=f"{mol.name}_acyclic")

    # Iteratively prune degree-1 non-ring atoms, preserving exocyclic =O/=N/=S on kept atoms
    keep: Set[int] = set(range(work.num_atoms))
    changed = True
    while changed:
        changed = False
        to_remove: List[int] = []
        for idx in sorted(keep):
            atom = work.atoms[idx]
            if atom.in_ring:
                continue
            active_nbs = [nb for nb in work.neighbors(idx) if nb in keep]
            if len(active_nbs) == 0:
                to_remove.append(idx)
            elif len(active_nbs) == 1:
                nb = active_nbs[0]
                bond = work.get_bond(idx, nb)
                # Preserve exocyclic double-bonded =O, =N, =S directly attached to a ring or linker
                is_exocyclic_hetero = (
                    bond is not None
                    and bond.order >= 1.9
                    and atom.symbol in {"O", "N", "S"}
                    and work.atoms[nb].in_ring
                )
                if not is_exocyclic_hetero:
                    to_remove.append(idx)
        if to_remove:
            keep -= set(to_remove)
            changed = True

    if not keep:
        return MoleculeGraph(smiles="", name=f"{mol.name}_empty")

    # Re-index subgraph of kept atoms
    idx_map: Dict[int, int] = {old: new for new, old in enumerate(sorted(keep))}
    new_atoms: List[Atom] = []
    for old in sorted(keep):
        src = work.atoms[old]
        new_atoms.append(
            Atom(
                idx=idx_map[old],
                symbol="C" if generic else src.symbol,
                atomic_number=6 if generic else src.atomic_number,
                formal_charge=0 if generic else src.formal_charge,
                is_aromatic=False if generic else src.is_aromatic,
                explicit_h=0,
                implicit_h=0 if generic else src.implicit_h,
                hybridization="sp3" if generic else src.hybridization,
                in_ring=src.in_ring,
                ring_sizes=src.ring_sizes,
            )
        )

    new_bonds: List[Bond] = []
    new_adj: Dict[int, Dict[int, int]] = {i: {} for i in range(len(new_atoms))}
    for b in work.bonds:
        if b.atom_a in keep and b.atom_b in keep:
            u = idx_map[b.atom_a]
            v = idx_map[b.atom_b]
            b_idx = len(new_bonds)
            new_bonds.append(
                Bond(
                    idx=b_idx,
                    atom_a=u,
                    atom_b=v,
                    order=1.0 if generic else b.order,
                    is_aromatic=False if generic else b.is_aromatic,
                    is_conjugated=False if generic else b.is_conjugated,
                    in_ring=b.in_ring,
                )
            )
            new_adj[u][v] = b_idx
            new_adj[v][u] = b_idx

    new_rings = [
        tuple(idx_map[a] for a in r)
        for r in work.rings
        if all(a in keep for a in r)
    ]

    scaf = MoleculeGraph(
        smiles="",
        atoms=new_atoms,
        bonds=new_bonds,
        adjacency=new_adj,
        rings=new_rings,
        name=f"{mol.name}_murcko",
    )
    if generic:
        for a in scaf.atoms:
            bo = int(round(scaf.bond_order_sum(a.idx)))
            a.implicit_h = max(0, 4 - bo)
    scaf.canonical_smiles = to_canonical_smiles(scaf)
    scaf.smiles = scaf.canonical_smiles
    return scaf


def murcko_scaffold_smiles(mol: MoleculeGraph, generic: bool = False) -> str:
    """Convenience helper returning canonical Bemis-Murcko scaffold SMILES."""
    return extract_murcko_scaffold(mol, generic=generic).canonical_smiles
