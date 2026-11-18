"""SMILES parser, Morgan-invariant canonicalizer, and SMARTS substructure matcher.

Implements a pure-Python Daylight SMILES recursive/stack parser supporting:
- Organic subset atoms (`B`, `C`, `N`, `O`, `P`, `S`, `F`, `Cl`, `Br`, `I`) and
  aromatic lowercase symbols (`b`, `c`, `n`, `o`, `p`, `s`).
- Bracketed atom specifications (`[isotope? symbol chiral? hcount? charge? class?]`).
- Explicit single (`-`), double (`=`), triple (`#`), aromatic (`:`), and directional
  (`/`, `\\`) bonds, multi-digit (`%10`) ring closures, branching, and fragments (`.`).
- Valence-consistent implicit hydrogen calculation and hybridization assignment.
- Extended Morgan invariant canonicalization and canonical DFS SMILES emission.
- SMARTS atom/bond primitive parser and VF2-style backtracking subgraph matcher.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from chemi_cascade.types import (
    ATOMIC_NUMBER_TO_SYMBOL,
    ORGANIC_SUBSET,
    Atom,
    Bond,
    MoleculeGraph,
    get_element,
)

_VALENCE_LADDER: Dict[str, Tuple[int, ...]] = {
    "B": (3,),
    "C": (4,),
    "N": (3, 5),
    "O": (2,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "F": (1,),
    "Cl": (1,),
    "Br": (1,),
    "I": (1,),
    "Si": (4,),
}

_AROMATIC_LOWER: Set[str] = {"b", "c", "n", "o", "p", "s"}

_BRACKET_RE = re.compile(
    r"^(\d+)?([A-Z][a-z]?|[bcnops])(@{1,2})?(H\d*)?([+-]+\d*|\+\d*|-\d*)?(:\d+)?$"
)


class SmilesSyntaxError(ValueError):
    """Raised when a SMILES or SMARTS string cannot be parsed cleanly."""


def _compute_implicit_h(
    symbol: str,
    bond_order_sum: float,
    explicit_h: int,
    formal_charge: int,
    is_aromatic: bool,
    bracketed: bool,
) -> int:
    """Determine implicit hydrogen count from Daylight SMILES valence rules."""
    if bracketed:
        return 0
    valences = _VALENCE_LADDER.get(symbol)
    if not valences:
        return 0
    # Aromatic atoms contribute 1.5 per aromatic bond; subtract 1 pi-electron if needed
    effective_bo = int(round(bond_order_sum + (0.5 if is_aromatic and bond_order_sum % 1.0 == 0.5 else 0.0)))
    if is_aromatic:
        effective_bo = int(round(bond_order_sum))
        # Unbracketed aromatic organic-subset atoms use their lowest standard valence
        arom_valence = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2}.get(symbol, valences[0])
        used = effective_bo + explicit_h
        adj_target = arom_valence - formal_charge if symbol != "C" else arom_valence - abs(formal_charge)
        return max(0, adj_target - used)
    used = int(round(bond_order_sum)) + explicit_h
    for target in valences:
        if used <= target:
            return max(0, target - used)
    return 0


def _parse_bracket_atom(content: str) -> Tuple[str, Optional[int], str, int, int, bool]:
    """Parse contents inside ``[...]`` into (symbol, isotope, chiral, explicit_h, charge, is_arom)."""
    m = _BRACKET_RE.match(content)
    if not m:
        raise SmilesSyntaxError(f"Invalid bracketed atom specification: [{content}]")
    iso_s, raw_sym, chiral, h_s, chg_s, _ = m.groups()
    isotope = int(iso_s) if iso_s else None
    is_aromatic = raw_sym in _AROMATIC_LOWER
    symbol = raw_sym.upper() if is_aromatic else raw_sym
    if symbol not in {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "Fe", "Zn"}:
        # Allow any known element from table
        get_element(symbol)
    explicit_h = 0
    if h_s:
        explicit_h = 1 if len(h_s) == 1 else int(h_s[1:])
    formal_charge = 0
    if chg_s:
        if chg_s.startswith("+"):
            if len(chg_s) == 1:
                formal_charge = 1
            elif chg_s[1:].isdigit():
                formal_charge = int(chg_s[1:])
            elif all(c == "+" for c in chg_s):
                formal_charge = len(chg_s)
            else:
                raise SmilesSyntaxError(f"Malformed positive charge: {chg_s}")
        elif chg_s.startswith("-"):
            if len(chg_s) == 1:
                formal_charge = -1
            elif chg_s[1:].isdigit():
                formal_charge = -int(chg_s[1:])
            elif all(c == "-" for c in chg_s):
                formal_charge = -len(chg_s)
            else:
                raise SmilesSyntaxError(f"Malformed negative charge: {chg_s}")
    return symbol, isotope, chiral or "", explicit_h, formal_charge, is_aromatic


def _add_bond(
    mol: MoleculeGraph,
    u: int,
    v: int,
    bond_sym: Optional[str],
    stereo: str = "",
) -> None:
    """Create and index a covalent bond between atoms ``u`` and ``v``."""
    if u == v:
        raise SmilesSyntaxError(f"Self-loop bond on atom index {u} is invalid.")
    if v in mol.adjacency.get(u, {}):
        raise SmilesSyntaxError(f"Duplicate bond between atoms {u} and {v}.")

    a_u = mol.atoms[u]
    a_v = mol.atoms[v]
    is_arom = False
    is_conj = False
    order = 1.0

    if bond_sym == "-":
        order = 1.0
    elif bond_sym == "=":
        order = 2.0
        is_conj = True
    elif bond_sym == "#":
        order = 3.0
        is_conj = True
    elif bond_sym == ":":
        order = 1.5
        is_arom = True
        is_conj = True
    elif bond_sym is None:
        if a_u.is_aromatic and a_v.is_aromatic:
            order = 1.5
            is_arom = True
            is_conj = True
        else:
            order = 1.0
    else:
        order = 1.0

    b_idx = len(mol.bonds)
    bond = Bond(
        idx=b_idx,
        atom_a=u,
        atom_b=v,
        order=order,
        is_aromatic=is_arom,
        is_conjugated=is_conj,
        stereo=stereo,
    )
    mol.bonds.append(bond)
    mol.adjacency.setdefault(u, {})[v] = b_idx
    mol.adjacency.setdefault(v, {})[u] = b_idx


def _find_simple_rings(mol: MoleculeGraph) -> List[Tuple[int, ...]]:
    """Find fundamental / small rings up to size 12 using BFS cycle detection."""
    n = mol.num_atoms
    if n < 3 or mol.num_bonds < 3:
        return []

    # First identify bridge bonds via Tarjan's algorithm to avoid searching trees
    disc = [-1] * n
    low = [-1] * n
    timer = [0]
    bridges: Set[Tuple[int, int]] = set()

    def dfs(u: int, parent: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in mol.neighbors(u):
            if v == parent:
                continue
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            else:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)

    # For each non-bridge bond (u, v), find shortest path from u to v excluding edge (u, v)
    seen_cycles: Set[Tuple[int, ...]] = set()
    rings: List[Tuple[int, ...]] = []
    for b in mol.bonds:
        u, v = b.atom_a, b.atom_b
        if (min(u, v), max(u, v)) in bridges:
            continue
        # Mark bond in ring immediately
        b.in_ring = True
        # BFS for shortest cycle through (u, v)
        queue: List[Tuple[int, List[int]]] = [(u, [u])]
        visited = {u}
        found: Optional[List[int]] = None
        while queue:
            curr, path = queue.pop(0)
            if len(path) > 12:
                break
            for nxt in mol.neighbors(curr):
                if curr == u and nxt == v:
                    continue
                if (min(curr, nxt), max(curr, nxt)) in bridges:
                    continue
                if nxt == v:
                    found = path + [v]
                    break
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
            if found is not None:
                break
        if found is not None:
            key = tuple(sorted(found))
            if key not in seen_cycles:
                seen_cycles.add(key)
                rings.append(tuple(found))

    # Keep linearly independent smallest cycles (SSSR upper bound M - N + C)
    rings.sort(key=len)
    # Count connected components
    comp_seen: Set[int] = set()
    n_comp = 0
    for i in range(n):
        if i not in comp_seen:
            n_comp += 1
            stack = [i]
            comp_seen.add(i)
            while stack:
                cur = stack.pop()
                for nxt in mol.neighbors(cur):
                    if nxt not in comp_seen:
                        comp_seen.add(nxt)
                        stack.append(nxt)
    circuit_rank = max(0, mol.num_bonds - n + n_comp)

    # Filter via XOR basis over bond bitmasks to yield minimal SSSR basis
    bond_map = {tuple(sorted((b.atom_a, b.atom_b))): b.idx for b in mol.bonds}
    basis: List[int] = []
    sssr: List[Tuple[int, ...]] = []

    def cycle_mask(cyc: Tuple[int, ...]) -> int:
        mask = 0
        m_len = len(cyc)
        for k in range(m_len):
            edge = tuple(sorted((cyc[k], cyc[(k + 1) % m_len])))
            mask |= 1 << bond_map[edge]
        return mask

    def independent(mask: int) -> bool:
        x = mask
        for bvec in basis:
            lsb = bvec & -bvec
            if x & lsb:
                x ^= bvec
        return x != 0

    for cyc in rings:
        if len(sssr) >= circuit_rank:
            break
        mask = cycle_mask(cyc)
        if independent(mask):
            # Insert into Gaussian elimination basis
            x = mask
            for bvec in basis:
                lsb = bvec & -bvec
                if x & lsb:
                    x ^= bvec
            if x != 0:
                basis.append(x)
                sssr.append(cyc)

    # Annotate atoms & bonds with ring membership
    atom_ring_sizes: Dict[int, List[int]] = {i: [] for i in range(n)}
    for cyc in sssr:
        r_len = len(cyc)
        for idx, a_idx in enumerate(cyc):
            mol.atoms[a_idx].in_ring = True
            atom_ring_sizes[a_idx].append(r_len)
            ring_bond = mol.get_bond(a_idx, cyc[(idx + 1) % r_len])
            if ring_bond is not None:
                ring_bond.in_ring = True
    for i in range(n):
        mol.atoms[i].ring_sizes = tuple(sorted(atom_ring_sizes[i]))

    mol.rings = sssr
    return sssr


def _annotate_hybridization_and_conjugation(mol: MoleculeGraph) -> None:
    """Assign sp / sp2 / sp3 hybridization and conjugation flags."""
    for a in mol.atoms:
        if a.is_aromatic:
            a.hybridization = "sp2"
            continue
        incident = [mol.bonds[b_idx] for b_idx in mol.adjacency.get(a.idx, {}).values()]
        has_triple = any(b.order >= 2.9 for b in incident)
        double_count = sum(1 for b in incident if 1.9 <= b.order <= 2.1)
        if has_triple or double_count >= 2:
            a.hybridization = "sp"
        elif double_count == 1:
            a.hybridization = "sp2"
        elif a.symbol in {"N", "O", "S"} and any(
            mol.atoms[nb].is_aromatic or any(
                mol.bonds[nb_b].order >= 1.9 for nb_b in mol.adjacency.get(nb, {}).values()
            )
            for nb in mol.neighbors(a.idx)
        ):
            # Resonance-conjugated lone pair adjacent to sp2/aromatic center
            a.hybridization = "sp2"
        else:
            a.hybridization = "sp3"

    for b in mol.bonds:
        if b.is_aromatic or b.order >= 1.9:
            b.is_conjugated = True
        else:
            u_sp2 = mol.atoms[b.atom_a].hybridization in {"sp2", "sp"}
            v_sp2 = mol.atoms[b.atom_b].hybridization in {"sp2", "sp"}
            b.is_conjugated = u_sp2 and v_sp2


def parse_smiles(smiles: str, name: str = "", canonicalize: bool = True) -> MoleculeGraph:
    """Parse a Daylight SMILES string into an annotated ``MoleculeGraph``.

    Parameters
    ----------
    smiles:
        Input SMILES string (non-empty).
    name:
        Optional human-readable molecule identifier.
    canonicalize:
        When True, computes Morgan-invariant canonical SMILES immediately.

    Returns
    -------
    MoleculeGraph
        Parsed molecular graph with ring perception, implicit H, hybridization,
        and canonical SMILES populated.
    """
    s = smiles.strip()
    if not s:
        raise SmilesSyntaxError("Empty SMILES string.")

    mol = MoleculeGraph(smiles=s, name=name)
    stack: List[Optional[int]] = []
    current_atom: Optional[int] = None
    pending_bond: Optional[str] = None
    pending_stereo: str = ""
    ring_opens: Dict[int, Tuple[int, Optional[str], str]] = {}
    bracket_flags: List[bool] = []

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == " ":
            i += 1
            continue
        if ch == "(":
            stack.append(current_atom)
            i += 1
        elif ch == ")":
            if not stack:
                raise SmilesSyntaxError(f"Unmatched ')' at position {i} in {s!r}")
            current_atom = stack.pop()
            pending_bond = None
            pending_stereo = ""
            i += 1
        elif ch == ".":
            current_atom = None
            pending_bond = None
            pending_stereo = ""
            i += 1
        elif ch in "-=#:":
            if pending_bond is not None:
                raise SmilesSyntaxError(f"Consecutive bond symbols at position {i} in {s!r}")
            pending_bond = ch
            i += 1
        elif ch in "/\\":
            pending_stereo = ch
            if pending_bond is None:
                pending_bond = "-"
            i += 1
        elif ch == "%" or ch.isdigit():
            if current_atom is None:
                raise SmilesSyntaxError(f"Ring closure digit without preceding atom at {i} in {s!r}")
            if ch == "%":
                if i + 2 >= n or not s[i + 1:i + 3].isdigit():
                    raise SmilesSyntaxError(f"Malformed %nn ring closure at {i} in {s!r}")
                rnum = int(s[i + 1:i + 3])
                i += 3
            else:
                rnum = int(ch)
                i += 1
            if rnum in ring_opens:
                open_atom, open_bond, open_stereo = ring_opens.pop(rnum)
                chosen_bond = pending_bond or open_bond
                chosen_stereo = pending_stereo or open_stereo
                _add_bond(mol, open_atom, current_atom, chosen_bond, chosen_stereo)
            else:
                ring_opens[rnum] = (current_atom, pending_bond, pending_stereo)
            pending_bond = None
            pending_stereo = ""
        elif ch == "[":
            close_idx = s.find("]", i + 1)
            if close_idx == -1:
                raise SmilesSyntaxError(f"Unclosed '[' at position {i} in {s!r}")
            content = s[i + 1:close_idx]
            symbol, isotope, chiral, explicit_h, formal_charge, is_arom = _parse_bracket_atom(content)
            elem = get_element(symbol)
            a_idx = len(mol.atoms)
            mol.atoms.append(
                Atom(
                    idx=a_idx,
                    symbol=symbol,
                    atomic_number=elem.atomic_number,
                    formal_charge=formal_charge,
                    isotope=isotope,
                    is_aromatic=is_arom,
                    explicit_h=explicit_h,
                    chiral_tag=chiral,
                )
            )
            mol.adjacency[a_idx] = {}
            bracket_flags.append(True)
            if current_atom is not None:
                _add_bond(mol, current_atom, a_idx, pending_bond, pending_stereo)
            current_atom = a_idx
            pending_bond = None
            pending_stereo = ""
            i = close_idx + 1
        else:
            # Organic subset atom (check 2-letter Cl, Br first)
            two = s[i:i + 2]
            if two in {"Cl", "Br"}:
                symbol = two
                is_arom = False
                i += 2
            elif ch in _AROMATIC_LOWER:
                symbol = ch.upper()
                is_arom = True
                i += 1
            elif ch in ORGANIC_SUBSET:
                symbol = ch
                is_arom = False
                i += 1
            else:
                raise SmilesSyntaxError(f"Unexpected token {ch!r} at position {i} in {s!r}")

            elem = get_element(symbol)
            a_idx = len(mol.atoms)
            mol.atoms.append(
                Atom(
                    idx=a_idx,
                    symbol=symbol,
                    atomic_number=elem.atomic_number,
                    is_aromatic=is_arom,
                )
            )
            mol.adjacency[a_idx] = {}
            bracket_flags.append(False)
            if current_atom is not None:
                _add_bond(mol, current_atom, a_idx, pending_bond, pending_stereo)
            current_atom = a_idx
            pending_bond = None
            pending_stereo = ""

    if stack:
        raise SmilesSyntaxError(f"Unclosed '(' in SMILES: {s!r}")
    if ring_opens:
        raise SmilesSyntaxError(f"Unclosed ring closure(s) {sorted(ring_opens)} in SMILES: {s!r}")
    if mol.num_atoms == 0:
        raise SmilesSyntaxError(f"No atoms parsed from SMILES: {s!r}")

    # Compute implicit hydrogens
    for idx, atom in enumerate(mol.atoms):
        bo_sum = mol.bond_order_sum(idx)
        atom.implicit_h = _compute_implicit_h(
            symbol=atom.symbol,
            bond_order_sum=bo_sum,
            explicit_h=atom.explicit_h,
            formal_charge=atom.formal_charge,
            is_aromatic=atom.is_aromatic,
            bracketed=bracket_flags[idx],
        )

    _find_simple_rings(mol)
    _annotate_hybridization_and_conjugation(mol)
    if canonicalize:
        mol.canonical_smiles = to_canonical_smiles(mol)
    return mol


def _morgan_atom_ranks(mol: MoleculeGraph, iterations: int = 8) -> List[int]:
    """Compute canonical integer atom ranks via Morgan invariant refinement."""
    n = mol.num_atoms
    if n == 0:
        return []

    def initial_invariant(a: Atom) -> int:
        deg = mol.degree(a.idx)
        bo = int(round(mol.bond_order_sum(a.idx) * 2))
        tup = (
            a.atomic_number,
            deg,
            bo,
            a.formal_charge + 10,
            a.total_h,
            1 if a.is_aromatic else 0,
            1 if a.in_ring else 0,
            a.isotope or 0,
        )
        digest = hashlib.blake2b(repr(tup).encode("ascii"), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    invariants = [initial_invariant(a) for a in mol.atoms]
    for _ in range(iterations):
        next_inv: List[int] = []
        for idx in range(n):
            nb_tokens = []
            for nb in mol.neighbors(idx):
                bond = mol.get_bond(idx, nb)
                bo_code = int(round((bond.order if bond else 1.0) * 2))
                nb_tokens.append((invariants[nb], bo_code))
            nb_tokens.sort()
            payload = (invariants[idx], tuple(nb_tokens))
            digest = hashlib.blake2b(repr(payload).encode("ascii"), digest_size=8).digest()
            next_inv.append(int.from_bytes(digest, "big"))
        if len(set(next_inv)) == len(set(invariants)) and len(set(next_inv)) == n:
            invariants = next_inv
            break
        invariants = next_inv

    # Convert arbitrary 64-bit hashes into dense deterministic tie-broken ranks 0..n-1
    order = sorted(range(n), key=lambda i: (invariants[i], mol.atoms[i].atomic_number, mol.degree(i)))
    ranks = [0] * n
    for r, atom_idx in enumerate(order):
        ranks[atom_idx] = r
    return ranks


def _format_atom_smiles(atom: Atom, mol: MoleculeGraph) -> str:
    """Format an atom as canonical SMILES token (organic subset or bracketed)."""
    needs_bracket = (
        atom.formal_charge != 0
        or atom.isotope is not None
        or atom.symbol not in ORGANIC_SUBSET
        or (atom.is_aromatic and atom.symbol == "N" and atom.total_h > 0)
        or (atom.is_aromatic and atom.symbol in {"O", "S"} and atom.total_h > 0)
    )
    sym = atom.symbol.lower() if atom.is_aromatic else atom.symbol
    if not needs_bracket:
        return sym
    parts = ["["]
    if atom.isotope is not None:
        parts.append(str(atom.isotope))
    parts.append(sym)
    if atom.total_h == 1:
        parts.append("H")
    elif atom.total_h > 1:
        parts.append(f"H{atom.total_h}")
    if atom.formal_charge > 0:
        parts.append("+" if atom.formal_charge == 1 else f"+{atom.formal_charge}")
    elif atom.formal_charge < 0:
        parts.append("-" if atom.formal_charge == -1 else f"-{abs(atom.formal_charge)}")
    parts.append("]")
    return "".join(parts)


def _format_bond_smiles(bond: Bond) -> str:
    """Format bond order symbol for SMILES emission (omitting implicit single/aromatic)."""
    if bond.is_aromatic:
        return ""
    if bond.order >= 2.9:
        return "#"
    if bond.order >= 1.9:
        return "="
    return ""


def _ring_token(num: int) -> str:
    """Format ring closure index using Daylight %nn rule for >= 10."""
    return str(num) if num < 10 else f"%{num}"


def to_canonical_smiles(mol: MoleculeGraph) -> str:
    """Emit a deterministic canonical SMILES string for ``mol``.

    Handles disconnected fragments, ring closures, and branching in canonical
    Morgan-rank order so graph-isomorphic inputs map to identical strings.
    """
    n = mol.num_atoms
    if n == 0:
        return ""

    ranks = _morgan_atom_ranks(mol)
    visited = [False] * n
    # Find connected components sorted by minimum canonical rank
    components: List[List[int]] = []
    comp_visited = [False] * n
    for i in sorted(range(n), key=lambda idx: ranks[idx]):
        if not comp_visited[i]:
            comp: List[int] = []
            stk = [i]
            comp_visited[i] = True
            while stk:
                cur = stk.pop()
                comp.append(cur)
                for nb in mol.neighbors(cur):
                    if not comp_visited[nb]:
                        comp_visited[nb] = True
                        stk.append(nb)
            comp.sort(key=lambda idx: ranks[idx])
            components.append(comp)

    frag_strings: List[str] = []
    for comp in components:
        root = comp[0]
        # Build spanning DFS tree and identify back-edges (ring closures)
        dfs_order: List[int] = []
        parent_of: Dict[int, Optional[int]] = {root: None}
        tree_children: Dict[int, List[int]] = {u: [] for u in comp}
        back_edges: Dict[int, List[Tuple[int, int]]] = {u: [] for u in comp}
        seen_back: Set[Tuple[int, int]] = set()
        ring_counter = 1
        edge_to_ring: Dict[Tuple[int, int], int] = {}

        def build_tree(u: int) -> None:
            nonlocal ring_counter
            visited[u] = True
            dfs_order.append(u)
            nbs = sorted(mol.neighbors(u), key=lambda x: ranks[x])
            for v in nbs:
                if v == parent_of.get(u):
                    continue
                if not visited[v]:
                    parent_of[v] = u
                    tree_children[u].append(v)
                    build_tree(v)
                else:
                    ekey = (min(u, v), max(u, v))
                    if ekey not in seen_back:
                        seen_back.add(ekey)
                        rnum = ring_counter
                        ring_counter += 1
                        edge_to_ring[ekey] = rnum
                        back_edges[v].append((u, rnum))
                        back_edges[u].append((v, rnum))

        build_tree(root)

        def emit(u: int) -> str:
            out = [_format_atom_smiles(mol.atoms[u], mol)]
            # Ring closures attached to u, ordered deterministically
            closures = sorted(
                back_edges[u],
                key=lambda item: item[1],
            )
            for other, rnum in closures:
                bond = mol.get_bond(u, other)
                bsym = _format_bond_smiles(bond) if bond is not None else ""
                # Only emit bond symbol on the opening side or both consistently
                if ranks[u] < ranks[other]:
                    out.append(bsym)
                out.append(_ring_token(rnum))

            children = tree_children[u]
            for idx, v in enumerate(children):
                bond = mol.get_bond(u, v)
                bsym = _format_bond_smiles(bond) if bond is not None else ""
                sub = bsym + emit(v)
                if idx < len(children) - 1:
                    out.append(f"({sub})")
                else:
                    out.append(sub)
            return "".join(out)

        frag_strings.append(emit(root))

    frag_strings.sort()
    return ".".join(frag_strings)


def molecular_formula(mol: MoleculeGraph) -> str:
    """Compute Hill-system molecular formula (C first, H second, alphabetical)."""
    counts: Dict[str, int] = {}
    h_total = 0
    for a in mol.atoms:
        if a.symbol == "H":
            h_total += 1 + a.total_h
        else:
            counts[a.symbol] = counts.get(a.symbol, 0) + 1
            h_total += a.total_h
    if h_total > 0:
        counts["H"] = h_total

    ordered: List[str] = []
    if "C" in counts:
        ordered.append("C")
        if "H" in counts:
            ordered.append("H")
    for sym in sorted(k for k in counts if k not in {"C", "H"}):
        ordered.append(sym)
    parts: List[str] = []
    for sym in ordered:
        cnt = counts[sym]
        parts.append(sym if cnt == 1 else f"{sym}{cnt}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# SMARTS Pattern Matching Engine
# ---------------------------------------------------------------------------

_SMARTS_TOKEN_RE = re.compile(
    r"!?("
    r"#\d+|"
    r"R\d*|r\d+|D\d+|X\d+|H\d*|"
    r"[+-]\d*|"
    r"Cl|Br|Si|Fe|Zn|"
    r"[BCNOPSFIbcnopsaA*]"
    r")"
)


@dataclass
class _SmartsAtomExpr:
    """Parsed SMARTS atom expression supporting AND/OR/NOT over primitives."""

    raw: str

    def matches(self, atom: Atom, mol: MoleculeGraph) -> bool:
        """Evaluate whether ``atom`` satisfies this SMARTS atom expression."""
        or_clauses = self.raw.split(",")
        return any(self._eval_and(clause, atom, mol) for clause in or_clauses)

    def _eval_and(self, clause: str, atom: Atom, mol: MoleculeGraph) -> bool:
        parts = re.split(r"[;&]", clause)
        for part in parts:
            p = part.strip()
            if not p:
                continue
            prims = _SMARTS_TOKEN_RE.findall(p)
            if not prims:
                if not self._eval_prim(p, atom, mol):
                    return False
            else:
                # Re-scan keeping '!' prefix attached to each primitive
                idx = 0
                while idx < len(p):
                    m = _SMARTS_TOKEN_RE.match(p, idx)
                    if m is None:
                        return False
                    if not self._eval_prim(m.group(0), atom, mol):
                        return False
                    idx = m.end()
        return True

    def _eval_prim(self, tok: str, atom: Atom, mol: MoleculeGraph) -> bool:
        negated = False
        if tok.startswith("!"):
            negated = True
            tok = tok[1:]
        res = self._match_single(tok, atom, mol)
        return (not res) if negated else res

    def _match_single(self, tok: str, atom: Atom, mol: MoleculeGraph) -> bool:
        if tok == "*":
            return True
        if tok == "a":
            return atom.is_aromatic
        if tok == "A":
            return not atom.is_aromatic
        if tok == "R":
            return atom.in_ring
        if tok.startswith("R") and tok[1:].isdigit():
            cnt = int(tok[1:])
            return (len(atom.ring_sizes) == cnt) if cnt > 0 else (not atom.in_ring)
        if tok.startswith("r") and tok[1:].isdigit():
            sz = int(tok[1:])
            return sz in atom.ring_sizes
        if tok.startswith("#") and tok[1:].isdigit():
            return atom.atomic_number == int(tok[1:])
        if tok.startswith("D") and tok[1:].isdigit():
            return mol.degree(atom.idx) == int(tok[1:])
        if tok.startswith("X") and tok[1:].isdigit():
            return (mol.degree(atom.idx) + atom.total_h) == int(tok[1:])
        if tok.startswith("H") and (len(tok) == 1 or tok[1:].isdigit()):
            want = 1 if len(tok) == 1 else int(tok[1:])
            return atom.total_h == want
        if tok.startswith("+"):
            want = 1 if len(tok) == 1 else int(tok[1:])
            return atom.formal_charge == want
        if tok.startswith("-"):
            want = -1 if len(tok) == 1 else -int(tok[1:])
            return atom.formal_charge == want
        if tok in _AROMATIC_LOWER:
            return atom.is_aromatic and atom.symbol == tok.upper()
        if tok in {"C", "N", "O", "S"}:
            return atom.symbol == tok and not atom.is_aromatic
        if tok in ATOMIC_NUMBER_TO_SYMBOL.values():
            return atom.symbol == tok
        return atom.symbol == tok.upper()


@dataclass
class _SmartsPattern:
    atoms: List[_SmartsAtomExpr]
    edges: List[Tuple[int, int, str]]
    adj: Dict[int, List[Tuple[int, str]]]


def _parse_smarts(smarts: str) -> _SmartsPattern:
    """Parse a SMARTS query string into query atoms and bond predicates."""
    s = smarts.strip()
    if not s:
        raise SmilesSyntaxError("Empty SMARTS pattern.")

    atoms: List[_SmartsAtomExpr] = []
    edges: List[Tuple[int, int, str]] = []
    adj: Dict[int, List[Tuple[int, str]]] = {}
    stack: List[Optional[int]] = []
    curr: Optional[int] = None
    pbond: str = ""
    rings: Dict[int, Tuple[int, str]] = {}

    def link(u: int, v: int, bspec: str) -> None:
        edges.append((u, v, bspec))
        adj.setdefault(u, []).append((v, bspec))
        adj.setdefault(v, []).append((u, bspec))

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            stack.append(curr)
            i += 1
        elif ch == ")":
            curr = stack.pop()
            pbond = ""
            i += 1
        elif ch in "-=#:~@":
            pbond = ch
            i += 1
        elif ch.isdigit():
            if curr is None:
                raise SmilesSyntaxError(f"Ring digit without atom in SMARTS {s!r}")
            rnum = int(ch)
            if rnum in rings:
                other, ob = rings.pop(rnum)
                link(other, curr, pbond or ob or "~")
            else:
                rings[rnum] = (curr, pbond)
            pbond = ""
            i += 1
        elif ch == "[":
            end = s.find("]", i + 1)
            if end == -1:
                raise SmilesSyntaxError(f"Unclosed '[' in SMARTS {s!r}")
            expr = s[i + 1:end]
            idx = len(atoms)
            atoms.append(_SmartsAtomExpr(expr))
            adj[idx] = []
            if curr is not None:
                link(curr, idx, pbond or "-")
            curr = idx
            pbond = ""
            i = end + 1
        else:
            two = s[i:i + 2]
            if two in {"Cl", "Br"}:
                expr = two
                i += 2
            else:
                expr = ch
                i += 1
            idx = len(atoms)
            atoms.append(_SmartsAtomExpr(expr))
            adj[idx] = []
            if curr is not None:
                link(curr, idx, pbond or "~")
            curr = idx
            pbond = ""

    return _SmartsPattern(atoms=atoms, edges=edges, adj=adj)


def _bond_matches_smarts(bond: Bond, spec: str) -> bool:
    if spec in ("~", ""):
        return True
    if spec == ":":
        return bond.is_aromatic
    if spec == "-":
        return (not bond.is_aromatic) and abs(bond.order - 1.0) < 0.2
    if spec == "=":
        return (not bond.is_aromatic) and abs(bond.order - 2.0) < 0.2
    if spec == "#":
        return abs(bond.order - 3.0) < 0.2
    if spec == "@":
        return bond.in_ring
    return True


def substructure_matches(mol: MoleculeGraph, smarts: str, max_matches: int = 32) -> List[Tuple[int, ...]]:
    """Find atom-index tuples in ``mol`` matching a SMARTS query pattern."""
    pat = _parse_smarts(smarts)
    q_len = len(pat.atoms)
    if q_len == 0 or q_len > mol.num_atoms:
        return []

    candidates: List[List[int]] = [
        [a.idx for a in mol.atoms if q_expr.matches(a, mol)]
        for q_expr in pat.atoms
    ]
    if any(not c for c in candidates):
        return []

    # Order query vertices by fewest candidates first (MRV heuristic)
    order = sorted(range(q_len), key=lambda qi: len(candidates[qi]))
    pos_of = {qi: pos for pos, qi in enumerate(order)}

    mapping: Dict[int, int] = {}
    used_target: Set[int] = set()
    results: List[Tuple[int, ...]] = []
    seen_sets: Set[FrozenSetKey] = set()  # type: ignore[name-defined]

    def backtrack(step: int) -> None:
        if len(results) >= max_matches:
            return
        if step == q_len:
            key = frozenset(mapping.values())
            if key not in seen_sets:
                seen_sets.add(key)
                results.append(tuple(mapping[qi] for qi in range(q_len)))
            return
        qi = order[step]
        for ti in candidates[qi]:
            if ti in used_target:
                continue
            ok = True
            for q_nb, bspec in pat.adj.get(qi, []):
                if pos_of[q_nb] < step:
                    t_nb = mapping[q_nb]
                    bond = mol.get_bond(ti, t_nb)
                    if bond is None or not _bond_matches_smarts(bond, bspec):
                        ok = False
                        break
            if not ok:
                continue
            mapping[qi] = ti
            used_target.add(ti)
            backtrack(step + 1)
            used_target.remove(ti)
            del mapping[qi]

    backtrack(0)
    return results


FrozenSetKey = frozenset[int]


def has_substructure(mol: MoleculeGraph, smarts: str) -> bool:
    """Return True if ``mol`` contains at least one match to ``smarts``."""
    return len(substructure_matches(mol, smarts, max_matches=1)) > 0
