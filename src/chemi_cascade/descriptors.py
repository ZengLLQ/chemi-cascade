"""Quantum-inspired partial charges, topological indices, and drug-likeness scoring.

Implements self-contained physicochemical and graph-theoretic algorithms:
- Gasteiger-Marsili PEOE (Partial Equalization of Orbital Electronegativity) solver
  with exact formal-charge conservation across iterative damped charge transfer.
- Graph topological invariants: Wiener index, Zagreb M1/M2, Randic connectivity chi,
  Balaban centric J index, and Bertz topological complexity C_T.
- Wildman-Crippen atom-contribution cLogP and Molar Refractivity (MR).
- Ertl Topological Polar Surface Area (TPSA) over N, O, S, P polar fragments.
- Bickerton Quantitative Estimate of Drug-likeness (QED) via 8 ADS functions,
  Lipinski Rule of 5, Veber oral bioavailability, Pfizer CNS-MPO, synthetic
  accessibility (SA-score heuristic), and SMARTS-based PAINS/Brenk alert filters.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from chemi_cascade.smiles import has_substructure, molecular_formula, to_canonical_smiles
from chemi_cascade.types import (
    GASTEIGER_HYBRID_PARAMS,
    Atom,
    DescriptorProfile,
    MoleculeGraph,
    get_element,
)

# Bickerton et al. (2012) QED asymmetric double-sigmoidal parameters:
# (w, a, b, c, d, e, f, dx_max)
_QED_ADS_PARAMS: Dict[str, Tuple[float, float, float, float, float, float, float]] = {
    "MW": (0.66, 2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707),
    "ALOGP": (0.46, 3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 57.62955913),
    "HBD": (0.61, 1.82504065, 2.404606024, 1.15836775, 1.076253827, 0.236766151, 1.464821602),
    "HBA": (0.05, 2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958),
    "PSA": (0.06, 1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732),
    "ROTB": (0.65, 0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063571),
    "AROM": (0.48, 3.217788970, 957.7374108, 2.274627939, 0.000000000, 1.317690384, 0.375760881),
    "ALERTS": (0.95, 0.010000000, 1199.240710, -0.090028830, 0.000000000, 0.185904477, 0.875193782),
}

# Curated medicinal-chemistry PAINS & Brenk SMARTS alerts
STRUCTURAL_ALERTS: Tuple[Tuple[str, str], ...] = (
    ("nitro_group", "[N+](=O)[O-]"),
    ("acyl_chloride", "C(=O)Cl"),
    ("michael_acceptor_enal", "C=CC=O"),
    ("azo_dye", "N=N"),
    ("thiocarbonyl", "C=S"),
    ("peroxide", "OO"),
    ("epoxide", "C1OC1"),
    ("aziridine", "C1NC1"),
    ("hydrazine", "NN"),
    ("quinone_like", "O=C1C=CC(=O)C=C1"),
    ("rhodanine_ring", "S=C1NC(=O)CS1"),
    ("polyhalogenated_methyl", "C(Cl)(Cl)Cl"),
)


def compute_gasteiger_charges(
    mol: MoleculeGraph,
    iterations: int = 8,
    damping: float = 0.5,
) -> List[float]:
    """Compute Gasteiger-Marsili PEOE partial charges on heavy atoms (and implicit H).

    Uses orbital electronegativity ``chi(q) = a + b*q + c*q^2`` with geometric
    damping ``damping^k`` per iteration ``k``. Total molecular formal charge is
    strictly conserved to floating-point precision: ``sum(q_heavy) + sum(q_H) == Q_tot``.
    """
    n = mol.num_atoms
    if n == 0:
        return []

    # Include implicit hydrogens in the charge equilibration network, then fold
    # or return heavy-atom + net H-adjusted charges? Standard cheminformatics stores
    # heavy-atom partial charges including implicit-H polarization OR heavy+explicit.
    # To guarantee sum(heavy_charges) == sum(formal_charges) as an exact invariant
    # on heavy-atom graphs while still capturing H electronegativity, we equilibrate
    # heavy-atom network + implicit H nodes and sum implicit-H charges back onto
    # their parent heavy atoms (united-atom Gasteiger representation) OR return exact
    # heavy-atom charges after neutralizing H transfer onto parent!
    # Returning united-atom charges (q_heavy_net = q_heavy + sum_q_attached_H)
    # guarantees sum(returned_charges) == sum(formal_charge) ALWAYS!
    total_nodes = n + sum(a.total_h for a in mol.atoms)
    a_coef = np.zeros(total_nodes, dtype=np.float64)
    b_coef = np.zeros(total_nodes, dtype=np.float64)
    c_coef = np.zeros(total_nodes, dtype=np.float64)
    q = np.zeros(total_nodes, dtype=np.float64)
    chi_plus = np.zeros(total_nodes, dtype=np.float64)

    edges: List[Tuple[int, int]] = []
    parent_of_h: Dict[int, int] = {}

    for i, atom in enumerate(mol.atoms):
        key = (atom.symbol, atom.hybridization)
        if key in GASTEIGER_HYBRID_PARAMS:
            pa, pb, pc = GASTEIGER_HYBRID_PARAMS[key]
        else:
            ed = get_element(atom.symbol)
            pa, pb, pc = ed.gasteiger_a, ed.gasteiger_b, ed.gasteiger_c
        a_coef[i] = pa
        b_coef[i] = pb
        c_coef[i] = pc
        q[i] = float(atom.formal_charge)
        chi_plus[i] = max(1e-6, pa + pb + pc)

    for b in mol.bonds:
        edges.append((b.atom_a, b.atom_b))

    h_idx = n
    ha, hb, hc = GASTEIGER_HYBRID_PARAMS[("H", "s")]
    h_chi_plus = ha + hb + hc
    for i, atom in enumerate(mol.atoms):
        for _ in range(atom.total_h):
            a_coef[h_idx] = ha
            b_coef[h_idx] = hb
            c_coef[h_idx] = hc
            q[h_idx] = 0.0
            chi_plus[h_idx] = h_chi_plus
            edges.append((i, h_idx))
            parent_of_h[h_idx] = i
            h_idx += 1

    # Iterative PEOE charge equilibration
    damp = damping
    for _ in range(iterations):
        chi = a_coef + b_coef * q + c_coef * (q * q)
        dq = np.zeros(total_nodes, dtype=np.float64)
        for u, v in edges:
            diff = chi[v] - chi[u]
            denom = chi_plus[u] if diff > 0.0 else chi_plus[v]
            transfer = damp * diff / max(denom, 1e-6)
            dq[u] += transfer
            dq[v] -= transfer
        q += dq
        damp *= damping

    # Collapse implicit H partial charges back onto their parent heavy atoms
    # so united-atom heavy-atom charges strictly conserve total formal charge
    united_q = q[:n].copy()
    for h_node, parent in parent_of_h.items():
        united_q[parent] += q[h_node]

    for i, val in enumerate(united_q):
        mol.atoms[i].partial_charge = float(val)
    return [float(x) for x in united_q]


def all_pairs_shortest_paths(mol: MoleculeGraph) -> NDArray[np.int32]:
    """Compute exact shortest-path topological distance matrix via BFS from each atom."""
    n = mol.num_atoms
    dist = np.full((n, n), -1, dtype=np.int32)
    for src in range(n):
        dist[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = int(dist[src, u])
            for v in mol.neighbors(u):
                if dist[src, v] == -1:
                    dist[src, v] = du + 1
                    q.append(v)
    return dist


def compute_topological_indices(mol: MoleculeGraph) -> Dict[str, float]:
    """Compute Wiener, Zagreb M1/M2, Randic chi, Balaban J, and Bertz complexity."""
    n = mol.num_atoms
    if n <= 1:
        return {
            "wiener": 0.0,
            "zagreb_m1": 0.0,
            "zagreb_m2": 0.0,
            "randic_chi": 0.0,
            "balaban_j": 0.0,
            "bertz_ct": 0.0,
        }

    dist = all_pairs_shortest_paths(mol)
    upper = dist[np.triu_indices(n, k=1)]
    wiener = int(np.sum(upper[upper > 0]))

    zagreb_m1 = sum(mol.degree(i) ** 2 for i in range(n))
    zagreb_m2 = sum(mol.degree(b.atom_a) * mol.degree(b.atom_b) for b in mol.bonds)

    randic_chi = 0.0
    for b in mol.bonds:
        du = max(1, mol.degree(b.atom_a))
        dv = max(1, mol.degree(b.atom_b))
        randic_chi += 1.0 / math.sqrt(du * dv)

    # Balaban centric J index
    s_dist = np.sum(np.maximum(dist, 0), axis=1, dtype=np.float64)
    m_bonds = max(1, mol.num_bonds)
    # Count components for cyclomatic mu = M - N + C
    seen = set()
    n_comp = 0
    for i in range(n):
        if i not in seen:
            n_comp += 1
            stk = [i]
            seen.add(i)
            while stk:
                cur = stk.pop()
                for nb in mol.neighbors(cur):
                    if nb not in seen:
                        seen.add(nb)
                        stk.append(nb)
    mu = max(0, m_bonds - n + n_comp)
    sum_bal = 0.0
    for b in mol.bonds:
        su = float(s_dist[b.atom_a])
        sv = float(s_dist[b.atom_b])
        if su > 0 and sv > 0:
            sum_bal += 1.0 / math.sqrt(su * sv)
    balaban_j = (m_bonds / (mu + 1.0)) * sum_bal

    # Bertz complexity C_T (graph bond-pair connection diversity + heteroatom entropy)
    # Using adjacency pair classes (atom_sym_u, bo, atom_sym_v, deg_u, deg_v)
    pair_classes: Dict[Tuple[object, ...], int] = {}
    for b in mol.bonds:
        au, av = mol.atoms[b.atom_a], mol.atoms[b.atom_b]
        left = (au.symbol, mol.degree(au.idx), au.is_aromatic)
        right = (av.symbol, mol.degree(av.idx), av.is_aromatic)
        if left > right:
            left, right = right, left
        key = (left, round(b.order, 1), right)
        pair_classes[key] = pair_classes.get(key, 0) + 1
    total_e = max(1, mol.num_bonds)
    entropy = 0.0
    for cnt in pair_classes.values():
        p = cnt / total_e
        entropy -= p * math.log2(p)
    bertz_ct = (2.0 * total_e * math.log2(max(2, total_e))) + total_e * entropy

    return {
        "wiener": float(wiener),
        "zagreb_m1": float(zagreb_m1),
        "zagreb_m2": float(zagreb_m2),
        "randic_chi": float(randic_chi),
        "balaban_j": float(balaban_j),
        "bertz_ct": float(bertz_ct),
    }


def _crippen_atom_contribution(atom: Atom, mol: MoleculeGraph) -> Tuple[float, float]:
    """Wildman-Crippen (1999) atom-typed (logP, MR) contribution heuristic."""
    sym = atom.symbol
    deg = mol.degree(atom.idx)
    nbs = [mol.atoms[nb] for nb in mol.neighbors(atom.idx)]
    has_hetero_nb = any(nb.symbol in {"N", "O", "S", "P", "F", "Cl", "Br", "I"} for nb in nbs)
    double_o = any(
        nb.symbol == "O" and (b := mol.get_bond(atom.idx, nb.idx)) is not None and b.order >= 1.9
        for nb in nbs
    )
    h_cnt = atom.total_h

    if sym == "C":
        if atom.is_aromatic:
            if has_hetero_nb:
                return (0.05 + 0.04 * h_cnt, 3.25)
            return (0.158 + 0.07 * h_cnt, 3.40)
        if atom.hybridization == "sp":
            return (0.33, 3.60)
        if atom.hybridization == "sp2":
            if double_o:
                return (-0.28, 3.15)
            return (0.14 + 0.05 * h_cnt, 3.35)
        # sp3 carbon
        base = 0.1441 if h_cnt >= 2 else (0.12 if h_cnt == 1 else 0.08)
        if has_hetero_nb:
            base -= 0.18
        return (base + 0.12 * h_cnt, 2.50 + 0.85 * h_cnt)

    if sym == "N":
        if atom.formal_charge > 0:
            return (-1.02, 2.20)
        if atom.is_aromatic:
            return (-0.32 if h_cnt == 0 else -0.42, 2.30)
        if double_o:
            return (-0.45, 2.50)
        if deg == 1 and any(
            (b := mol.get_bond(atom.idx, nb.idx)) is not None and b.order >= 2.9 for nb in nbs
        ):
            return (-0.56, 2.90)  # nitrile
        return (-0.70 - 0.15 * h_cnt, 2.26 + 0.45 * h_cnt)

    if sym == "O":
        if atom.formal_charge < 0:
            return (-1.20, 1.40)
        if h_cnt >= 1:
            return (-0.289 if any(nb.is_aromatic for nb in nbs) else -0.48, 1.55)
        if any(
            (b := mol.get_bond(atom.idx, nb.idx)) is not None and b.order >= 1.9 for nb in nbs
        ):
            return (-0.31, 1.15)  # carbonyl =O
        return (-0.15, 1.35)  # ether / ester -O-

    if sym == "S":
        if double_o:
            return (-0.55, 4.10)  # sulfone/sulfoxide S
        if atom.is_aromatic:
            return (0.62, 7.60)
        return (0.42 - 0.10 * h_cnt, 7.35)

    if sym == "P":
        return (0.25, 6.20)
    if sym == "F":
        return (0.41, 0.92)
    if sym == "Cl":
        return (0.69, 5.85)
    if sym == "Br":
        return (0.85, 8.85)
    if sym == "I":
        return (1.05, 13.95)
    if sym == "B":
        return (-0.10, 3.00)
    return (0.0, 2.0)


def compute_tpsa(mol: MoleculeGraph) -> float:
    """Compute Ertl (2000) Topological Polar Surface Area (A^2) over N, O, S, P."""
    tpsa = 0.0
    for atom in mol.atoms:
        sym = atom.symbol
        if sym not in {"N", "O", "S", "P"}:
            continue
        h = atom.total_h
        deg = mol.degree(atom.idx)
        nbs = mol.neighbors(atom.idx)
        orders = [
            mol.bonds[mol.adjacency[atom.idx][nb]].order for nb in nbs
        ]
        n_double = sum(1 for o in orders if 1.9 <= o <= 2.1)
        n_triple = sum(1 for o in orders if o >= 2.9)

        if sym == "N":
            if atom.is_aromatic:
                tpsa += 15.79 if h >= 1 else 12.89
            elif n_triple >= 1:
                tpsa += 23.79
            elif n_double >= 1:
                tpsa += 23.85 if h >= 1 else 12.36
            elif deg == 1 and h >= 2:
                tpsa += 26.02
            elif deg == 2 and h >= 1:
                tpsa += 12.03
            elif atom.formal_charge > 0:
                tpsa += 4.44 if h == 0 else (16.61 if h == 2 else 27.64)
            else:
                tpsa += 3.24
        elif sym == "O":
            if atom.is_aromatic:
                tpsa += 13.14
            elif n_double >= 1:
                tpsa += 17.07
            elif h >= 1:
                tpsa += 20.23
            elif atom.formal_charge < 0:
                tpsa += 23.06
            else:
                tpsa += 9.23
        elif sym == "S":
            if atom.is_aromatic:
                tpsa += 28.24
            elif n_double >= 2:
                tpsa += 8.38
            elif n_double == 1:
                tpsa += 19.21
            else:
                tpsa += 38.80 if h >= 1 else 25.30
        elif sym == "P":
            tpsa += 9.81 if n_double >= 1 else 13.59
    return round(tpsa, 3)


def count_hbd_hba(mol: MoleculeGraph) -> Tuple[int, int]:
    """Count Lipinski Hydrogen Bond Donors (OH/NH) and Acceptors (N/O)."""
    hbd = 0
    hba = 0
    for atom in mol.atoms:
        if atom.symbol in {"N", "O"}:
            hba += 1
            if atom.total_h > 0:
                hbd += atom.total_h
    return hbd, hba


def count_rotatable_bonds(mol: MoleculeGraph) -> int:
    """Count acyclic single bonds between non-terminal, non-amide heavy atoms."""
    rot = 0
    for b in mol.bonds:
        if b.in_ring or b.is_aromatic or abs(b.order - 1.0) > 0.2:
            continue
        au, av = mol.atoms[b.atom_a], mol.atoms[b.atom_b]
        if mol.degree(au.idx) <= 1 or mol.degree(av.idx) <= 1:
            continue
        # Exclude C(=O)-N amide bonds (partial double bond character)
        if {au.symbol, av.symbol} == {"C", "N"}:
            c_atom = au if au.symbol == "C" else av
            has_carbonyl = any(
                mol.atoms[nb].symbol == "O"
                and (cb := mol.get_bond(c_atom.idx, nb)) is not None
                and cb.order >= 1.9
                for nb in mol.neighbors(c_atom.idx)
            )
            if has_carbonyl:
                continue
        rot += 1
    return rot


def _ads_desirability(x: float, params: Tuple[float, float, float, float, float, float, float]) -> float:
    """Bickerton Asymmetric Double Sigmoidal (ADS) function normalized to [0, 1]."""
    _, a, b, c, d, e, f = params
    # Protect against overflow in exp
    t1 = math.exp(-max(-50.0, min(50.0, (x - c + d / 2.0) / max(e, 1e-6))))
    t2 = math.exp(-max(-50.0, min(50.0, (x - c - d / 2.0) / max(f, 1e-6))))
    val = a + b * (1.0 / (1.0 + t1)) * (1.0 - 1.0 / (1.0 + t2))
    # Approximate d_max by sampling around c
    peak = a + b * (1.0 / (1.0 + math.exp(-d / (2.0 * max(e, 1e-6))))) * (
        1.0 - 1.0 / (1.0 + math.exp(d / (2.0 * max(f, 1e-6))))
    )
    return max(1e-6, min(1.0, val / max(peak, 1e-6)))


def compute_qed(
    mw: float,
    alogp: float,
    hbd: int,
    hba: int,
    psa: float,
    rotb: int,
    arom_rings: int,
    alerts: int,
) -> float:
    """Compute Bickerton QED weighted geometric mean over 8 physicochemical properties."""
    vals = {
        "MW": mw,
        "ALOGP": alogp,
        "HBD": float(hbd),
        "HBA": float(hba),
        "PSA": psa,
        "ROTB": float(rotb),
        "AROM": float(arom_rings),
        "ALERTS": float(alerts),
    }
    log_sum = 0.0
    w_sum = 0.0
    for key, x in vals.items():
        params = _QED_ADS_PARAMS[key]
        w = params[0]
        d_i = _ads_desirability(x, params)
        log_sum += w * math.log(d_i)
        w_sum += w
    return float(max(0.0, min(1.0, math.exp(log_sum / max(w_sum, 1e-9)))))


def compute_sa_score(mol: MoleculeGraph) -> float:
    """Estimate Ertl-Schuffenhauer-inspired Synthetic Accessibility score in [1.0, 10.0].

    1.0 = trivial commercial building block; 10.0 = highly strained/complex synthesis.
    Combines structural fragment abundance proxy with ring-bridging, spiro,
    chiral-center, and size penalties.
    """
    n = mol.num_atoms
    if n == 0:
        return 10.0

    # Heuristic fragment ease: common organic atoms & aromatic 6-rings reduce penalty
    common_atoms = sum(1 for a in mol.atoms if a.symbol in {"C", "N", "O", "F", "Cl"})
    uncommon_penalty = 0.4 * (n - common_atoms)
    size_penalty = max(0.0, (n ** 1.005 - n) + 0.045 * max(0, n - 18))

    # Ring complexity: spiro atoms (belonging to >=2 rings sharing 1 atom) & fused bridges
    spiro_or_bridge = sum(1 for a in mol.atoms if len(a.ring_sizes) >= 2)
    macrocycle_penalty = sum(0.8 for r in mol.rings if len(r) >= 8)
    small_strained = sum(0.45 for r in mol.rings if len(r) in (3, 4))
    chiral_centers = sum(
        1 for a in mol.atoms
        if a.symbol == "C" and a.hybridization == "sp3" and mol.degree(a.idx) >= 3 and a.total_h <= 1
    )
    stereo_penalty = 0.25 * chiral_centers

    raw = (
        1.6
        + size_penalty
        + uncommon_penalty
        + 0.35 * max(0, spiro_or_bridge - 2)
        + macrocycle_penalty
        + small_strained
        + stereo_penalty
    )
    return float(round(max(1.0, min(10.0, raw)), 3))


def compute_cns_mpo(
    mw: float,
    clogp: float,
    tpsa: float,
    hbd: int,
    basic_centers: int,
) -> float:
    """Compute Wager (2010) Pfizer CNS-MPO desirability score in [0.0, 6.0]."""
    def ramp(val: float, good_hi: float, bad_hi: float) -> float:
        if val <= good_hi:
            return 1.0
        if val >= bad_hi:
            return 0.0
        return (bad_hi - val) / (bad_hi - good_hi)

    t_clogp = ramp(clogp, 3.0, 5.0)
    t_logd = ramp(clogp - 0.3 * basic_centers, 2.0, 4.0)
    t_mw = ramp(mw, 360.0, 500.0)
    if 40.0 <= tpsa <= 90.0:
        t_tpsa = 1.0
    elif tpsa < 40.0:
        t_tpsa = max(0.0, (tpsa - 20.0) / 20.0)
    else:
        t_tpsa = max(0.0, (120.0 - tpsa) / 30.0)
    t_hbd = 1.0 if hbd == 0 else (0.75 if hbd == 1 else (0.5 if hbd == 2 else (0.25 if hbd == 3 else 0.0)))
    # Approximate strongest basic pKa from aliphatic amine count
    est_pka = 9.2 if basic_centers > 0 else 4.0
    t_pka = ramp(est_pka, 8.0, 10.0)
    return round(t_clogp + t_logd + t_mw + t_tpsa + t_hbd + t_pka, 3)


def screen_structural_alerts(mol: MoleculeGraph) -> Tuple[str, ...]:
    """Match curated PAINS/Brenk SMARTS alerts against ``mol``."""
    triggered: List[str] = []
    for alert_name, smarts in STRUCTURAL_ALERTS:
        if has_substructure(mol, smarts):
            triggered.append(alert_name)
    return tuple(triggered)


def compute_descriptors(mol: MoleculeGraph) -> DescriptorProfile:
    """Compute the full physicochemical, topological, and drug-likeness profile."""
    if not mol.canonical_smiles:
        mol.canonical_smiles = to_canonical_smiles(mol)

    charges = compute_gasteiger_charges(mol)
    topo = compute_topological_indices(mol)

    mw = 0.0
    exact_mass = 0.0
    c_total = 0
    c_sp3 = 0
    basic_amines = 0

    for a in mol.atoms:
        ed = get_element(a.symbol)
        mw += ed.atomic_mass + a.total_h * 1.008
        exact_mass += ed.atomic_mass + a.total_h * 1.007825
        if a.symbol == "C":
            c_total += 1
            if a.hybridization == "sp3":
                c_sp3 += 1
        if a.symbol == "N" and not a.is_aromatic and a.hybridization == "sp3":
            # Exclude amides/sulfonamides from basic amine count
            is_amide = any(
                mol.atoms[nb].symbol in {"C", "S"}
                and any(
                    mol.atoms[nb2].symbol == "O"
                    and (b2 := mol.get_bond(nb, nb2)) is not None
                    and b2.order >= 1.9
                    for nb2 in mol.neighbors(nb)
                )
                for nb in mol.neighbors(a.idx)
            )
            if not is_amide:
                basic_amines += 1

    crippen_pairs = [_crippen_atom_contribution(a, mol) for a in mol.atoms]
    clogp = sum(p[0] for p in crippen_pairs)
    mr = sum(p[1] for p in crippen_pairs)
    tpsa = compute_tpsa(mol)
    hbd, hba = count_hbd_hba(mol)
    rotb = count_rotatable_bonds(mol)

    ring_count = len(mol.rings)
    arom_ring_count = sum(
        1 for r in mol.rings if all(mol.atoms[idx].is_aromatic for idx in r)
    )
    fsp3 = (c_sp3 / c_total) if c_total > 0 else 0.0
    alerts = screen_structural_alerts(mol)
    qed = compute_qed(mw, clogp, hbd, hba, tpsa, rotb, arom_ring_count, len(alerts))
    sa = compute_sa_score(mol)

    ro5 = int(mw > 500.0) + int(clogp > 5.0) + int(hbd > 5) + int(hba > 10)
    veber = (rotb <= 10) and (tpsa <= 140.0)
    cns_mpo = compute_cns_mpo(mw, clogp, tpsa, hbd, basic_amines)

    return DescriptorProfile(
        canonical_smiles=mol.canonical_smiles,
        molecular_formula=molecular_formula(mol),
        molecular_weight=round(mw, 3),
        exact_mass=round(exact_mass, 4),
        heavy_atom_count=mol.num_atoms,
        ring_count=ring_count,
        aromatic_ring_count=arom_ring_count,
        rotatable_bonds=rotb,
        hbd=hbd,
        hba=hba,
        tpsa=round(tpsa, 3),
        clogp=round(clogp, 3),
        molar_refractivity=round(mr, 3),
        fraction_csp3=round(fsp3, 4),
        qed=round(qed, 4),
        sa_score=sa,
        lipinski_violations=ro5,
        veber_pass=veber,
        cns_mpo=cns_mpo,
        wiener_index=int(topo["wiener"]),
        zagreb_m1=int(topo["zagreb_m1"]),
        zagreb_m2=int(topo["zagreb_m2"]),
        randic_chi=round(topo["randic_chi"], 4),
        balaban_j=round(topo["balaban_j"], 4),
        bertz_complexity=round(topo["bertz_ct"], 3),
        alerts=alerts,
        gasteiger_charges=tuple(round(c, 5) for c in charges),
    )
