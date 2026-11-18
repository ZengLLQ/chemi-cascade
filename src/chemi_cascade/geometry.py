"""3D Distance-Geometry conformer embedding, UFF minimization, and Kabsch RMSD.

Generates physically realistic 3D atomic coordinates from 2D ``MoleculeGraph``
topologies using pure NumPy/SciPy:
1. Covalent bond-length and 1-3 / 1-4 distance-bounds construction.
2. Floyd-Warshall triangle inequality bound smoothing.
3. Crippen-Havel Metric Matrix Distance Geometry embedding via ``eigh``.
4. UFF-inspired analytical force field (harmonic bonds, cosine angles,
   planarity/torsion restraints, and non-bonded Lennard-Jones repulsion)
   minimized with SciPy BFGS using exact analytical gradients.
5. Kabsch SVD optimal 3D superposition and conformer ensemble generation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from chemi_cascade.descriptors import all_pairs_shortest_paths
from chemi_cascade.types import Atom, Bond, Conformer3D, MoleculeGraph, get_element


def ideal_bond_length(atom_a: Atom, atom_b: Atom, bond: Bond) -> float:
    """Estimate equilibrium bond length (Angstroms) from covalent radii and order."""
    ra = get_element(atom_a.symbol).covalent_radius
    rb = get_element(atom_b.symbol).covalent_radius
    r_single = ra + rb
    # Pauling / Schomaker-Stevenson electronegativity contraction
    en_diff = abs(get_element(atom_a.symbol).electronegativity - get_element(atom_b.symbol).electronegativity)
    r_single -= 0.05 * en_diff
    if bond.is_aromatic:
        return max(1.15, r_single - 0.13)
    if bond.order >= 2.9:
        return max(1.05, r_single - 0.30)
    if bond.order >= 1.9:
        return max(1.10, r_single - 0.19)
    return max(1.15, r_single)


def ideal_bond_angle_rad(center: Atom, mol: MoleculeGraph) -> float:
    """Return equilibrium bond angle (radians) at ``center`` atom."""
    if 3 in center.ring_sizes:
        return math.radians(60.0)
    if 4 in center.ring_sizes:
        return math.radians(90.0)
    if 5 in center.ring_sizes and center.is_aromatic:
        return math.radians(108.0)
    if center.hybridization == "sp":
        return math.radians(180.0)
    if center.hybridization == "sp2" or center.is_aromatic:
        return math.radians(120.0)
    return math.radians(109.47)


def build_distance_bounds(mol: MoleculeGraph) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct lower (L) and upper (U) pairwise distance matrices (Angstroms)."""
    n = mol.num_atoms
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))

    sp_dist = all_pairs_shortest_paths(mol)
    L = np.zeros((n, n), dtype=np.float64)
    U = np.full((n, n), 45.0, dtype=np.float64)
    np.fill_diagonal(U, 0.0)

    # Non-bonded minimum steric floor (scaled vdW sum)
    vdw = np.array([get_element(a.symbol).vdw_radius for a in mol.atoms], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if sp_dist[i, j] >= 3 or sp_dist[i, j] == -1:
                lb = 0.72 * (vdw[i] + vdw[j])
                L[i, j] = L[j, i] = lb
                if sp_dist[i, j] > 0:
                    U[i, j] = U[j, i] = float(sp_dist[i, j]) * 1.55

    # 1-2 bonded lengths
    bond_len: Dict[Tuple[int, int], float] = {}
    for b in mol.bonds:
        r0 = ideal_bond_length(mol.atoms[b.atom_a], mol.atoms[b.atom_b], b)
        bond_len[(b.atom_a, b.atom_b)] = r0
        bond_len[(b.atom_b, b.atom_a)] = r0
        L[b.atom_a, b.atom_b] = L[b.atom_b, b.atom_a] = max(0.95, r0 - 0.03)
        U[b.atom_a, b.atom_b] = U[b.atom_b, b.atom_a] = r0 + 0.03

    # 1-3 geminal angles
    for j in range(n):
        nbs = mol.neighbors(j)
        theta = ideal_bond_angle_rad(mol.atoms[j], mol)
        cos_t = math.cos(theta)
        for a_idx in range(len(nbs)):
            i = nbs[a_idx]
            r_ij = bond_len[(i, j)]
            for b_idx in range(a_idx + 1, len(nbs)):
                k = nbs[b_idx]
                r_jk = bond_len[(j, k)]
                d_ik = math.sqrt(max(0.5, r_ij * r_ij + r_jk * r_jk - 2.0 * r_ij * r_jk * cos_t))
                L[i, k] = L[k, i] = max(L[i, k], d_ik - 0.08)
                U[i, k] = U[k, i] = min(U[i, k], d_ik + 0.08)

    # 1-4 dihedral bounds (cis phi=0 vs trans phi=pi)
    for b in mol.bonds:
        j, k = b.atom_a, b.atom_b
        r_jk = bond_len[(j, k)]
        th_j = ideal_bond_angle_rad(mol.atoms[j], mol)
        th_k = ideal_bond_angle_rad(mol.atoms[k], mol)
        for i in mol.neighbors(j):
            if i == k:
                continue
            r_ij = bond_len[(i, j)]
            for l_atom in mol.neighbors(k):
                if l_atom in (j, i):
                    continue
                r_kl = bond_len[(k, l_atom)]
                # Planar projection formula for cis (0) and trans (pi)
                x_cis = r_jk - r_ij * math.cos(th_j) - r_kl * math.cos(th_k)
                y_cis = r_ij * math.sin(th_j) - r_kl * math.sin(th_k)
                d_cis = math.hypot(x_cis, y_cis)
                y_trans = r_ij * math.sin(th_j) + r_kl * math.sin(th_k)
                d_trans = math.hypot(x_cis, y_trans)
                d_lo, d_hi = min(d_cis, d_trans), max(d_cis, d_trans)
                same_ring = bool(
                    set(mol.atoms[i].ring_sizes)
                    & set(mol.atoms[l_atom].ring_sizes)
                    & set(mol.atoms[j].ring_sizes)
                )
                if same_ring and (mol.atoms[j].is_aromatic or b.order >= 1.9):
                    L[i, l_atom] = L[l_atom, i] = max(L[i, l_atom], d_lo - 0.12)
                    U[i, l_atom] = U[l_atom, i] = min(U[i, l_atom], d_lo + 0.15)
                else:
                    L[i, l_atom] = L[l_atom, i] = max(L[i, l_atom], d_lo - 0.15)
                    U[i, l_atom] = U[l_atom, i] = min(U[i, l_atom], d_hi + 0.15)

    return L, U


def triangle_smooth(L: NDArray[np.float64], U: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Floyd-Warshall triangle inequality smoothing to distance bounds."""
    n = L.shape[0]
    lo = L.copy()
    hi = U.copy()
    for k in range(n):
        for i in range(n):
            u_ik = hi[i, k]
            l_ik = lo[i, k]
            for j in range(i + 1, n):
                u_new = u_ik + hi[k, j]
                if u_new < hi[i, j]:
                    hi[i, j] = hi[j, i] = u_new
                l_new = max(l_ik - hi[k, j], lo[k, j] - u_ik)
                if l_new > lo[i, j]:
                    lo[i, j] = lo[j, i] = l_new
    # Repair tiny numerical inversions safely
    hi = np.maximum(hi, lo + 1e-4)
    np.fill_diagonal(lo, 0.0)
    np.fill_diagonal(hi, 0.0)
    return lo, hi


def metric_matrix_embed(
    L: NDArray[np.float64],
    U: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Embed 3D coordinates from smoothed distance bounds via Metric Matrix DG."""
    n = L.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 3), dtype=np.float64)

    # Sample symmetric distance matrix inside [L, U]
    frac = rng.uniform(0.15, 0.85, size=(n, n))
    frac = 0.5 * (frac + frac.T)
    D = L + frac * (U - L)
    np.fill_diagonal(D, 0.0)

    D2 = D * D
    # Double-centered Gram matrix: G = -0.5 * J * D^2 * J
    row_mean = np.mean(D2, axis=1, keepdims=True)
    col_mean = np.mean(D2, axis=0, keepdims=True)
    grand_mean = float(np.mean(D2))
    G = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    eigvals, eigvecs = np.linalg.eigh(G)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    top_vals = np.maximum(eigvals[order[:3]], 1e-4)
    top_vecs = eigvecs[:, order[:3]]
    coords = top_vecs * np.sqrt(top_vals)[None, :]
    if coords.shape[1] < 3:
        pad = np.zeros((n, 3 - coords.shape[1]), dtype=np.float64)
        coords = np.hstack([coords, pad])
    # Break exact planar degeneracy if molecule has non-aromatic 3D sp3 centers
    coords += rng.normal(0.0, 0.03, size=coords.shape)
    return coords.astype(np.float64)


class _UFFObjective:
    """Precomputed UFF-style potential energy and exact analytical gradient evaluator."""

    def __init__(self, mol: MoleculeGraph) -> None:
        self.mol = mol
        self.n = mol.num_atoms

        # 1. Bond terms: (u, v, r0, k_b)
        self.bonds: List[Tuple[int, int, float, float]] = []
        bond_len: Dict[Tuple[int, int], float] = {}
        for b in mol.bonds:
            r0 = ideal_bond_length(mol.atoms[b.atom_a], mol.atoms[b.atom_b], b)
            kb = 320.0 * (1.0 + 0.35 * (b.order - 1.0))
            self.bonds.append((b.atom_a, b.atom_b, r0, kb))
            bond_len[(b.atom_a, b.atom_b)] = r0
            bond_len[(b.atom_b, b.atom_a)] = r0

        # 2. Angle terms: (i, j, k, cos_t0, k_theta)
        self.angles: List[Tuple[int, int, int, float, float]] = []
        for j in range(self.n):
            nbs = mol.neighbors(j)
            theta0 = ideal_bond_angle_rad(mol.atoms[j], mol)
            cos_t0 = math.cos(theta0)
            for a_i in range(len(nbs)):
                for b_i in range(a_i + 1, len(nbs)):
                    self.angles.append((nbs[a_i], j, nbs[b_i], cos_t0, 65.0))

        # 3. Aromatic / sp2 planarity & improper out-of-plane terms: (center, n1, n2, n3, k_oop)
        self.planars: List[Tuple[int, int, int, int, float]] = []
        for a in mol.atoms:
            if (a.is_aromatic or a.hybridization == "sp2") and mol.degree(a.idx) == 3:
                nbs = mol.neighbors(a.idx)
                self.planars.append((a.idx, nbs[0], nbs[1], nbs[2], 35.0))

        # 4. Aromatic / double-bond torsion planarity (1-4 cos^2 torsion)
        self.torsions: List[Tuple[int, int, int, int, float, int]] = []
        seen_tor: Set[Tuple[int, int, int, int]] = set()
        for b in mol.bonds:
            j, k = b.atom_a, b.atom_b
            if not (b.is_aromatic or b.order >= 1.9):
                continue
            for i in mol.neighbors(j):
                if i == k:
                    continue
                for l_atom in mol.neighbors(k):
                    if l_atom in (j, i):
                        continue
                    key = min((i, j, k, l_atom), (l_atom, k, j, i))
                    if key not in seen_tor:
                        seen_tor.add(key)
                        # E = V * (1 - cos(2*phi)) -> minima at phi=0, pi
                        self.torsions.append((i, j, k, l_atom, 18.0, 2))

        # 5. Non-bonded pairs (>= 3 bonds apart)
        sp_dist = all_pairs_shortest_paths(mol)
        self.nonbonded: List[Tuple[int, int, float, float]] = []
        for i in range(self.n):
            ri = get_element(mol.atoms[i].symbol).vdw_radius
            for j in range(i + 1, self.n):
                if sp_dist[i, j] >= 3 or sp_dist[i, j] == -1:
                    rj = get_element(mol.atoms[j].symbol).vdw_radius
                    r_star = 0.88 * (ri + rj)
                    self.nonbonded.append((i, j, r_star, 0.12))

    def energy_and_grad(
        self,
        flat_coords: NDArray[np.float64],
    ) -> Tuple[float, NDArray[np.float64], Dict[str, float]]:
        """Evaluate total UFF energy and analytical gradient."""
        x = flat_coords.reshape((self.n, 3))
        grad = np.zeros_like(x)
        e_bond = 0.0
        e_angle = 0.0
        e_planar = 0.0
        e_vdw = 0.0

        # 1. Bonds
        for u, v, r0, kb in self.bonds:
            dv = x[u] - x[v]
            r = float(np.linalg.norm(dv))
            r_safe = max(r, 1e-6)
            diff = r - r0
            e_bond += 0.5 * kb * diff * diff
            g = (kb * diff / r_safe) * dv
            grad[u] += g
            grad[v] -= g

        # 2. Cosine Angles
        for i, j, k, cos_t0, kt in self.angles:
            v1 = x[i] - x[j]
            v2 = x[k] - x[j]
            r1 = max(float(np.linalg.norm(v1)), 1e-6)
            r2 = max(float(np.linalg.norm(v2)), 1e-6)
            u1 = v1 / r1
            u2 = v2 / r2
            cos_t = float(np.clip(np.dot(u1, u2), -0.99999, 0.99999))
            diff = cos_t - cos_t0
            e_angle += 0.5 * kt * diff * diff
            pref = kt * diff
            gi = (pref / r1) * (u2 - cos_t * u1)
            gk = (pref / r2) * (u1 - cos_t * u2)
            grad[i] += gi
            grad[k] += gk
            grad[j] -= (gi + gk)

        # 3. Planarity (center distance to plane spanned by n1, n2, n3 via scalar triple product)
        for c, n1, n2, n3, koop in self.planars:
            a = x[n1] - x[c]
            b = x[n2] - x[c]
            d = x[n3] - x[c]
            b_x_d = np.cross(b, d)
            vol = float(np.dot(a, b_x_d))
            e_planar += 0.5 * koop * vol * vol
            pref = koop * vol
            gn1 = pref * b_x_d
            gn2 = pref * np.cross(d, a)
            gn3 = pref * np.cross(a, b)
            grad[n1] += gn1
            grad[n2] += gn2
            grad[n3] += gn3
            grad[c] -= (gn1 + gn2 + gn3)

        # 4. Conjugated/aromatic torsion restraint via cos(2*phi) = 2*cos^2(phi) - 1
        for i, j, k, l_atom, v_tor, _ in self.torsions:
            b1 = x[j] - x[i]
            b2 = x[k] - x[j]
            b3 = x[l_atom] - x[k]
            cr1 = np.cross(b1, b2)
            cr2 = np.cross(b2, b3)
            nn1 = max(float(np.linalg.norm(cr1)), 1e-6)
            nn2 = max(float(np.linalg.norm(cr2)), 1e-6)
            u1 = cr1 / nn1
            u2 = cr2 / nn2
            cos_p = float(np.clip(np.dot(u1, u2), -0.99999, 0.99999))
            # E = V * (1 - cos(2*phi)) = 2 * V * (1 - cos^2(phi))
            e_planar += 2.0 * v_tor * (1.0 - cos_p * cos_p)
            # Analytical gradient via numerical-free chain rule on cross products
            d_cos = -4.0 * v_tor * cos_p
            du1 = (d_cos / nn1) * (u2 - cos_p * u1)
            du2 = (d_cos / nn2) * (u1 - cos_p * u2)
            # d(b1 x b2) / db1, db2; d(b2 x b3) / db2, db3
            db1 = np.cross(b2, du1)
            db2 = np.cross(du1, b1) + np.cross(b3, du2)
            db3 = np.cross(du2, b2)
            grad[i] -= db1
            grad[j] += db1 - db2
            grad[k] += db2 - db3
            grad[l_atom] += db3

        # 5. Soft Lennard-Jones 12-6 non-bonded interaction (capped at small r)
        for i, j, r_star, eps in self.nonbonded:
            dv = x[i] - x[j]
            r2 = float(np.dot(dv, dv))
            r_eff2 = max(r2, 0.35 * r_star * r_star)
            sr2 = (r_star * r_star) / r_eff2
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            e_vdw += eps * (sr12 - 2.0 * sr6)
            if r2 >= 0.35 * r_star * r_star:
                # dE/dr * (dv/r) = -12 * eps * (sr12 - sr6) / r^2 * dv
                f_pref = (-12.0 * eps * (sr12 - sr6)) / max(r2, 1e-6)
                g = f_pref * dv
                grad[i] += g
                grad[j] -= g

        total = e_bond + e_angle + e_planar + e_vdw
        comp = {
            "bond": round(e_bond, 4),
            "angle": round(e_angle, 4),
            "planar_torsion": round(e_planar, 4),
            "vdw": round(e_vdw, 4),
        }
        return total, grad.ravel(), comp


def kabsch_rmsd(coords_a: NDArray[np.float64], coords_b: NDArray[np.float64]) -> float:
    """Compute minimum heavy-atom RMSD (Angstroms) after optimal Kabsch alignment."""
    if coords_a.shape != coords_b.shape or len(coords_a) == 0:
        return 0.0
    a = coords_a - np.mean(coords_a, axis=0, keepdims=True)
    b = coords_b - np.mean(coords_b, axis=0, keepdims=True)
    cov = a.T @ b
    u, _, vt = np.linalg.svd(cov)
    d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0.0 else -1.0
    diag = np.diag([1.0, 1.0, d])
    rot = vt.T @ diag @ u.T
    a_rot = a @ rot.T
    diff = a_rot - b
    return float(math.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def embed_conformer(
    mol: MoleculeGraph,
    seed: int = 42,
    max_iter: int = 160,
) -> Conformer3D:
    """Generate and UFF-minimize a 3D conformer for ``mol``."""
    n = mol.num_atoms
    if n == 0:
        return Conformer3D(mol, np.zeros((0, 3)), 0.0, {}, True, 0.0)
    if n == 1:
        return Conformer3D(mol, np.zeros((1, 3)), 0.0, {"bond": 0.0}, True, 0.0)

    rng = np.random.default_rng(seed)
    L, U = build_distance_bounds(mol)
    lo, hi = triangle_smooth(L, U)
    init_coords = metric_matrix_embed(lo, hi, rng)

    obj = _UFFObjective(mol)

    def fun(flat_x: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
        e, g, _ = obj.energy_and_grad(flat_x)
        return e, g

    res = minimize(
        fun,
        init_coords.ravel(),
        method="BFGS",
        jac=True,
        options={"maxiter": max_iter, "gtol": 1e-3},
    )
    opt_coords = res.x.reshape((n, 3))
    opt_coords -= np.mean(opt_coords, axis=0, keepdims=True)
    final_e, _, comp = obj.energy_and_grad(opt_coords.ravel())
    rmsd = kabsch_rmsd(init_coords, opt_coords)

    return Conformer3D(
        molecule=mol,
        coords=opt_coords,
        uff_energy=round(float(final_e), 4),
        strain_components=comp,
        converged=bool(res.success or final_e < 50.0),
        rmsd_to_initial=round(rmsd, 4),
    )


def generate_conformer_ensemble(
    mol: MoleculeGraph,
    n_confs: int = 5,
    rmsd_threshold: float = 0.4,
    seed: int = 42,
) -> List[Conformer3D]:
    """Generate a diverse energy-ranked 3D conformer ensemble via multi-seed DG."""
    raw: List[Conformer3D] = []
    for k in range(n_confs * 2):
        conf = embed_conformer(mol, seed=seed + k * 17)
        raw.append(conf)
    raw.sort(key=lambda c: c.uff_energy)

    kept: List[Conformer3D] = []
    for cand in raw:
        if all(kabsch_rmsd(cand.coords, prev.coords) >= rmsd_threshold for prev in kept):
            kept.append(cand)
        if len(kept) >= n_confs:
            break
    return kept if kept else raw[:1]
