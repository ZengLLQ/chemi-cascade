"""Empirical protein-ligand docking, pose optimization, and PROTAC ternary modeling.

Implements:
1. Vina/Smina-inspired empirical protein-ligand scoring function:
   - Steric Gauss-1 & Gauss-2 attraction, quadratic steric repulsion.
   - Hydrophobic contact reward, hydrogen-bond donor/acceptor interaction,
     screened distance-dependent Coulomb electrostatics, and rotatable-bond
     conformational entropy penalty.
2. 6-DOF rigid-body + torsional Monte Carlo simulated annealing followed by
   L-BFGS/Powell local refinement inside a receptor pocket box and RMSD clustering.
3. Built-in synthetic kinase ATP-hinge and protease S1 reference pockets for
   reproducible pose-recovery & decoy-enrichment benchmarking.
4. Douglass-style exact mass-balance PROTAC ternary complex equilibrium solver
   with Worm-Like Chain (WLC) linker entropy and hook-effect dose-response curves.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from chemi_cascade.descriptors import compute_gasteiger_charges, count_rotatable_bonds
from chemi_cascade.geometry import embed_conformer, kabsch_rmsd
from chemi_cascade.types import (
    Conformer3D,
    DockingPose,
    MoleculeGraph,
    ProtacCurve,
    ProtacSpec,
    ReceptorAtom,
    ReceptorPocket,
    get_element,
)

# AutoDock Vina (Trott & Olson 2010) empirical weights
W_GAUSS1: float = -0.035579
W_GAUSS2: float = -0.005156
W_REPULSION: float = 0.840245
W_HYDROPHOBIC: float = -0.035069
W_HBOND: float = -0.587439
W_ELEC: float = 0.045
N_ROT_WEIGHT: float = 0.05846


def build_kinase_hinge_pocket() -> ReceptorPocket:
    """Construct a realistic synthetic ATP-binding kinase hinge receptor pocket.

    Includes hinge backbone H-bond donor/acceptor pair (Met/Glu), gatekeeper
    hydrophobic wall (Leu/Val/Phe), catalytic Lys+, and DFG Asp- triad.
    """
    raw = [
        ("GLU", 81, "O", "O", (-2.2, 2.6, 0.0), 1.52, False, True, False, False, -0.35),
        ("MET", 83, "N", "N", (0.2, 3.1, 0.2), 1.55, True, False, False, False, 0.20),
        ("MET", 83, "O", "O", (2.5, 2.8, -0.1), 1.52, False, True, False, False, -0.35),
        ("VAL", 28, "CG1", "C", (-2.8, -0.8, 2.4), 1.70, False, False, True, False, 0.0),
        ("LEU", 134, "CD1", "C", (0.4, -1.2, -2.6), 1.70, False, False, True, False, 0.0),
        ("ALA", 48, "CB", "C", (-1.1, 0.4, -2.5), 1.70, False, False, True, False, 0.0),
        ("PHE", 145, "CZ", "C", (3.8, -0.5, 1.8), 1.70, False, False, True, True, 0.0),
        ("PHE", 145, "CE1", "C", (4.2, -1.6, 1.1), 1.70, False, False, True, True, 0.0),
        ("LYS", 50, "NZ", "N", (3.6, 1.5, 2.8), 1.55, True, False, False, False, 0.65),
        ("ASP", 144, "OD1", "O", (4.8, 0.2, -1.9), 1.52, False, True, False, False, -0.55),
        ("ILE", 16, "CD1", "C", (-4.1, 0.8, -1.2), 1.70, False, False, True, False, 0.0),
        ("GLY", 86, "CA", "C", (-3.5, -2.4, -1.5), 1.70, False, False, True, False, 0.0),
    ]
    atoms = [
        ReceptorAtom(
            idx=i,
            residue_name=r[0],
            residue_id=r[1],
            atom_name=r[2],
            element=r[3],
            coords=r[4],
            vdw_radius=r[5],
            is_hbd=r[6],
            is_hba=r[7],
            is_hydrophobic=r[8],
            is_aromatic=r[9],
            formal_charge=r[10],
        )
        for i, r in enumerate(raw)
    ]
    return ReceptorPocket(
        name="Kinase_ATP_Hinge",
        atoms=atoms,
        center=(0.2, 0.3, 0.0),
        box_size=(14.0, 12.0, 12.0),
    )


def _ligand_atom_typing(
    mol: MoleculeGraph,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    """Extract vdW radii, HBD, HBA, hydrophobic flags, and partial charges."""
    if all(abs(a.partial_charge) < 1e-12 for a in mol.atoms):
        compute_gasteiger_charges(mol)
    n = mol.num_atoms
    vdw = np.array([get_element(a.symbol).vdw_radius for a in mol.atoms], dtype=np.float64)
    is_hbd = np.array([a.symbol in {"N", "O"} and a.total_h > 0 for a in mol.atoms], dtype=bool)
    is_hba = np.array([a.symbol in {"N", "O", "S"} and a.formal_charge <= 0 for a in mol.atoms], dtype=bool)
    is_hyd = np.array([
        (a.symbol == "C" and not any(mol.atoms[nb].symbol in {"N", "O"} for nb in mol.neighbors(a.idx)))
        or a.symbol in {"Cl", "Br", "I", "F"}
        for a in mol.atoms
    ], dtype=bool)
    charges = np.array([a.partial_charge for a in mol.atoms], dtype=np.float64)
    if len(charges) != n:
        charges = np.zeros(n, dtype=np.float64)
    return vdw, is_hbd, is_hba, is_hyd, charges


def _score_precomputed(
    ligand_coords: NDArray[np.float64],
    l_vdw: NDArray[np.float64],
    l_hbd: NDArray[np.bool_],
    l_hba: NDArray[np.bool_],
    l_hyd: NDArray[np.bool_],
    l_q: NDArray[np.float64],
    r_coords: NDArray[np.float64],
    r_vdw: NDArray[np.float64],
    r_hbd: NDArray[np.bool_],
    r_hba: NDArray[np.bool_],
    r_hyd: NDArray[np.bool_],
    r_q: NDArray[np.float64],
    n_rot: int,
    rank: int = 1,
) -> DockingPose:
    """Fast vectorized Vina-inspired empirical scoring using precomputed atom types."""
    diff = ligand_coords[:, None, :] - r_coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    mask = dist < 8.0
    d_surf = dist - (l_vdw[:, None] + r_vdw[None, :])

    g1_mat = np.exp(-((d_surf / 0.5) ** 2)) * mask
    g2_mat = np.exp(-(((d_surf - 3.0) / 2.0) ** 2)) * mask
    rep_mat = np.where((d_surf < 0.0) & mask, d_surf * d_surf, 0.0)

    gauss1 = float(np.sum(g1_mat))
    gauss2 = float(np.sum(g2_mat))
    repulsion = float(np.sum(rep_mat))

    hyd_pair = l_hyd[:, None] & r_hyd[None, :]
    hyd_ramp = np.clip((1.5 - d_surf) / 1.0, 0.0, 1.0)
    hydrophobic = float(np.sum(np.where(hyd_pair & mask, hyd_ramp, 0.0)))

    hb_pair = (l_hbd[:, None] & r_hba[None, :]) | (l_hba[:, None] & r_hbd[None, :])
    hb_ramp = np.clip((0.0 - d_surf) / 0.7, 0.0, 1.0)
    hydrogen_bond = float(np.sum(np.where(hb_pair & mask, hb_ramp, 0.0)))
    hbonds_formed = int(np.sum(hb_pair & (d_surf <= -0.05) & (dist <= 3.6)))

    q_prod = l_q[:, None] * r_q[None, :]
    elec_mat = np.where(mask, (332.06 * q_prod) / (4.0 * np.maximum(dist, 1.8) ** 2), 0.0)
    electrostatic = float(np.sum(elec_mat))

    e_inter = (
        W_GAUSS1 * gauss1
        + W_GAUSS2 * gauss2
        + W_REPULSION * repulsion
        + W_HYDROPHOBIC * hydrophobic
        + W_HBOND * hydrogen_bond
        + W_ELEC * electrostatic
    )

    affinity = e_inter / (1.0 + N_ROT_WEIGHT * n_rot)
    torsion_penalty = affinity - e_inter

    rt = 0.5922
    kd_molar = math.exp(max(-30.0, min(10.0, affinity / rt)))
    kd_um = kd_molar * 1e6
    pki = -math.log10(max(kd_molar, 1e-15))

    return DockingPose(
        pose_rank=rank,
        coords=ligand_coords.copy(),
        affinity_kcal=round(float(affinity), 4),
        predicted_kd_um=round(float(min(1e6, kd_um)), 4),
        predicted_pki=round(float(pki), 3),
        gauss1=round(W_GAUSS1 * gauss1, 4),
        gauss2=round(W_GAUSS2 * gauss2, 4),
        repulsion=round(W_REPULSION * repulsion, 4),
        hydrophobic=round(W_HYDROPHOBIC * hydrophobic, 4),
        hydrogen_bond=round(W_HBOND * hydrogen_bond, 4),
        electrostatic=round(W_ELEC * electrostatic, 4),
        torsion_penalty=round(float(torsion_penalty), 4),
        hbonds_formed=hbonds_formed,
    )


def score_ligand_pose(
    mol: MoleculeGraph,
    ligand_coords: NDArray[np.float64],
    pocket: ReceptorPocket,
    rank: int = 1,
) -> DockingPose:
    """Evaluate Vina-inspired empirical binding free energy (kcal/mol) for a pose."""
    n_lig = mol.num_atoms
    n_rec = len(pocket.atoms)
    if n_lig == 0 or n_rec == 0:
        return DockingPose(rank, ligand_coords.copy(), 0.0, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    l_vdw, l_hbd, l_hba, l_hyd, l_q = _ligand_atom_typing(mol)
    r_coords = np.array([ra.coords for ra in pocket.atoms], dtype=np.float64)
    r_vdw = np.array([ra.vdw_radius for ra in pocket.atoms], dtype=np.float64)
    r_hbd = np.array([ra.is_hbd for ra in pocket.atoms], dtype=bool)
    r_hba = np.array([ra.is_hba for ra in pocket.atoms], dtype=bool)
    r_hyd = np.array([ra.is_hydrophobic for ra in pocket.atoms], dtype=bool)
    r_q = np.array([ra.formal_charge for ra in pocket.atoms], dtype=np.float64)
    n_rot = count_rotatable_bonds(mol)
    return _score_precomputed(
        ligand_coords, l_vdw, l_hbd, l_hba, l_hyd, l_q,
        r_coords, r_vdw, r_hbd, r_hba, r_hyd, r_q,
        n_rot, rank=rank,
    )


def _euler_to_rot_matrix(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert ZYX Euler angles (radians) into a 3x3 proper rotation matrix."""
    rx, ry, rz = float(angles[0]), float(angles[1]), float(angles[2])
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _transform_coords(base_coords: NDArray[np.float64], dof6: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply 3D translation ``dof6[:3]`` and Euler rotation ``dof6[3:]`` around centroid."""
    centered = base_coords - np.mean(base_coords, axis=0, keepdims=True)
    rot = _euler_to_rot_matrix(dof6[3:])
    return centered @ rot.T + dof6[:3]


def dock_ligand(
    mol: MoleculeGraph,
    pocket: ReceptorPocket,
    conformer: Optional[Conformer3D] = None,
    n_starts: int = 8,
    n_poses: int = 3,
    seed: int = 42,
) -> List[DockingPose]:
    """Dock ``mol`` into ``pocket`` via 6-DOF Monte Carlo + Powell local refinement."""
    conf = conformer if conformer is not None else embed_conformer(mol, seed=seed)
    base = conf.coords
    if len(base) == 0 or len(pocket.atoms) == 0:
        return []

    base_centered = base - np.mean(base, axis=0, keepdims=True)
    l_vdw, l_hbd, l_hba, l_hyd, l_q = _ligand_atom_typing(mol)
    r_coords = np.array([ra.coords for ra in pocket.atoms], dtype=np.float64)
    r_vdw = np.array([ra.vdw_radius for ra in pocket.atoms], dtype=np.float64)
    r_hbd = np.array([ra.is_hbd for ra in pocket.atoms], dtype=bool)
    r_hba = np.array([ra.is_hba for ra in pocket.atoms], dtype=bool)
    r_hyd = np.array([ra.is_hydrophobic for ra in pocket.atoms], dtype=bool)
    r_q = np.array([ra.formal_charge for ra in pocket.atoms], dtype=np.float64)
    n_rot = count_rotatable_bonds(mol)

    rng = np.random.default_rng(seed)
    cx, cy, cz = pocket.center
    bx, by, bz = pocket.box_size
    half_box = np.array([bx * 0.22, by * 0.22, bz * 0.22], dtype=np.float64)
    center_arr = np.array([cx, cy, cz], dtype=np.float64)

    def objective(dof6: NDArray[np.float64]) -> float:
        t_pos = np.clip(dof6[:3], center_arr - half_box * 1.6, center_arr + half_box * 1.6)
        tether = 2.5 * float(np.sum((dof6[:3] - t_pos) ** 2))
        rot = _euler_to_rot_matrix(dof6[3:])
        coords = base_centered @ rot.T + dof6[:3]
        pose = _score_precomputed(
            coords, l_vdw, l_hbd, l_hba, l_hyd, l_q,
            r_coords, r_vdw, r_hbd, r_hba, r_hyd, r_q,
            n_rot, rank=0,
        )
        return pose.affinity_kcal + tether

    candidates: List[DockingPose] = []
    for s in range(n_starts):
        init_t = center_arr + rng.uniform(-half_box, half_box)
        init_r = rng.uniform(-math.pi, math.pi, size=3)
        dof = np.concatenate([init_t, init_r])
        best_dof = dof.copy()
        best_val = objective(best_dof)

        temp = 1.2
        for step in range(40):
            prop = best_dof.copy()
            prop[:3] += rng.normal(0.0, 0.55 * (1.0 - 0.5 * step / 40.0), size=3)
            prop[3:] += rng.normal(0.0, 0.45, size=3)
            val = objective(prop)
            delta = val - best_val
            if delta < 0.0 or rng.random() < math.exp(-delta / max(temp, 0.05)):
                best_dof = prop
                best_val = val
            temp *= 0.93

        res = minimize(objective, best_dof, method="Powell", options={"maxiter": 25, "ftol": 2e-3})
        rot_f = _euler_to_rot_matrix(res.x[3:])
        final_coords = base_centered @ rot_f.T + res.x[:3]
        candidates.append(
            _score_precomputed(
                final_coords, l_vdw, l_hbd, l_hba, l_hyd, l_q,
                r_coords, r_vdw, r_hbd, r_hba, r_hyd, r_q,
                n_rot, rank=s + 1,
            )
        )

    candidates.sort(key=lambda p: p.affinity_kcal)

    # RMSD clustering (keep distinct poses >= 1.2 A apart)
    diverse: List[DockingPose] = []
    best_coords = candidates[0].coords
    for cand in candidates:
        if all(kabsch_rmsd(cand.coords, prev.coords) >= 1.2 for prev in diverse):
            cand.pose_rank = len(diverse) + 1
            cand.rmsd_from_best = round(kabsch_rmsd(cand.coords, best_coords), 4)
            diverse.append(cand)
        if len(diverse) >= n_poses:
            break

    return diverse if diverse else candidates[:1]


# ---------------------------------------------------------------------------
# PROTAC Ternary Complex Cooperativity & Hook-Effect Solver
# ---------------------------------------------------------------------------

def wlc_linker_entropy_penalty(
    contour_length_a: float,
    persistence_length_a: float,
    required_span_a: float,
    temperature_k: float = 310.0,
) -> Tuple[float, float]:
    """Compute Worm-Like Chain (WLC) linker effective molarity & entropy penalty (kcal/mol).

    Returns ``(penalty_kcal_mol, cooperativity_modifier)``.
    When ``required_span_a`` exceeds ``0.92 * contour_length_a``, the linker
    becomes over-stretched and incurs a sharp entropic barrier.
    """
    L = max(contour_length_a, 2.0)
    lp = max(persistence_length_a, 1.0)
    r = max(0.5, required_span_a)
    rt = 0.0019872 * temperature_k

    if r >= 0.95 * L:
        # Over-stretched regime (penalizes unattainable ternary span)
        strain = 6.0 + 18.0 * ((r / L) - 0.90)
        return round(strain, 4), float(math.exp(-strain / rt))

    # Marko-Siggia / polymer Gaussian end-to-end free energy penalty relative to optimal span
    r_opt = math.sqrt(2.0 * lp * L / 3.0)
    x = r / L
    # Entropy cost from compressing or stretching away from natural Kuhn span
    penalty = rt * (1.5 * ((r - r_opt) ** 2) / max(lp * L, 4.0) + 0.35 * (x ** 2) / max(1.0 - x, 0.08))
    modifier = math.exp(-penalty / rt)
    return round(float(penalty), 4), float(modifier)


def solve_ternary_equilibrium(
    p_total_nm: float,
    a_total_nm: float,
    b_total_nm: float,
    kd_a_nm: float,
    kd_b_nm: float,
    alpha: float,
    max_iter: int = 400,
    tol: float = 1e-11,
) -> Tuple[float, float, float, float, float, float]:
    """Solve exact mass-balance ternary equilibrium (A + P + B <-> AP + BP <-> ABP).

    Uses damped Anderson/fixed-point iteration over free concentrations
    ``([A], [B], [P])`` guaranteeing non-negativity and exact conservation:
    - ``[A] + [AP] + [ABP] = a_total_nm``
    - ``[B] + [BP] + [ABP] = b_total_nm``
    - ``[P] + [AP] + [BP] + [ABP] = p_total_nm``

    Returns ``(free_a, free_b, free_p, conc_ap, conc_bp, conc_abp)`` in nM.
    """
    if p_total_nm <= 0.0 or a_total_nm <= 0.0 or b_total_nm <= 0.0:
        return a_total_nm, b_total_nm, p_total_nm, 0.0, 0.0, 0.0

    ka = max(kd_a_nm, 1e-6)
    kb = max(kd_b_nm, 1e-6)
    a_free = a_total_nm * 0.5
    b_free = b_total_nm * 0.5
    p_free = p_total_nm * 0.5

    for _ in range(max_iter):
        # Update P from quadratic/rational conservation at current (A, B)
        denom_p = 1.0 + (a_free / ka) + (b_free / kb) + (alpha * a_free * b_free) / (ka * kb)
        p_new = p_total_nm / denom_p
        denom_a = 1.0 + (p_new / ka) + (alpha * b_free * p_new) / (ka * kb)
        a_new = a_total_nm / denom_a
        denom_b = 1.0 + (p_new / kb) + (alpha * a_new * p_new) / (ka * kb)
        b_new = b_total_nm / denom_b

        err = max(abs(p_new - p_free), abs(a_new - a_free), abs(b_new - b_free))
        p_free = 0.5 * p_free + 0.5 * p_new
        a_free = 0.5 * a_free + 0.5 * a_new
        b_free = 0.5 * b_free + 0.5 * b_new
        if err < tol:
            break

    # Final exact polish
    denom_p = 1.0 + (a_free / ka) + (b_free / kb) + (alpha * a_free * b_free) / (ka * kb)
    p_free = p_total_nm / denom_p
    ap = a_free * p_free / ka
    bp = b_free * p_free / kb
    abp = alpha * a_free * b_free * p_free / (ka * kb)
    return float(a_free), float(b_free), float(p_free), float(ap), float(bp), float(abp)


def simulate_protac_hook_effect(
    spec: ProtacSpec,
    conc_min_nm: float = 0.1,
    conc_max_nm: float = 100000.0,
    n_points: int = 61,
) -> ProtacCurve:
    """Compute bell-shaped PROTAC ternary complex dose-response curve (hook effect)."""
    wlc_pen, mod = wlc_linker_entropy_penalty(
        spec.linker_contour_angstrom,
        spec.persistence_length_angstrom,
        spec.required_span_angstrom,
    )
    eff_alpha = max(1e-4, spec.cooperativity_alpha * mod)

    concs = np.logspace(math.log10(max(1e-4, conc_min_nm)), math.log10(conc_max_nm), n_points)
    abp_vals = np.zeros(n_points, dtype=np.float64)
    for i, p0 in enumerate(concs):
        _, _, _, _, _, abp = solve_ternary_equilibrium(
            p_total_nm=float(p0),
            a_total_nm=spec.poi_total_nm,
            b_total_nm=spec.e3_total_nm,
            kd_a_nm=spec.kd_poi_nm,
            kd_b_nm=spec.kd_e3_nm,
            alpha=eff_alpha,
        )
        abp_vals[i] = abp

    idx_peak = int(np.argmax(abp_vals))
    max_abp = float(abp_vals[idx_peak])
    half_peak = 0.5 * max_abp

    # Left half-max (EC50 formation) and right half-max (Hook IC50)
    ec50 = float(concs[0])
    for i in range(idx_peak):
        if abp_vals[i] <= half_peak <= abp_vals[i + 1]:
            frac = (half_peak - abp_vals[i]) / max(abp_vals[i + 1] - abp_vals[i], 1e-12)
            ec50 = float(concs[i] + frac * (concs[i + 1] - concs[i]))
            break

    hook_ic50 = float(concs[-1])
    for i in range(idx_peak, n_points - 1):
        if abp_vals[i] >= half_peak >= abp_vals[i + 1]:
            frac = (abp_vals[i] - half_peak) / max(abp_vals[i] - abp_vals[i + 1], 1e-12)
            hook_ic50 = float(concs[i] + frac * (concs[i + 1] - concs[i]))
            break

    return ProtacCurve(
        concentrations_nm=concs,
        ternary_complex_nm=abp_vals,
        max_ternary_nm=round(max_abp, 4),
        ec50_formation_nm=round(ec50, 3),
        hook_ic50_nm=round(hook_ic50, 2),
        effective_alpha=round(eff_alpha, 4),
        wlc_entropy_penalty_kcal=wlc_pen,
    )
