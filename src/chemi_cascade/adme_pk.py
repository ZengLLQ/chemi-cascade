"""Mechanistic ADME scaling, Henderson-Hasselbalch ionization, and 2-compartment PBPK.

Bridges GNN-predicted intrinsic clearance, permeability, and lipophilicity to
human physiological pharmacokinetics:
- Acid/base pKa microspecies estimation & pH-dependent logD(pH).
- Well-stirred and parallel-tube hepatic clearance models scaling microsomal
  CL_int (uL/min/mg) via MPPGL (40 mg/g liver) and liver weight (1500 g).
- Oral absorption (Fa from Caco-2 Papp), gut-wall escape (Fg), and first-pass
  hepatic availability (Fh = 1 - E_H).
- 2-Compartment oral PBPK ODE solver (depot gut -> central plasma -> peripheral
  tissue) via SciPy RK45, reporting Cmax, Tmax, AUC_0_inf, terminal t_1/2,beta,
  and hERG safety margin.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.featurizer import featurize_molecule
from chemi_cascade.gnn import MolecularMPNN
from chemi_cascade.smiles import has_substructure, parse_smiles
from chemi_cascade.types import ADMEPrediction, MoleculeGraph, PKProfile

# Human physiological PBPK constants (70 kg adult)
HUMAN_QH_L_H: float = 90.0        # Hepatic blood flow (L/h)
HUMAN_LIVER_G: float = 1500.0     # Liver weight (g)
MPPGL_MG_G: float = 40.0          # Microsomal protein per gram liver (mg/g)


def estimate_dominant_pka(mol: MoleculeGraph) -> Tuple[Optional[float], Optional[float]]:
    """Estimate strongest acidic and basic pKa values from structural motifs."""
    acidic_pka: Optional[float] = None
    basic_pka: Optional[float] = None

    if has_substructure(mol, "C(=O)[OX2H1]") or has_substructure(mol, "C(=O)O"):
        # Carboxylic acid check (distinguish ester vs free acid via O H-count)
        for a in mol.atoms:
            if a.symbol == "C":
                nbs = [mol.atoms[nb] for nb in mol.neighbors(a.idx)]
                has_db_o = any(
                    nb.symbol == "O"
                    and (b := mol.get_bond(a.idx, nb.idx)) is not None
                    and b.order >= 1.9
                    for nb in nbs
                )
                has_oh = any(
                    nb.symbol == "O"
                    and nb.total_h >= 1
                    and (b := mol.get_bond(a.idx, nb.idx)) is not None
                    and b.order < 1.5
                    for nb in nbs
                )
                if has_db_o and has_oh:
                    acidic_pka = 4.2
                    break

    if acidic_pka is None and has_substructure(mol, "c[OX2H1]"):
        acidic_pka = 9.8  # phenolic OH

    # Basic aliphatic amine (sp3 N not bonded to carbonyl/sulfonyl)
    for a in mol.atoms:
        if a.symbol == "N" and not a.is_aromatic and a.hybridization == "sp3":
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
                deg = mol.degree(a.idx)
                basic_pka = 10.2 if deg == 1 else (9.6 if deg == 2 else 8.8)
                break
        elif a.symbol == "N" and a.is_aromatic and a.total_h == 0:
            basic_pka = 5.1 if basic_pka is None else basic_pka

    return acidic_pka, basic_pka


def henderson_hasselbalch_logd(
    clogp: float,
    acidic_pka: Optional[float],
    basic_pka: Optional[float],
    ph: float = 7.4,
) -> Tuple[float, float]:
    """Compute (logD_at_pH, fraction_unionized) via Henderson-Hasselbalch."""
    ion_factor = 1.0
    if acidic_pka is not None:
        ion_factor += 10.0 ** np.clip(ph - acidic_pka, -12.0, 12.0)
    if basic_pka is not None:
        ion_factor += 10.0 ** np.clip(basic_pka - ph, -12.0, 12.0)
    f_unionized = 1.0 / ion_factor
    logd = clogp + math.log10(max(f_unionized, 1e-12))
    return round(float(logd), 3), round(float(f_unionized), 5)


def scale_hepatic_clearance(
    cl_int_ul_min_mg: float,
    fup: float = 0.15,
    rb: float = 1.0,
    model: str = "well_stirred",
) -> Tuple[float, float]:
    """Scale microsomal CL_int (uL/min/mg) to whole-body human hepatic CL_H (L/h).

    Returns ``(cl_h_l_h, extraction_ratio_eh)``.
    """
    # Convert uL/min/mg -> L/h whole liver:
    # cl_int_l_h = cl_int * 1e-6 (L/uL) * 60 (min/h) * MPPGL * Liver_g
    cl_int_liver_l_h = max(1e-4, cl_int_ul_min_mg) * 1e-6 * 60.0 * MPPGL_MG_G * HUMAN_LIVER_G
    qh = HUMAN_QH_L_H
    fu_cl = max(1e-6, fup) * cl_int_liver_l_h

    if model == "well_stirred":
        cl_h = (qh * fu_cl) / (qh + fu_cl / max(rb, 1e-3))
    elif model == "parallel_tube":
        eh = 1.0 - math.exp(-fu_cl / (qh * max(rb, 1e-3)))
        cl_h = qh * eh
    else:
        raise ValueError(f"Unknown hepatic clearance model: {model!r}")

    eh = min(0.999, max(0.0, cl_h / qh))
    return round(float(cl_h), 4), round(float(eh), 4)


def simulate_two_compartment_pk(
    adme: ADMEPrediction,
    mw: float,
    dose_mg: float = 100.0,
    tau_hours: float = 24.0,
    n_points: int = 241,
    fup: float = 0.15,
    vd_central_l: float = 35.0,
    vd_peripheral_l: float = 75.0,
    q_dist_l_h: float = 12.0,
) -> PKProfile:
    """Integrate 2-compartment oral PBPK ODE system over [0, tau_hours].

    State vector ``y = [A_gut (mg), A_central (mg), A_peripheral (mg)]``:
    - ``dA_gut/dt = -k_a * A_gut`` with ``A_gut(0) = F_a * F_g * Dose``
    - ``dA_c/dt = k_a * F_h * A_gut - (CL_H + Q)/V_c * A_c + (Q/V_p) * A_p``
      (applying first-pass hepatic extraction at portal input via ``F_h``)
    - ``dA_p/dt = (Q/V_c) * A_c - (Q/V_p) * A_p``
    """
    cl_h, eh = scale_hepatic_clearance(adme.cl_int_ul_min_mg, fup=fup, model="well_stirred")
    fh = max(0.01, 1.0 - eh)
    # Oral fraction absorbed sigmoidal function of Caco-2 Papp (10^-6 cm/s)
    fa = 1.0 / (1.0 + math.exp(-0.35 * (adme.papp_caco2 - 4.0)))
    fg = max(0.45, 1.0 - 0.35 * eh)
    f_oral = fa * fg * fh

    # Absorption rate constant ka (1/h) from permeability
    ka = float(np.clip(0.3 + 0.12 * adme.papp_caco2, 0.25, 3.0))
    kel = cl_h / max(vd_central_l, 1e-3)
    k12 = q_dist_l_h / max(vd_central_l, 1e-3)
    k21 = q_dist_l_h / max(vd_peripheral_l, 1e-3)

    def rhs(_t: float, y: NDArray[np.float64]) -> List[float]:
        ag, ac, ap = float(y[0]), float(y[1]), float(y[2])
        d_ag = -ka * ag
        d_ac = ka * fh * ag - (kel + k12) * ac + k21 * ap
        d_ap = k12 * ac - k21 * ap
        return [d_ag, d_ac, d_ap]

    y0 = np.array([dose_mg * fa * fg, 0.0, 0.0], dtype=np.float64)
    t_eval = np.linspace(0.0, tau_hours, n_points)
    sol = solve_ivp(rhs, (0.0, tau_hours), y0, t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9)

    c_plasma = np.maximum(0.0, sol.y[1] / vd_central_l) * 1000.0      # mg/L -> ug/L
    c_periph = np.maximum(0.0, sol.y[2] / vd_peripheral_l) * 1000.0

    idx_max = int(np.argmax(c_plasma))
    cmax = float(c_plasma[idx_max])
    tmax = float(t_eval[idx_max])

    # Analytical AUC_0_inf = Dose * F / CL_H (in ug*h/L = mg*h/L * 1000)
    auc_inf = (dose_mg * f_oral / max(cl_h, 1e-4)) * 1000.0

    # Terminal disposition eigenvalue beta of 2-compartment linear matrix
    tr = kel + k12 + k21
    det = kel * k21
    disc = max(0.0, tr * tr - 4.0 * det)
    beta = 0.5 * (tr - math.sqrt(disc))
    half_life_beta = math.log(2.0) / max(beta, 1e-6)

    # Free plasma Cmax (uM) vs hERG IC50 (uM) safety margin
    cmax_free_um = (cmax * fup) / max(mw, 1.0)
    herg_ic50_um = (10.0 ** (-adme.herg_pic50)) * 1e6
    margin = herg_ic50_um / max(cmax_free_um, 1e-9)

    return PKProfile(
        dose_mg=dose_mg,
        tau_hours=tau_hours,
        times_h=t_eval,
        c_plasma_ug_l=c_plasma,
        c_peripheral_ug_l=c_periph,
        cmax_ug_l=round(cmax, 3),
        tmax_h=round(tmax, 3),
        auc_0_inf=round(auc_inf, 2),
        half_life_beta_h=round(half_life_beta, 3),
        bioavailability_f=round(f_oral, 4),
        hepatic_cl_l_h=cl_h,
        therapeutic_margin=round(float(min(1e6, margin)), 2),
    )


def synthetic_adme_ground_truth(mol: MoleculeGraph) -> NDArray[np.float64]:
    """Compute physically grounded synthetic ADME labels for training & validation.

    Targets: ``[log_s, log_d_74, cl_int_log10, papp_log10, log_bb, herg_pic50]``.
    Combines Delaney ESOL-style solubility physics, Henderson-Hasselbalch logD,
    lipophilicity/soft-spot clearance, TPSA-dependent permeability, Clark logBB,
    and basic-amine/aromatic hERG pharmacophore risk.
    """
    prof = compute_descriptors(mol)
    acid_pka, basic_pka = estimate_dominant_pka(mol)
    logd_74, _ = henderson_hasselbalch_logd(prof.clogp, acid_pka, basic_pka, ph=7.4)

    # 1. ESOL-like aqueous solubility logS (mol/L)
    rb_term = -0.0062 * prof.molecular_weight + 0.066 * prof.rotatable_bonds
    arom_prop = prof.aromatic_ring_count / max(1.0, prof.heavy_atom_count / 6.0)
    log_s = 0.16 - 0.63 * prof.clogp + rb_term - 0.74 * arom_prop + 0.015 * prof.tpsa

    # 2. Microsomal intrinsic clearance log10(uL/min/mg)
    sp3_ch = sum(1 for a in mol.atoms if a.symbol == "C" and a.hybridization == "sp3" and a.total_h > 0)
    hal_block = sum(1 for a in mol.atoms if a.symbol in {"F", "Cl"})
    cl_int_log10 = (
        0.75
        + 0.24 * max(0.0, logd_74)
        + 0.05 * sp3_ch
        - 0.14 * hal_block
        - 0.004 * prof.tpsa
    )
    cl_int_log10 = float(np.clip(cl_int_log10, 0.1, 3.2))

    # 3. Caco-2 permeability log10(10^-6 cm/s)
    papp_log10 = 1.25 - 0.011 * prof.tpsa + 0.14 * logd_74 - 0.09 * prof.hbd - 0.0012 * prof.molecular_weight
    papp_log10 = float(np.clip(papp_log10, -1.0, 2.0))

    # 4. Clark-like blood-brain barrier logBB
    log_bb = 0.152 * prof.clogp - 0.0148 * prof.tpsa + 0.139
    if acid_pka is not None and acid_pka < 5.5:
        log_bb -= 0.45
    log_bb = float(np.clip(log_bb, -2.5, 1.5))

    # 5. hERG channel pIC50 (elevated by basic amine + lipophilicity + aromatic rings)
    has_basic = 1.0 if (basic_pka is not None and basic_pka > 7.5) else 0.0
    herg_pic50 = 4.1 + 0.65 * has_basic + 0.22 * max(0.0, prof.clogp) + 0.18 * prof.aromatic_ring_count
    if acid_pka is not None and acid_pka < 5.5:
        herg_pic50 -= 0.6
    herg_pic50 = float(np.clip(herg_pic50, 3.5, 8.5))

    return np.array([log_s, logd_74, cl_int_log10, papp_log10, log_bb, herg_pic50], dtype=np.float64)


# Curated diverse reference medicinal chemistry library (32 drug-like scaffolds)
REFERENCE_DRUG_SMILES: Tuple[Tuple[str, str], ...] = (
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("Diclofenac", "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("Naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("Lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("Fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"),
    ("Gefitinib_scaffold", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
    ("Imatinib_frag", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("Celecoxib_analog", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(=O)(=O)N)cc2)cc1"),
    ("Atorvastatin_frag", "CC(C)c1n(CCC(O)CC(O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1"),
    ("Metformin_analog", "CN(C)C(=N)NC(=N)N"),
    ("Ciprofloxacin_scaffold", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
    ("Erlotinib_frag", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"),
    ("Diazepam", "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"),
    ("Sertraline", "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc12"),
    ("Atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("Omeprazole_scaffold", "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1"),
    ("Sorafenib", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1"),
    ("Dasatinib_frag", "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"),
    ("Furosemide", "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl"),
    ("Indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1"),
    ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("Sildenafil_frag", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"),
    ("Tamoxifen_frag", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ("Fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    ("Venlafaxine", "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1"),
    ("Lapatinib_frag", "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1"),
    ("Prazosin", "COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC"),
    ("Losartan_frag", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"),
)


def build_calibrated_adme_model(
    epochs: int = 30,
    hidden_dim: int = 40,
    seed: int = 42,
) -> Tuple[MolecularMPNN, Dict[str, float]]:
    """Train a calibrated ``MolecularMPNN`` on the reference medicinal chemistry set."""
    mols = [parse_smiles(smi, name=name) for name, smi in REFERENCE_DRUG_SMILES]
    graphs = [featurize_molecule(m) for m in mols]
    targets = np.vstack([synthetic_adme_ground_truth(m) for m in mols])

    model = MolecularMPNN(hidden_dim=hidden_dim, num_layers=2, seed=seed)
    hist = model.fit(graphs, targets, epochs=epochs, batch_size=8, lr=4e-3, seed=seed)

    preds = np.vstack([model.predict_raw(g) for g in graphs])
    rmse_all = float(np.sqrt(np.mean((preds - targets) ** 2)))
    metrics = {
        "final_train_loss": round(hist.train_losses[-1], 5) if hist.train_losses else 0.0,
        "overall_rmse": round(rmse_all, 4),
    }
    return model, metrics
