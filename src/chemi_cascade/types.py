"""Core chemical data structures, physical constants, and empirical parameter tables.

This module defines the canonical immutable and mutable representations of atoms,
bonds, molecular graphs, 3D conformers, receptor pockets, docking poses, ADME/PK
profiles, PROTAC ternary complexes, and optimization traces used throughout
``chemi_cascade``.

Physical & Chemical Units:
- Atomic mass: Daltons (g/mol).
- Bond lengths & 3D coordinates: Angstroms (A).
- Partial charges: elementary charge units (e).
- Energies & binding free energies: kcal/mol.
- Clearance: mL/min/kg or uL/min/mg protein; permeability: 10^-6 cm/s.
- Concentrations: micromolar (uM) or mg/L; time: hours (h).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ElementData:
    """Periodic-table and force-field physical constants for a chemical element."""

    atomic_number: int
    symbol: str
    atomic_mass: float
    covalent_radius: float
    vdw_radius: float
    electronegativity: float
    default_valences: Tuple[int, ...]
    gasteiger_a: float
    gasteiger_b: float
    gasteiger_c: float


ELEMENT_TABLE: Dict[str, ElementData] = {
    "H": ElementData(1, "H", 1.008, 0.31, 1.20, 2.20, (1,), 7.17, 6.24, -0.56),
    "B": ElementData(5, "B", 10.81, 0.84, 1.92, 2.04, (3,), 5.98, 6.82, 1.60),
    "C": ElementData(6, "C", 12.011, 0.76, 1.70, 2.55, (4,), 7.98, 9.18, 1.88),
    "N": ElementData(7, "N", 14.007, 0.71, 1.55, 3.04, (3, 5), 11.54, 10.82, 1.36),
    "O": ElementData(8, "O", 15.999, 0.66, 1.52, 3.44, (2,), 14.18, 12.92, 1.39),
    "F": ElementData(9, "F", 18.998, 0.57, 1.47, 3.98, (1,), 14.66, 13.85, 2.31),
    "P": ElementData(15, "P", 30.974, 1.07, 1.80, 2.19, (3, 5), 8.90, 8.24, 0.96),
    "S": ElementData(16, "S", 32.06, 1.05, 1.80, 2.58, (2, 4, 6), 10.14, 9.13, 1.38),
    "Cl": ElementData(17, "Cl", 35.45, 1.02, 1.75, 3.16, (1,), 11.00, 9.69, 1.35),
    "Br": ElementData(35, "Br", 79.904, 1.20, 1.85, 2.96, (1,), 10.08, 8.47, 1.16),
    "I": ElementData(53, "I", 126.904, 1.39, 1.98, 2.66, (1,), 9.90, 7.96, 0.96),
    "Si": ElementData(14, "Si", 28.085, 1.11, 2.10, 1.90, (4,), 7.30, 6.56, 0.66),
    "Fe": ElementData(26, "Fe", 55.845, 1.32, 2.00, 1.83, (2, 3), 6.50, 5.80, 0.50),
    "Zn": ElementData(30, "Zn", 65.38, 1.22, 1.39, 1.65, (2,), 6.10, 5.50, 0.45),
    "Na": ElementData(11, "Na", 22.990, 1.66, 2.27, 0.93, (1,), 3.00, 3.20, 0.20),
    "K": ElementData(19, "K", 39.098, 2.03, 2.75, 0.82, (1,), 2.70, 2.90, 0.20),
    "Mg": ElementData(12, "Mg", 24.305, 1.41, 1.73, 1.31, (2,), 4.10, 4.00, 0.30),
    "Ca": ElementData(20, "Ca", 40.078, 1.76, 2.31, 1.00, (2,), 3.50, 3.50, 0.25),
    "Se": ElementData(34, "Se", 78.971, 1.20, 1.90, 2.55, (2, 4, 6), 9.60, 8.60, 1.10),
}

ATOMIC_NUMBER_TO_SYMBOL: Dict[int, str] = {
    data.atomic_number: sym for sym, data in ELEMENT_TABLE.items()
}

ORGANIC_SUBSET: FrozenSet[str] = frozenset({"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"})
HALOGENS: FrozenSet[str] = frozenset({"F", "Cl", "Br", "I"})
HETEROATOMS: FrozenSet[str] = frozenset({"N", "O", "S", "P"})

# Gasteiger-Marsili (1980) hybridization-specific (a, b, c) parameters in eV
GASTEIGER_HYBRID_PARAMS: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    ("H", "s"): (7.17, 6.24, -0.56),
    ("C", "sp3"): (7.98, 9.18, 1.88),
    ("C", "sp2"): (8.79, 9.32, 1.51),
    ("C", "sp"): (10.39, 9.45, 0.73),
    ("N", "sp3"): (11.54, 10.82, 1.36),
    ("N", "sp2"): (12.87, 11.15, 0.85),
    ("N", "sp"): (15.68, 11.70, -0.27),
    ("O", "sp3"): (14.18, 12.92, 1.39),
    ("O", "sp2"): (17.07, 13.79, 0.47),
    ("F", "sp3"): (14.66, 13.85, 2.31),
    ("Cl", "sp3"): (11.00, 9.69, 1.35),
    ("Br", "sp3"): (10.08, 8.47, 1.16),
    ("I", "sp3"): (9.90, 7.96, 0.96),
    ("S", "sp3"): (10.14, 9.13, 1.38),
    ("S", "sp2"): (10.88, 9.49, 1.33),
    ("P", "sp3"): (8.90, 8.24, 0.96),
}


@dataclass
class Atom:
    """Graph representation of a heavy atom (and optional explicit hydrogens)."""

    idx: int
    symbol: str
    atomic_number: int
    formal_charge: int = 0
    isotope: Optional[int] = None
    is_aromatic: bool = False
    explicit_h: int = 0
    implicit_h: int = 0
    chiral_tag: str = ""
    hybridization: str = "sp3"
    in_ring: bool = False
    ring_sizes: Tuple[int, ...] = ()
    partial_charge: float = 0.0

    @property
    def total_h(self) -> int:
        """Return total attached hydrogen count (explicit + implicit)."""
        return self.explicit_h + self.implicit_h


@dataclass
class Bond:
    """Undirected covalent bond connecting two atom indices in a MoleculeGraph."""

    idx: int
    atom_a: int
    atom_b: int
    order: float = 1.0
    is_aromatic: bool = False
    is_conjugated: bool = False
    in_ring: bool = False
    stereo: str = ""

    def other(self, atom_idx: int) -> int:
        """Return the neighbor atom index across this bond."""
        if atom_idx == self.atom_a:
            return self.atom_b
        if atom_idx == self.atom_b:
            return self.atom_a
        raise ValueError(f"Atom {atom_idx} is not incident to bond {self.idx}")


@dataclass
class MoleculeGraph:
    """In-memory topological and annotated molecular graph.

    Invariants:
    - ``atoms[i].idx == i`` for all ``0 <= i < len(atoms)``.
    - ``bonds[k].idx == k`` and endpoints are valid indices in ``atoms``.
    - ``adjacency[i]`` maps neighbor index ``j`` to ``bond_idx``.
    """

    smiles: str
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    adjacency: Dict[int, Dict[int, int]] = field(default_factory=dict)
    rings: List[Tuple[int, ...]] = field(default_factory=list)
    canonical_smiles: str = ""
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        """Return number of heavy (explicit) atoms in the graph."""
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        """Return number of covalent bonds in the graph."""
        return len(self.bonds)

    def neighbors(self, atom_idx: int) -> List[int]:
        """Return sorted list of neighbor atom indices for ``atom_idx``."""
        return sorted(self.adjacency.get(atom_idx, {}).keys())

    def degree(self, atom_idx: int) -> int:
        """Return heavy-atom graph degree of ``atom_idx``."""
        return len(self.adjacency.get(atom_idx, {}))

    def get_bond(self, u: int, v: int) -> Optional[Bond]:
        """Return the Bond between atoms ``u`` and ``v``, or None if absent."""
        b_idx = self.adjacency.get(u, {}).get(v)
        return self.bonds[b_idx] if b_idx is not None else None

    def bond_order_sum(self, atom_idx: int) -> float:
        """Return sum of covalent bond orders incident to ``atom_idx``."""
        return sum(self.bonds[b_idx].order for b_idx in self.adjacency.get(atom_idx, {}).values())

    def clone(self) -> "MoleculeGraph":
        """Create a deep structural copy of the molecular graph."""
        atoms_copy = [
            Atom(
                idx=a.idx,
                symbol=a.symbol,
                atomic_number=a.atomic_number,
                formal_charge=a.formal_charge,
                isotope=a.isotope,
                is_aromatic=a.is_aromatic,
                explicit_h=a.explicit_h,
                implicit_h=a.implicit_h,
                chiral_tag=a.chiral_tag,
                hybridization=a.hybridization,
                in_ring=a.in_ring,
                ring_sizes=a.ring_sizes,
                partial_charge=a.partial_charge,
            )
            for a in self.atoms
        ]
        bonds_copy = [
            Bond(
                idx=b.idx,
                atom_a=b.atom_a,
                atom_b=b.atom_b,
                order=b.order,
                is_aromatic=b.is_aromatic,
                is_conjugated=b.is_conjugated,
                in_ring=b.in_ring,
                stereo=b.stereo,
            )
            for b in self.bonds
        ]
        adj_copy: Dict[int, Dict[int, int]] = {
            k: dict(v) for k, v in self.adjacency.items()
        }
        return MoleculeGraph(
            smiles=self.smiles,
            atoms=atoms_copy,
            bonds=bonds_copy,
            adjacency=adj_copy,
            rings=[tuple(r) for r in self.rings],
            canonical_smiles=self.canonical_smiles,
            name=self.name,
            metadata=dict(self.metadata),
        )


@dataclass
class DescriptorProfile:
    """Physicochemical, topological, and drug-likeness descriptor bundle."""

    canonical_smiles: str
    molecular_formula: str
    molecular_weight: float
    exact_mass: float
    heavy_atom_count: int
    ring_count: int
    aromatic_ring_count: int
    rotatable_bonds: int
    hbd: int
    hba: int
    tpsa: float
    clogp: float
    molar_refractivity: float
    fraction_csp3: float
    qed: float
    sa_score: float
    lipinski_violations: int
    veber_pass: bool
    cns_mpo: float
    wiener_index: int
    zagreb_m1: int
    zagreb_m2: int
    randic_chi: float
    balaban_j: float
    bertz_complexity: float
    alerts: Tuple[str, ...] = ()
    gasteiger_charges: Tuple[float, ...] = ()


@dataclass
class FingerprintVector:
    """Circular or topological molecular fingerprint with bit provenance."""

    kind: str
    radius: int
    nbits: int
    bits: NDArray[np.int32]
    counts: NDArray[np.int32]
    bit_info: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)


@dataclass
class ClusterResult:
    """Taylor-Butina sphere-exclusion clustering output."""

    threshold: float
    clusters: List[List[int]]
    centroids: List[int]
    assignments: List[int]
    gini_diversity: float


@dataclass
class MatchedMolecularPair:
    """Hussain-Rea matched molecular pair sharing a Murcko/acyclic core."""

    core_smiles: str
    from_idx: int
    to_idx: int
    from_fragment: str
    to_fragment: str
    delta_properties: Dict[str, float]


@dataclass
class Conformer3D:
    """3D molecular conformer embedded via distance geometry and UFF refinement."""

    molecule: MoleculeGraph
    coords: NDArray[np.float64]
    uff_energy: float
    strain_components: Dict[str, float]
    converged: bool
    rmsd_to_initial: float = 0.0


@dataclass(frozen=True)
class ReceptorAtom:
    """Typed receptor pocket atom for empirical protein-ligand scoring."""

    idx: int
    residue_name: str
    residue_id: int
    atom_name: str
    element: str
    coords: Tuple[float, float, float]
    vdw_radius: float
    is_hbd: bool = False
    is_hba: bool = False
    is_hydrophobic: bool = False
    is_aromatic: bool = False
    formal_charge: float = 0.0


@dataclass
class ReceptorPocket:
    """Binding pocket containing typed receptor atoms and grid bounds."""

    name: str
    atoms: List[ReceptorAtom]
    center: Tuple[float, float, float]
    box_size: Tuple[float, float, float]


@dataclass
class DockingPose:
    """Scored 3D ligand pose inside a receptor pocket."""

    pose_rank: int
    coords: NDArray[np.float64]
    affinity_kcal: float
    predicted_kd_um: float
    predicted_pki: float
    gauss1: float
    gauss2: float
    repulsion: float
    hydrophobic: float
    hydrogen_bond: float
    electrostatic: float
    torsion_penalty: float
    hbonds_formed: int
    rmsd_from_best: float = 0.0


@dataclass
class ADMEPrediction:
    """Multitask GNN ADME/Tox prediction with epistemic uncertainty and attribution."""

    smiles: str
    log_s: float
    log_d_74: float
    cl_int_ul_min_mg: float
    papp_caco2: float
    log_bb: float
    herg_pic50: float
    uncertainties: Dict[str, float]
    atom_attributions: Dict[str, List[float]]
    som_scores: List[float] = field(default_factory=list)


@dataclass
class PKProfile:
    """2-Compartment oral PBPK simulation summary over a dosing interval."""

    dose_mg: float
    tau_hours: float
    times_h: NDArray[np.float64]
    c_plasma_ug_l: NDArray[np.float64]
    c_peripheral_ug_l: NDArray[np.float64]
    cmax_ug_l: float
    tmax_h: float
    auc_0_inf: float
    half_life_beta_h: float
    bioavailability_f: float
    hepatic_cl_l_h: float
    therapeutic_margin: float


@dataclass
class ProtacSpec:
    """Thermodynamic specification of a bivalent PROTAC ternary complex."""

    name: str
    kd_poi_nm: float
    kd_e3_nm: float
    cooperativity_alpha: float
    poi_total_nm: float = 100.0
    e3_total_nm: float = 50.0
    linker_contour_angstrom: float = 18.0
    persistence_length_angstrom: float = 4.5
    required_span_angstrom: float = 12.0


@dataclass
class ProtacCurve:
    """Hook-effect ternary complex dose-response profile."""

    concentrations_nm: NDArray[np.float64]
    ternary_complex_nm: NDArray[np.float64]
    max_ternary_nm: float
    ec50_formation_nm: float
    hook_ic50_nm: float
    effective_alpha: float
    wlc_entropy_penalty_kcal: float


@dataclass
class OptimizationCandidate:
    """De novo generated molecule evaluated along the multi-objective Pareto front."""

    smiles: str
    canonical_smiles: str
    generation: int
    parent_smiles: str
    operation: str
    mpo_score: float
    docking_kcal: float
    qed: float
    sa_score: float
    log_s: float
    cl_int: float
    herg_pic50: float
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    is_valid: bool = True


@dataclass
class OptimizationTrace:
    """Full genealogy and convergence trace from MCTS / NSGA-II lead optimization."""

    seed_smiles: str
    method: str
    total_evaluated: int
    pareto_front: List[OptimizationCandidate]
    history_best_mpo: List[float]
    history_hypervolume: List[float]
    all_candidates: List[OptimizationCandidate]


def get_element(symbol: str) -> ElementData:
    """Look up physical constants for an element symbol, falling back safely."""
    norm = symbol.capitalize() if len(symbol) <= 2 else symbol
    if norm in ELEMENT_TABLE:
        return ELEMENT_TABLE[norm]
    if symbol in ELEMENT_TABLE:
        return ELEMENT_TABLE[symbol]
    raise ValueError(f"Unsupported chemical element symbol: {symbol!r}")
