"""Example 03: Multitask MPNN-GAT ADME Training, Gradient Check & 2-Compartment PBPK."""

from __future__ import annotations

from chemi_cascade.adme_pk import build_calibrated_adme_model, simulate_two_compartment_pk
from chemi_cascade.descriptors import compute_descriptors
from chemi_cascade.featurizer import featurize_molecule
from chemi_cascade.smiles import parse_smiles
from chemi_cascade.visualizer import render_pk_curve


def main() -> None:
    """Train MPNN-GAT, verify analytical gradients, and simulate oral PK."""
    model, metrics = build_calibrated_adme_model(epochs=25, hidden_dim=36, seed=42)
    target = parse_smiles("Cc1ccc(NC(=O)c2ccccc2)cc1Nc1ncccc1", name="Kinase_Lead")
    grad_err = model.verify_gradients(featurize_molecule(target), eps=1e-5, seed=42)
    print(f"Calibrated MPNN RMSE={metrics['overall_rmse']:.4f} | Finite-Diff Grad Err={grad_err:.3e}")

    desc = compute_descriptors(target)
    adme = model.predict_with_uncertainty(target, mc_samples=10, seed=42)
    pk = simulate_two_compartment_pk(adme, mw=desc.molecular_weight, dose_mg=100.0)
    print(render_pk_curve(pk, adme=adme))


if __name__ == "__main__":
    main()
