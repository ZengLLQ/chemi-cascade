"""Vectorized Message-Passing & Graph Attention Neural Network (MPNN-GAT) in NumPy.

Implements an end-to-end differentiable multitask Graph Neural Network for ADME
and toxicity prediction with:
- Edge-conditioned message passing and smooth per-destination graph attention.
- Layer normalization, residual skip connections, and SiLU activations.
- Attentive + mean global graph pooling and 6-task ADME prediction head:
  ``(log_s, log_d_74, cl_int_log10, papp_log10, log_bb, herg_pic50)``.
- Exact analytical backpropagation (verified against central finite differences)
  with AdamW optimization, cosine learning-rate decay, and gradient clipping.
- Monte Carlo Dropout epistemic uncertainty estimation and integrated-gradients
  atom-level attribution.
- Guarded optional PyTorch interop helper (no runtime PyTorch dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from chemi_cascade.featurizer import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, GraphTensor, featurize_molecule
from chemi_cascade.types import ADMEPrediction, MoleculeGraph

ADME_TASKS: Tuple[str, ...] = (
    "log_s",
    "log_d_74",
    "cl_int_log10",
    "papp_log10",
    "log_bb",
    "herg_pic50",
)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _silu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    sig = _sigmoid(x)
    return x * sig


def _silu_grad(x: NDArray[np.float64], grad_out: NDArray[np.float64]) -> NDArray[np.float64]:
    sig = _sigmoid(x)
    return grad_out * (sig + x * sig * (1.0 - sig))


def _layernorm_fwd(
    x: NDArray[np.float64],
    gamma: NDArray[np.float64],
    beta: NDArray[np.float64],
    eps: float = 1e-5,
) -> Tuple[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    out = x_hat * gamma + beta
    return out, (x_hat, inv_std, gamma)


def _layernorm_bwd(
    grad_out: NDArray[np.float64],
    cache: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x_hat, inv_std, gamma = cache
    d_dim = grad_out.shape[1]
    d_gamma = np.sum(grad_out * x_hat, axis=0)
    d_beta = np.sum(grad_out, axis=0)
    dx_hat = grad_out * gamma
    dx = (
        inv_std
        / d_dim
        * (
            d_dim * dx_hat
            - np.sum(dx_hat, axis=1, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=1, keepdims=True)
        )
    )
    return dx, d_gamma, d_beta


def _scatter_softmax_fwd(
    logits: NDArray[np.float64],
    dst: NDArray[np.int32],
    num_nodes: int,
) -> NDArray[np.float64]:
    """Compute softmax grouped by destination node index."""
    if len(logits) == 0:
        return np.zeros(0, dtype=np.float64)
    max_per_node = np.full(num_nodes, -1e9, dtype=np.float64)
    np.maximum.at(max_per_node, dst, logits)
    shifted = np.clip(logits - max_per_node[dst], -40.0, 40.0)
    exp_v = np.exp(shifted)
    sum_per_node = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(sum_per_node, dst, exp_v)
    return exp_v / np.maximum(sum_per_node[dst], 1e-12)


def _scatter_softmax_bwd(
    grad_alpha: NDArray[np.float64],
    alpha: NDArray[np.float64],
    dst: NDArray[np.int32],
    num_nodes: int,
) -> NDArray[np.float64]:
    """Exact backward Jacobian-vector product of grouped scatter softmax."""
    if len(alpha) == 0:
        return np.zeros(0, dtype=np.float64)
    dot = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(dot, dst, grad_alpha * alpha)
    return alpha * (grad_alpha - dot[dst])


@dataclass
class TrainingHistory:
    """Loss and validation metrics recorded across MPNN training epochs."""

    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    final_lr: float


class MolecularMPNN:
    """Pure-NumPy multitask Message-Passing & Graph Attention Network for ADME."""

    def __init__(
        self,
        hidden_dim: int = 48,
        num_layers: int = 2,
        num_tasks: int = len(ADME_TASKS),
        dropout_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.dropout_rate = dropout_rate
        self.seed = seed

        rng = np.random.default_rng(seed)
        d = hidden_dim
        self.params: Dict[str, NDArray[np.float64]] = {}

        def init_w(fin: int, fout: int, scale: float = 1.0) -> NDArray[np.float64]:
            std = scale * math.sqrt(2.0 / max(1, fin + fout))
            return rng.normal(0.0, std, size=(fin, fout)).astype(np.float64)

        self.params["W_in"] = init_w(NODE_FEATURE_DIM, d)
        self.params["b_in"] = np.zeros(d, dtype=np.float64)
        self.params["W_edge"] = init_w(EDGE_FEATURE_DIM, d)
        self.params["b_edge"] = np.zeros(d, dtype=np.float64)

        for layer in range(num_layers):
            self.params[f"W_msg_{layer}"] = init_w(3 * d, d)
            self.params[f"b_msg_{layer}"] = np.zeros(d, dtype=np.float64)
            self.params[f"W_att_{layer}"] = init_w(3 * d, 1)
            self.params[f"b_att_{layer}"] = np.zeros(1, dtype=np.float64)
            self.params[f"W_upd_{layer}"] = init_w(d, d)
            self.params[f"b_upd_{layer}"] = np.zeros(d, dtype=np.float64)
            self.params[f"ln_gamma_{layer}"] = np.ones(d, dtype=np.float64)
            self.params[f"ln_beta_{layer}"] = np.zeros(d, dtype=np.float64)

        self.params["W_gate"] = init_w(d, 1)
        self.params["b_gate"] = np.zeros(1, dtype=np.float64)
        self.params["W_read1"] = init_w(2 * d, d)
        self.params["b_read1"] = np.zeros(d, dtype=np.float64)
        self.params["W_read2"] = init_w(d, num_tasks)
        self.params["b_read2"] = np.zeros(num_tasks, dtype=np.float64)

        # Target normalization stats (fitted on training labels)
        self.y_mean = np.zeros(num_tasks, dtype=np.float64)
        self.y_std = np.ones(num_tasks, dtype=np.float64)
        self._adam_m: Dict[str, NDArray[np.float64]] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self._adam_v: Dict[str, NDArray[np.float64]] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self._adam_step = 0

    def forward_graph(
        self,
        graph: GraphTensor,
        training: bool = False,
        rng: Optional[np.random.Generator] = None,
        node_features_override: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
        """Run full forward pass on a ``GraphTensor`` and return (prediction, cache)."""
        x = node_features_override if node_features_override is not None else graph.node_features
        n = graph.num_atoms
        d = self.hidden_dim
        if n == 0:
            return np.zeros(self.num_tasks, dtype=np.float64), {}

        p = self.params
        h_pre = x @ p["W_in"] + p["b_in"]
        h = _silu(h_pre)

        has_edges = graph.num_directed_edges > 0
        if has_edges:
            e_pre = graph.edge_features @ p["W_edge"] + p["b_edge"]
            e_emb = _silu(e_pre)
        else:
            e_pre = np.zeros((0, d), dtype=np.float64)
            e_emb = np.zeros((0, d), dtype=np.float64)

        layer_caches: List[Dict[str, Any]] = []
        h_in = h
        for layer in range(self.num_layers):
            h_prev = h
            if has_edges:
                h_src = h_prev[graph.edge_src]
                h_dst = h_prev[graph.edge_dst]
                z = np.concatenate([h_src, h_dst, e_emb], axis=1)
                msg_pre = z @ p[f"W_msg_{layer}"] + p[f"b_msg_{layer}"]
                msg = _silu(msg_pre)

                att_raw = (z @ p[f"W_att_{layer}"] + p[f"b_att_{layer}"]).ravel()
                att_tanh = np.tanh(att_raw) * 3.0
                alpha = _scatter_softmax_fwd(att_tanh, graph.edge_dst, n)

                agg = np.zeros((n, d), dtype=np.float64)
                np.add.at(agg, graph.edge_dst, alpha[:, None] * msg)
            else:
                z = np.zeros((0, 3 * d), dtype=np.float64)
                msg_pre = np.zeros((0, d), dtype=np.float64)
                msg = np.zeros((0, d), dtype=np.float64)
                att_raw = np.zeros(0, dtype=np.float64)
                alpha = np.zeros(0, dtype=np.float64)
                agg = np.zeros((n, d), dtype=np.float64)

            upd = agg @ p[f"W_upd_{layer}"] + p[f"b_upd_{layer}"]
            drop_mask = None
            if training and self.dropout_rate > 0.0 and rng is not None:
                keep_p = 1.0 - self.dropout_rate
                drop_mask = (rng.random(upd.shape) < keep_p).astype(np.float64) / keep_p
                upd = upd * drop_mask

            res = h_prev + upd
            h, ln_cache = _layernorm_fwd(
                res, p[f"ln_gamma_{layer}"], p[f"ln_beta_{layer}"]
            )
            layer_caches.append({
                "h_prev": h_prev,
                "z": z,
                "msg_pre": msg_pre,
                "msg": msg,
                "att_raw": att_raw,
                "alpha": alpha,
                "agg": agg,
                "drop_mask": drop_mask,
                "ln_cache": ln_cache,
            })

        # Global attentive + mean readout
        gate_logits = (h @ p["W_gate"] + p["b_gate"]).ravel()
        gate_shift = np.clip(gate_logits - np.max(gate_logits), -40.0, 40.0)
        gate_exp = np.exp(gate_shift)
        pi = gate_exp / np.maximum(np.sum(gate_exp), 1e-12)

        r_att = np.sum(pi[:, None] * h, axis=0)
        r_mean = np.mean(h, axis=0)
        readout = np.concatenate([r_att, r_mean], axis=0)

        read1_pre = readout @ p["W_read1"] + p["b_read1"]
        read1 = _silu(read1_pre)
        pred_norm = read1 @ p["W_read2"] + p["b_read2"]

        cache: Dict[str, Any] = {
            "graph": graph,
            "x": x,
            "h_pre": h_pre,
            "h_in": h_in,
            "e_pre": e_pre,
            "e_emb": e_emb,
            "layer_caches": layer_caches,
            "h_final": h,
            "pi": pi,
            "readout": readout,
            "read1_pre": read1_pre,
            "read1": read1,
        }
        return pred_norm, cache

    def backward_graph(
        self,
        grad_pred: NDArray[np.float64],
        cache: Dict[str, Any],
    ) -> Tuple[Dict[str, NDArray[np.float64]], NDArray[np.float64]]:
        """Compute exact analytical parameter gradients and input node gradients."""
        grads: Dict[str, NDArray[np.float64]] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        if not cache:
            return grads, np.zeros((0, NODE_FEATURE_DIM), dtype=np.float64)

        p = self.params
        d = self.hidden_dim
        graph: GraphTensor = cache["graph"]
        n = graph.num_atoms
        has_edges = graph.num_directed_edges > 0

        # Readout MLP backward
        read1: NDArray[np.float64] = cache["read1"]
        grads["W_read2"] += np.outer(read1, grad_pred)
        grads["b_read2"] += grad_pred
        d_read1 = grad_pred @ p["W_read2"].T
        d_read1_pre = _silu_grad(cache["read1_pre"], d_read1)

        readout: NDArray[np.float64] = cache["readout"]
        grads["W_read1"] += np.outer(readout, d_read1_pre)
        grads["b_read1"] += d_read1_pre
        d_readout = d_read1_pre @ p["W_read1"].T

        d_r_att = d_readout[:d]
        d_r_mean = d_readout[d:]

        h_final: NDArray[np.float64] = cache["h_final"]
        pi: NDArray[np.float64] = cache["pi"]
        d_h = (pi[:, None] * d_r_att[None, :]) + (d_r_mean[None, :] / float(n))
        d_pi = h_final @ d_r_att
        dot_pi = float(np.sum(d_pi * pi))
        d_gate_logits = pi * (d_pi - dot_pi)
        grads["W_gate"] += h_final.T @ d_gate_logits[:, None]
        grads["b_gate"] += np.array([np.sum(d_gate_logits)], dtype=np.float64)
        d_h += d_gate_logits[:, None] * p["W_gate"].T.ravel()[None, :]

        d_e_emb = np.zeros_like(cache["e_emb"])

        # Backprop through message-passing layers in reverse
        for layer in range(self.num_layers - 1, -1, -1):
            lc = cache["layer_caches"][layer]
            d_res, d_gamma, d_beta = _layernorm_bwd(d_h, lc["ln_cache"])
            grads[f"ln_gamma_{layer}"] += d_gamma
            grads[f"ln_beta_{layer}"] += d_beta

            d_h_prev = d_res.copy()
            d_upd = d_res
            if lc["drop_mask"] is not None:
                d_upd = d_upd * lc["drop_mask"]

            grads[f"W_upd_{layer}"] += lc["agg"].T @ d_upd
            grads[f"b_upd_{layer}"] += np.sum(d_upd, axis=0)
            d_agg = d_upd @ p[f"W_upd_{layer}"].T

            if has_edges:
                alpha: NDArray[np.float64] = lc["alpha"]
                msg: NDArray[np.float64] = lc["msg"]
                d_agg_dst = d_agg[graph.edge_dst]
                d_alpha = np.sum(d_agg_dst * msg, axis=1)
                d_msg = alpha[:, None] * d_agg_dst

                d_att_tanh = _scatter_softmax_bwd(d_alpha, alpha, graph.edge_dst, n)
                att_raw: NDArray[np.float64] = lc["att_raw"]
                d_att_raw = d_att_tanh * (3.0 * (1.0 - np.tanh(att_raw) ** 2))

                z: NDArray[np.float64] = lc["z"]
                grads[f"W_att_{layer}"] += z.T @ d_att_raw[:, None]
                grads[f"b_att_{layer}"] += np.array([np.sum(d_att_raw)], dtype=np.float64)

                d_msg_pre = _silu_grad(lc["msg_pre"], d_msg)
                grads[f"W_msg_{layer}"] += z.T @ d_msg_pre
                grads[f"b_msg_{layer}"] += np.sum(d_msg_pre, axis=0)

                d_z = (
                    d_msg_pre @ p[f"W_msg_{layer}"].T
                    + d_att_raw[:, None] * p[f"W_att_{layer}"].ravel()[None, :]
                )
                np.add.at(d_h_prev, graph.edge_src, d_z[:, :d])
                np.add.at(d_h_prev, graph.edge_dst, d_z[:, d:2 * d])
                d_e_emb += d_z[:, 2 * d:]

            d_h = d_h_prev

        if has_edges:
            d_e_pre = _silu_grad(cache["e_pre"], d_e_emb)
            grads["W_edge"] += graph.edge_features.T @ d_e_pre
            grads["b_edge"] += np.sum(d_e_pre, axis=0)

        d_h_pre = _silu_grad(cache["h_pre"], d_h)
        x: NDArray[np.float64] = cache["x"]
        grads["W_in"] += x.T @ d_h_pre
        grads["b_in"] += np.sum(d_h_pre, axis=0)
        d_x = d_h_pre @ p["W_in"].T
        return grads, d_x

    def _adamw_step(
        self,
        grads: Dict[str, NDArray[np.float64]],
        lr: float,
        weight_decay: float = 1e-4,
        clip_norm: float = 5.0,
    ) -> None:
        """Apply one AdamW update with global L2 gradient-norm clipping."""
        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        norm = math.sqrt(total_sq)
        scale = min(1.0, clip_norm / max(norm, 1e-12))

        self._adam_step += 1
        t = self._adam_step
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - (b1 ** t)
        bc2 = 1.0 - (b2 ** t)

        for name, param in self.params.items():
            g = grads[name] * scale
            if name.startswith("W_"):
                param -= lr * weight_decay * param
            m = self._adam_m[name]
            v = self._adam_v[name]
            m[:] = b1 * m + (1.0 - b1) * g
            v[:] = b2 * v + (1.0 - b2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        graphs: Sequence[GraphTensor],
        targets: NDArray[np.float64],
        val_graphs: Optional[Sequence[GraphTensor]] = None,
        val_targets: Optional[NDArray[np.float64]] = None,
        epochs: int = 40,
        batch_size: int = 8,
        lr: float = 3e-3,
        seed: int = 42,
    ) -> TrainingHistory:
        """Train the multitask MPNN-GAT with mini-batch AdamW and cosine annealing."""
        n = len(graphs)
        if n == 0:
            return TrainingHistory([], [], 0, lr)

        self.y_mean = np.mean(targets, axis=0)
        self.y_std = np.maximum(np.std(targets, axis=0), 1e-4)
        y_norm = (targets - self.y_mean) / self.y_std

        rng = np.random.default_rng(seed)
        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val = float("inf")
        best_epoch = 0
        best_params = {k: v.copy() for k, v in self.params.items()}
        cur_lr = lr

        for ep in range(epochs):
            cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * ep / max(1, epochs)))
            cur_lr = max(cur_lr, lr * 0.05)
            perm = rng.permutation(n)
            ep_loss = 0.0

            for start in range(0, n, batch_size):
                batch_idx = perm[start:start + batch_size]
                acc_grads = {k: np.zeros_like(v) for k, v in self.params.items()}
                b_len = float(len(batch_idx))
                for idx in batch_idx:
                    pred, cache = self.forward_graph(graphs[idx], training=True, rng=rng)
                    diff = pred - y_norm[idx]
                    # Smooth Huber / MSE loss gradient
                    ep_loss += float(np.mean(diff * diff))
                    grad_pred = (2.0 / self.num_tasks) * diff / b_len
                    g_one, _ = self.backward_graph(grad_pred, cache)
                    for k in acc_grads:
                        acc_grads[k] += g_one[k]
                self._adamw_step(acc_grads, lr=cur_lr)

            train_losses.append(ep_loss / n)

            if val_graphs is not None and val_targets is not None and len(val_graphs) > 0:
                v_norm = (val_targets - self.y_mean) / self.y_std
                v_loss = 0.0
                for vg, vt in zip(val_graphs, v_norm):
                    vp, _ = self.forward_graph(vg, training=False)
                    v_loss += float(np.mean((vp - vt) ** 2))
                v_loss /= len(val_graphs)
                val_losses.append(v_loss)
                if v_loss < best_val:
                    best_val = v_loss
                    best_epoch = ep
                    best_params = {k: v.copy() for k, v in self.params.items()}

        if val_losses:
            self.params = best_params

        return TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            final_lr=cur_lr,
        )

    def predict_raw(self, graph: GraphTensor) -> NDArray[np.float64]:
        """Predict denormalized 6-task ADME vector for a single ``GraphTensor``."""
        pred_norm, _ = self.forward_graph(graph, training=False)
        return pred_norm * self.y_std + self.y_mean

    def predict_with_uncertainty(
        self,
        mol: MoleculeGraph,
        mc_samples: int = 12,
        seed: int = 101,
    ) -> ADMEPrediction:
        """Predict ADME endpoints with MC-Dropout uncertainty and integrated gradients."""
        graph = featurize_molecule(mol)
        mean_pred = self.predict_raw(graph)

        rng = np.random.default_rng(seed)
        samples = np.zeros((mc_samples, self.num_tasks), dtype=np.float64)
        for s in range(mc_samples):
            p_norm, _ = self.forward_graph(graph, training=True, rng=rng)
            samples[s] = p_norm * self.y_std + self.y_mean
        stds = np.std(samples, axis=0)

        attributions: Dict[str, List[float]] = {}
        for t_idx in (0, 2, 5):  # log_s, cl_int_log10, herg_pic50
            attributions[ADME_TASKS[t_idx]] = self.compute_atom_attribution(graph, t_idx)

        # CYP soft-spot heuristic modulated by GNN clearance attribution + atom electronics
        som = _compute_som_soft_spots(mol, attributions.get("cl_int_log10", []))

        cl_int = float(10.0 ** np.clip(mean_pred[2], -1.0, 4.0))
        papp = float(10.0 ** np.clip(mean_pred[3], -1.5, 2.5))
        return ADMEPrediction(
            smiles=mol.canonical_smiles or mol.smiles,
            log_s=round(float(mean_pred[0]), 3),
            log_d_74=round(float(mean_pred[1]), 3),
            cl_int_ul_min_mg=round(cl_int, 3),
            papp_caco2=round(papp, 3),
            log_bb=round(float(mean_pred[4]), 3),
            herg_pic50=round(float(mean_pred[5]), 3),
            uncertainties={
                task: round(float(stds[i]), 4)
                for i, task in enumerate(ADME_TASKS)
            },
            atom_attributions=attributions,
            som_scores=som,
        )

    def compute_atom_attribution(
        self,
        graph: GraphTensor,
        task_index: int,
        steps: int = 10,
    ) -> List[float]:
        """Compute Integrated Gradients attribution per heavy atom for ``task_index``."""
        if graph.num_atoms == 0:
            return []
        x = graph.node_features
        baseline = np.zeros_like(x)
        acc_grad = np.zeros_like(x)
        unit = np.zeros(self.num_tasks, dtype=np.float64)
        unit[task_index] = float(self.y_std[task_index])

        for s in range(1, steps + 1):
            alpha = s / float(steps)
            x_interp = baseline + alpha * (x - baseline)
            _, cache = self.forward_graph(
                graph, training=False, node_features_override=x_interp
            )
            _, d_x = self.backward_graph(unit, cache)
            acc_grad += d_x

        ig = (x - baseline) * (acc_grad / float(steps))
        atom_scores = np.sum(ig, axis=1)
        return [round(float(v), 5) for v in atom_scores]

    def verify_gradients(
        self,
        graph: GraphTensor,
        eps: float = 1e-5,
        seed: int = 7,
    ) -> float:
        """Compare analytical backpropagation against central finite differences.

        Returns the maximum relative gradient error across sampled parameters.
        """
        rng = np.random.default_rng(seed)
        target = rng.normal(0.0, 1.0, size=self.num_tasks)

        def loss_fn() -> float:
            pred, _ = self.forward_graph(graph, training=False)
            return float(np.mean((pred - target) ** 2))

        pred, cache = self.forward_graph(graph, training=False)
        grad_pred = (2.0 / self.num_tasks) * (pred - target)
        ana_grads, _ = self.backward_graph(grad_pred, cache)

        max_rel_err = 0.0
        for name, arr in self.params.items():
            flat_p = arr.ravel()
            flat_g = ana_grads[name].ravel()
            check_count = min(4, len(flat_p))
            indices = rng.choice(len(flat_p), size=check_count, replace=False)
            for idx in indices:
                orig = flat_p[idx]
                flat_p[idx] = orig + eps
                lp = loss_fn()
                flat_p[idx] = orig - eps
                lm = loss_fn()
                flat_p[idx] = orig
                num_g = (lp - lm) / (2.0 * eps)
                ana_g = float(flat_g[idx])
                denom = max(1e-7, abs(num_g) + abs(ana_g))
                rel_err = abs(num_g - ana_g) / denom
                if abs(num_g) > 1e-6 or abs(ana_g) > 1e-6:
                    max_rel_err = max(max_rel_err, rel_err)
        return max_rel_err


def _compute_som_soft_spots(mol: MoleculeGraph, cl_attr: Sequence[float]) -> List[float]:
    """Score CYP450 metabolic site-of-metabolism susceptibility in [0, 1] per atom."""
    scores: List[float] = []
    for i, a in enumerate(mol.atoms):
        s = 0.1
        nbs = [mol.atoms[nb] for nb in mol.neighbors(i)]
        if a.symbol == "C" and a.hybridization == "sp3":
            if a.total_h > 0:
                s += 0.20
                if any(nb.is_aromatic for nb in nbs):
                    s += 0.35  # benzylic C-H oxidation
                if any(nb.symbol in {"N", "O", "S"} for nb in nbs):
                    s += 0.30  # alpha-heteroatom N/O-dealkylation
        elif a.symbol == "C" and a.is_aromatic and a.total_h > 0:
            s += 0.18
            if abs(a.partial_charge) < 0.08:
                s += 0.12
        elif a.symbol in {"N", "S"} and not a.is_aromatic:
            s += 0.25  # N/S-oxidation
        elif a.symbol in {"F", "Cl"}:
            s = 0.02  # metabolically blocked halogen
        if i < len(cl_attr):
            s += 0.15 * float(np.tanh(abs(cl_attr[i]) * 2.0))
        scores.append(round(float(min(1.0, max(0.0, s))), 4))
    return scores


class OptionalPyTorchBridge:
    """Guarded optional adapter exporting ``MolecularMPNN`` weights when torch exists."""

    @staticmethod
    def is_available() -> bool:
        """Return True if PyTorch is installed in the current environment."""
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def export_state_dict(model: MolecularMPNN) -> Dict[str, Any]:
        """Export model parameters as plain or PyTorch tensor dict."""
        if not OptionalPyTorchBridge.is_available():
            return {k: v.copy() for k, v in model.params.items()}
        import torch  # type: ignore[import-not-found]
        return {k: torch.from_numpy(v.copy()) for k, v in model.params.items()}
