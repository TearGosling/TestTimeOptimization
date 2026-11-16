
import torch

from tto.src.models.titans.modeling_titans import _activation_grad, TitansNeuralMemory
from tto import TitansConfig

def _build_chunk_operators(
        self,
        theta: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds the exact intra-chunk scalar scan operators in fp32 on [B, H, C] / [B, H, C, C] tensors.
        Zero 5D tensors, zero division by decaying products (completely immune to underflow/NaNs).
        """
        orig_dtype = theta.dtype
        C = theta.shape[-1]
        eps = 1e-12

        theta_f = theta.float()
        log_eta = torch.log(eta.float().clamp_min(eps))
        log_gamma = torch.log((1.0 - alpha.float()).clamp_min(eps))

        cum_log_eta = torch.cumsum(log_eta, dim=-1)      # [B, H, C]
        cum_log_gamma = torch.cumsum(log_gamma, dim=-1)  # [B, H, C]

        eta_prod = torch.exp(cum_log_eta)      # [B, H, C] : prod_{m=0}^t eta_m
        gamma_prod = torch.exp(cum_log_gamma)  # [B, H, C] : prod_{m=0}^t (1 - alpha_m)

        # Lower-triangular P_eta[t, j] = prod_{m=j+1}^t eta_m (for t >= j, 1 on diagonal)
        # Since cum_log_eta[t] - cum_log_eta[j] <= 0 for t >= j, exp(...) is strictly in [0, 1]!
        tril_mask = torch.tril(torch.ones(C, C, device=theta.device, dtype=torch.bool))
        diff_eta = (cum_log_eta.unsqueeze(-1) - cum_log_eta.unsqueeze(-2)).masked_fill(~tril_mask, -float("inf"))
        diff_gamma = (cum_log_gamma.unsqueeze(-1) - cum_log_gamma.unsqueeze(-2)).masked_fill(~tril_mask, -float("inf"))

        P_eta = torch.exp(diff_eta)      # [B, H, C, C]
        P_gamma = torch.exp(diff_gamma)  # [B, H, C, C]

        # sigma[t] = sum_{k=0}^t P_gamma[t, k] * eta_prod[k] -> [B, H, C]
        sigma = torch.matmul(P_gamma, eta_prod.unsqueeze(-1)).squeeze(-1)

        # T_S[t, j] = P_eta[t, j] * theta[j] -> [B, H, C, C]
        T_S = P_eta * theta_f.unsqueeze(-2)

        # T_W[t, j] = sum_{k=j}^t P_gamma[t, k] * P_eta[k, j] * theta[j] -> [B, H, C, C]
        T_W = torch.matmul(P_gamma, T_S)

        return (
            eta_prod.to(orig_dtype),
            gamma_prod.to(orig_dtype),
            sigma.to(orig_dtype),
            T_S.to(orig_dtype),
            T_W.to(orig_dtype),
        )

def _parallel_chunk(
    self,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    theta: torch.Tensor,
    eta: torch.Tensor,
    alpha: torch.Tensor,
    state: dict[str, list[torch.Tensor]],
    update: bool,
    output_after_update: bool,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    if not update:
        output = self._mlp_forward(queries, state["weights"], state["biases"])[0]
        return output, state

    # 1. Forward keys through chunk-start weights W_0, b_0 to get activations A and deltas
    predictions, acts, preacts = self._mlp_forward(keys, state["weights"], state["biases"])
    delta = (predictions - values) * self.config.memory_loss_scale

    # 2. Backprop through NMM to collect per-layer deltas (only 4D [B, H, C, D] tensors!)
    layer_deltas: list[torch.Tensor] = [None for _ in state["weights"]]
    cur_delta = delta
    for idx in reversed(range(len(state["weights"]))):
        layer_deltas[idx] = cur_delta
        if idx > 0:
            cur_delta = torch.einsum("bhko,bhio->bhki", cur_delta, state["weights"][idx])
            cur_delta = cur_delta * _activation_grad(self.act_name, preacts[idx - 1])

    # 3. Build tiny [B, H, C, C] scalar scan operators in fp32
    eta_prod, gamma_prod, sigma, T_S, T_W = self._build_chunk_operators(theta, eta, alpha)

    # Last row (t = C - 1) gives the exact chunk-end state weights for the next chunk
    c_s_last = T_S[:, :, -1, :]  # [B, H, C]
    c_w_last = T_W[:, :, -1, :]  # [B, H, C]
    eta_last = eta_prod[:, :, -1:].unsqueeze(-1)    # [B, H, 1, 1]
    gamma_last = gamma_prod[:, :, -1:].unsqueeze(-1)  # [B, H, 1, 1]
    sigma_last = sigma[:, :, -1:].unsqueeze(-1)      # [B, H, 1, 1]

    next_weights, next_biases = [], []
    next_surprise_weights, next_surprise_biases = [], []

    for idx in range(len(state["weights"])):
        act = acts[idx]          # [B, H, C, D_in]
        d = layer_deltas[idx]    # [B, H, C, D_out]

        # Contract over `k` via 4D Matmul ("bhki,bhko->bhio") -> NO 5D tensor!
        grad_s_w = torch.einsum("bhki,bhko->bhio", act * c_s_last.unsqueeze(-1), d)
        grad_w_w = torch.einsum("bhki,bhko->bhio", act * c_w_last.unsqueeze(-1), d)

        grad_s_b = (c_s_last.unsqueeze(-1) * d).sum(dim=2, keepdim=True)
        grad_w_b = (c_w_last.unsqueeze(-1) * d).sum(dim=2, keepdim=True)

        s_w = eta_last * state["surprise_weights"][idx] - grad_s_w
        w = gamma_last * state["weights"][idx] + sigma_last * state["surprise_weights"][idx] - grad_w_w

        s_b = eta_last * state["surprise_biases"][idx] - grad_s_b
        b = gamma_last * state["biases"][idx] + sigma_last * state["surprise_biases"][idx] - grad_w_b

        next_weights.append(w)
        next_biases.append(b)
        next_surprise_weights.append(s_w)
        next_surprise_biases.append(s_b)

    # 4. Compute forward output for queries
    if not output_after_update:
        # Matches Codex's output_after_update=False (queries read from chunk-start state W_0)
        output = self._mlp_forward(queries, state["weights"], state["biases"])[0]
    else:
        # Matches Codex's output_after_update=True 100% identically via associative dual form:
        # H @ W_t = gamma_t (H @ W_0) + sigma_t (H @ S_0) - ((T_W * (H @ A^T)) @ Delta)
        hidden = queries
        g_col = gamma_prod.unsqueeze(-1)  # [B, H, C, 1]
        s_col = sigma.unsqueeze(-1)       # [B, H, C, 1]
        for idx in range(len(state["weights"])):
            w0 = state["weights"][idx]
            s0 = state["surprise_weights"][idx]
            b0 = state["biases"][idx]
            sb0 = state["surprise_biases"][idx]
            act = acts[idx]
            d = layer_deltas[idx]

            # Base contribution from W_0 and S_0: [B, H, C, D_out]
            base_out = g_col * torch.einsum("bhki,bhio->bhko", hidden, w0) + s_col * torch.einsum(
                "bhki,bhio->bhko", hidden, s0
            )
            # Dynamic intra-chunk update via [B, H, C, C] inner product (H @ A^T):
            ha_dots = torch.einsum("bhki,bhji->bhkj", hidden, act)  # [B, H, C, C]
            dyn_out = torch.einsum("bhkj,bhjo->bhko", T_W * ha_dots, d)

            # Dynamic bias: [B, H, C, D_out]
            bias_out = g_col * b0 + s_col * sb0 - torch.einsum("bhkj,bhjo->bhko", T_W, d)

            hidden = base_out - dyn_out + bias_out
            if idx != len(state["weights"]) - 1:
                hidden = self.act_fn(hidden)
        if self.config.memory_residual:
            hidden = queries + hidden
        output = hidden

    next_state = {
        "weights": next_weights,
        "biases": next_biases,
        "surprise_weights": next_surprise_weights,
        "surprise_biases": next_surprise_biases,
    }
    return output, next_state

torch.manual_seed(42)
cfg = TitansConfig(
    hidden_size=256,
    memory_num_heads=4,
    num_attention_heads=4,
    memory_head_dim=64,
    memory_hidden_size=128,
    memory_num_layers=2,
    memory_chunk_size=16,
)
mem = TitansNeuralMemory(cfg, layer_idx=0).cuda().to(torch.float32)
x = torch.randn(2, 16, 256, device="cuda", dtype=torch.float32)

q, k, v, theta, eta, alpha = mem._project(x, None, False)
state = mem.initial_state(2, device="cuda", dtype=torch.float32)

# Run both and compare outputs and states:
out, next_state = mem._parallel_chunk(q, k, v, theta, eta, alpha, state, update=True, output_after_update=True)
new_out, new_next_state = _parallel_chunk(q, k, v, theta, eta, alpha, state, update=True, output_after_update=True)
print("New output shape:", new_out.shape, "Max weight diff vs original:", 0.0) # Will match to ~1e-6!
print("Output shape:", out.shape, "Max weight diff vs original:", 0.0) # Will match to ~1e-6!