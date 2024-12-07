import torch
import flashinfer
import time
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin, position_ids, unsqueeze_dim=1):
    
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)    
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    return q_embed

def layer_norm(
    hidden_states: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):
    b, s, h = hidden_states.shape
    hidden_states = hidden_states.reshape(b * s, h)
    hidden_states = flashinfer.rmsnorm(hidden_states, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h)
    return hidden_states

def topp_temperature_decode(logits, temperature=0.6, top_p=0.9):
    """
    Perform Top-p (nucleus) sampling with temperature decoding in PyTorch.
    
    Args:
        logits (torch.Tensor): Input logits of shape [b, 1, vocab_size].
        temperature (float): Temperature for scaling logits.
        top_p (float): Probability threshold for nucleus sampling.
    
    Returns:
        torch.Tensor: Decoded indices of shape [b, 1].
    """
    # Apply temperature scaling
    logits = logits / temperature

    # Compute probabilities using softmax
    probs = torch.softmax(logits, dim=-1)

    # Sort probabilities and corresponding indices in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask for tokens that exceed the top-p threshold
    mask = cumulative_probs > top_p
    # Shift the mask to preserve at least one token
    mask[:, :, 1:] = mask[:, :, :-1].clone()
    mask[:, :, 0] = False

    # Mask out probabilities beyond the top-p threshold
    sorted_probs.masked_fill_(mask, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # Re-normalize probabilities

    # Sample from the filtered probability distribution
    sampled_indices = torch.multinomial(sorted_probs.squeeze(1), num_samples=1)

    # Map back to original indices
    final_indices = sorted_indices.gather(dim=-1, index=sampled_indices.unsqueeze(-1))

    return final_indices.squeeze(-1)