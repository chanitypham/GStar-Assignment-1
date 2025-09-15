# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def flash_attention_triton_kernel(
    Q, K, V, O, M,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_ob, stride_oh, stride_ol, stride_od,
    stride_mb, stride_mh, stride_ml,
    B, H, H_KV, L, D,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Get program IDs
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    # Decompose batch and head indices
    off_b = off_bh // H
    off_h = off_bh % H
    
    # GQA mapping: map query head to key-value head
    off_h_kv = off_h // (H // H_KV) if H >= H_KV else off_h
    
    # Offsets for the current block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointer arithmetic for Q
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
    
    # Pointer arithmetic for K, V (using GQA head mapping)
    k_ptrs = K + off_b * stride_kb + off_h_kv * stride_kh + offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd
    v_ptrs = V + off_b * stride_vb + off_h_kv * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd
    
    # Pointer arithmetic for O, M
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
    m_ptrs = M + off_b * stride_mb + off_h * stride_mh + offs_m * stride_ml
    
    # Load Q for this block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < L, other=0.0)
    
    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over K, V blocks
    for start_n in range(0, L, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Calculate which keys are valid for this iteration
        offs_n_curr = start_n + offs_n
        
        # Causal mask: only attend to keys at positions <= query position
        causal_mask = offs_n_curr[None, :] <= offs_m[:, None]
        # Also mask out padding
        valid_mask = (offs_m[:, None] < L) & (offs_n_curr[None, :] < L) & causal_mask
        
        # Load K, V for current block
        k_ptrs_curr = k_ptrs + start_n * stride_kl
        v_ptrs_curr = v_ptrs + start_n * stride_vl
        
        k = tl.load(k_ptrs_curr, mask=offs_n_curr[:, None] < L, other=0.0)
        v = tl.load(v_ptrs_curr, mask=offs_n_curr[:, None] < L, other=0.0)
        
        # Compute attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk = qk * softmax_scale
        
        # Apply causal mask
        qk = tl.where(valid_mask, qk, -float('inf'))
        
        # Online softmax update
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_i_new[:, None]), 1)
        
        # Update accumulator
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # Compute attention weights and update accumulator
        p = tl.exp(qk - m_i_new[:, None])
        acc += tl.dot(p, v) * (beta / l_i_new)[:, None]
        
        # Update trackers
        l_i = l_i_new
        m_i = m_i_new
    
    # Store output and max values
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < L)
    tl.store(m_ptrs, m_i, mask=offs_m < L)

class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        # Call the Triton kernel
        flash_attention_triton_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            batch, n_heads, n_kv_heads, seq_len, head_dim,
            softmax_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # [OPTIONAL BONUS] STUDENT IMPLEMENTATION REQUIRED
        # Implement the Triton backward kernel for GQA from scratch.
        # You should:
        #   1. Precompute delta = sum(dO * O)
        #   2. Recompute attention probabilities P = softmax(QK^T)
        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        #   4. Respect GQA mapping and causal mask
        
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)