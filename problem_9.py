import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program IDs
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # GQA: Map query head to corresponding K/V head
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load query block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    # Determine attention window bounds
    query_start = q_block_idx * BLOCK_M
    query_end = tl.minimum(query_start + BLOCK_M, SEQ_LEN)
    
    # Calculate key range based on sliding window and attention sinks
    max_query_idx = query_end - 1
    
    # Keys that can be attended to: 
    # 1. Sink tokens: [0, SINK_SIZE)
    # 2. Sliding window: [max(0, max_query_idx - WINDOW_SIZE + 1), max_query_idx + 1)
    window_start = tl.maximum(0, max_query_idx - WINDOW_SIZE + 1)
    window_end = max_query_idx + 1
    
    # Process sink tokens first
    if SINK_SIZE > 0:
        sink_end_block = tl.cdiv(SINK_SIZE, BLOCK_N)
        for k_block_idx in range(sink_end_block):
            k_start = k_block_idx * BLOCK_N
            k_end = tl.minimum(k_start + BLOCK_N, SINK_SIZE)
            
            # Load K and V blocks
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                     (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            k_block = tl.load(k_ptrs, mask=k_offsets[:, None] < SINK_SIZE, other=0.0)
            
            v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                     (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SINK_SIZE, other=0.0)
            
            # Compute attention scores
            qk = tl.dot(q_block, tl.trans(k_block))
            qk = qk * softmax_scale
            
            # Apply causal mask
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            qk = tl.where(causal_mask, qk, -float('inf'))
            
            # Update max and normalizer
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(tl.max(qk, axis=1) - m_new)
            l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
            
            # Update accumulator
            acc = acc * alpha[:, None]
            p = tl.exp(qk - m_new[:, None])
            acc += tl.dot(p, v_block)
            
            # Update states
            m_i = m_new
            l_i = l_new

    # Process sliding window tokens
    window_start_block = tl.maximum(window_start // BLOCK_N, SINK_SIZE // BLOCK_N)
    window_end_block = tl.cdiv(window_end, BLOCK_N)
    
    for k_block_idx in range(window_start_block, window_end_block):
        k_start = k_block_idx * BLOCK_N
        k_end = tl.minimum(k_start + BLOCK_N, SEQ_LEN)
        
        # Skip if block is entirely outside attention window
        if k_start >= window_end or k_end <= window_start:
            continue
            
        # Skip sink region (already processed)
        if k_start < SINK_SIZE:
            continue
        
        # Load K and V blocks
        k_offsets = k_start + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_block = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        qk = tl.dot(q_block, tl.trans(k_block))
        qk = qk * softmax_scale
        
        # Apply sliding window + causal mask
        sliding_mask = (k_offsets[None, :] >= window_start) & (k_offsets[None, :] < window_end)
        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        mask = sliding_mask & causal_mask
        qk = tl.where(mask, qk, -float('inf'))
        
        # Update max and normalizer
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(tl.max(qk, axis=1) - m_new)
        l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        
        # Update accumulator
        acc = acc * alpha[:, None]
        p = tl.exp(qk - m_new[:, None])
        acc += tl.dot(p, v_block)
        
        # Update states
        m_i = m_new
        l_i = l_new

    # Normalize output
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + \
             (q_offsets[:, None] * o_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)
    
    # Store max values
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    tl.store(m_ptrs, m_i, mask=q_offsets < SEQ_LEN)

@triton.jit
def _flash_attention_backward_swa_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, dO_ptr, M_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    d_stride_b, d_stride_h, d_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Parameters
    softmax_scale,
    BATCH_SIZE: int,
    N_Q_HEADS: int,
    N_KV_HEADS: int,
    SEQ_LEN: int,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Tile Sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get current thread block info
    kv_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_KV_HEADS
    kv_head_idx = batch_head_idx % N_KV_HEADS
    
    # GQA: Find which query heads use this KV head
    num_groups = N_Q_HEADS // N_KV_HEADS
    q_head_start = kv_head_idx * num_groups
    q_head_end = q_head_start + num_groups
    
    # Initialize gradients
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    
    # Load K and V blocks
    k_offsets = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
             (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    k_block = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
    
    v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
             (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
    
    k_start = kv_block_idx * BLOCK_N
    k_end = tl.minimum(k_start + BLOCK_N, SEQ_LEN)
    
    # Determine which query blocks can attend to this key block
    # For attention sinks
    if k_start < SINK_SIZE:
        # Sink tokens can be attended by all queries
        q_start_block = 0
        q_end_block = tl.cdiv(SEQ_LEN, BLOCK_M)
    else:
        # Sliding window attention
        # Queries that can attend to key k: q in [k, k + WINDOW_SIZE]
        max_k_idx = k_end - 1
        q_start = tl.maximum(0, max_k_idx)
        q_end = tl.minimum(SEQ_LEN, max_k_idx + WINDOW_SIZE)
        q_start_block = q_start // BLOCK_M
        q_end_block = tl.cdiv(q_end, BLOCK_M)
    
    # Iterate over query blocks
    for q_block_idx in range(q_start_block, q_end_block):
        q_start_pos = q_block_idx * BLOCK_M
        q_end_pos = tl.minimum(q_start_pos + BLOCK_M, SEQ_LEN)
        
        # Process each query head in the group
        for q_head_idx in range(q_head_start, q_head_end):
            # Load Q, dO, M, D for this query head
            q_offsets = q_start_pos + tl.arange(0, BLOCK_M)
            q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
                     (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
            
            do_ptrs = dO_ptr + batch_idx * do_stride_b + q_head_idx * do_stride_h + \
                      (q_offsets[:, None] * do_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            do_block = tl.load(do_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
            
            m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
            m_block = tl.load(m_ptrs, mask=q_offsets < SEQ_LEN, other=-float('inf'))
            
            d_ptrs = D_ptr + batch_idx * d_stride_b + q_head_idx * d_stride_h + q_offsets * d_stride_s
            d_block = tl.load(d_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)
            
            # Compute attention scores
            qk = tl.dot(q_block, tl.trans(k_block))
            qk = qk * softmax_scale
            
            # Apply attention mask
            if k_start < SINK_SIZE:
                # Sink attention: causal mask only
                mask = q_offsets[:, None] >= k_offsets[None, :]
            else:
                # Sliding window attention
                window_start = tl.maximum(0, tl.max(q_offsets) - WINDOW_SIZE + 1)
                window_end = tl.max(q_offsets) + 1
                sliding_mask = (k_offsets[None, :] >= window_start) & (k_offsets[None, :] < window_end)
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                mask = sliding_mask & causal_mask
            
            qk = tl.where(mask, qk, -float('inf'))
            
            # Compute softmax probabilities
            p = tl.exp(qk - m_block[:, None])
            
            # Compute dp = dO @ V^T
            dp = tl.dot(do_block, tl.trans(v_block))
            
            # Compute ds = p * (dp - D)
            ds = p * (dp - d_block[:, None])
            ds = ds * softmax_scale
            
            # Accumulate gradients
            dk_acc += tl.dot(tl.trans(ds), q_block)
            dv_acc += tl.dot(tl.trans(p), do_block)
            
            # Compute dQ for this block
            dq_block = tl.dot(ds, k_block)
            
            # Store dQ
            dq_ptrs = dQ_ptr + batch_idx * dq_stride_b + q_head_idx * dq_stride_h + \
                      (q_offsets[:, None] * dq_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            # Load existing dQ and add to it
            existing_dq = tl.load(dq_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
            tl.store(dq_ptrs, existing_dq + dq_block.to(dQ_ptr.dtype.element_ty), 
                    mask=q_offsets[:, None] < SEQ_LEN)
    
    # Store dK and dV
    dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
              (k_offsets[:, None] * dk_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=k_offsets[:, None] < SEQ_LEN)
    
    dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
              (k_offsets[:, None] * dv_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=k_offsets[:, None] < SEQ_LEN)

class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)


        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Compute D = rowsum(dO * O)
        D = torch.sum(do * o, dim=-1, dtype=torch.float32)
        
        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_N), batch * n_kv_heads)
        
        _flash_attention_backward_swa_kernel[grid](
            q, k, v, do, M, D,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            softmax_scale,
            batch,
            n_q_heads,
            n_kv_heads,
            seq_len,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)