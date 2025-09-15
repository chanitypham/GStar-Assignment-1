import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
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
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    # 3. Phase 2: Diagonal Blocks
    q_start = q_block_idx * BLOCK_M

    # Phase 0: Sink blocks (those entirely in [0, SINK_SIZE)) if SINK_SIZE > 0
    if SINK_SIZE > 0:
        k_start = 0
        sink_limit = SINK_SIZE
        while k_start < sink_limit:
            # Load K and V blocks (BLOCK_N, HEAD_DIM)
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            kv_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                      (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            k_block = tl.load(kv_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

            v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                      (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

            # Compute qk^T: (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            qk = tl.dot(q_block, tl.trans(k_block)) * softmax_scale

            # Apply causal + sliding-window + sink mask
            q_abs = q_start + tl.arange(0, BLOCK_M)
            k_abs = k_start + tl.arange(0, BLOCK_N)
            q_abs_b = q_abs[:, None]
            k_abs_b = k_abs[None, :]
            causal_mask = k_abs_b <= q_abs_b
            window_mask = k_abs_b >= (q_abs_b - (WINDOW_SIZE - 1))
            sink_mask = k_abs_b < SINK_SIZE
            full_mask = causal_mask & (window_mask | sink_mask)
            qk = tl.where(full_mask, qk, -float('inf'))

            # Numerically stable online softmax accumulation
            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            qk_shifted_prev = tl.exp2((m_i - m_i_new) * qk_scale)
            qk_shifted_curr = tl.exp2((qk - m_i_new[:, None]) * qk_scale)
            l_i_new = l_i * qk_shifted_prev + tl.sum(qk_shifted_curr, axis=1)
            acc_scale_prev = (l_i * qk_shifted_prev) / tl.where(l_i_new == 0, 1.0, l_i_new)
            acc_scale_curr = 1.0 / tl.where(l_i_new == 0, 1.0, l_i_new)
            v_contrib = tl.dot(qk_shifted_curr.to(v_block.dtype), v_block)
            acc = acc * acc_scale_prev[:, None] + v_contrib * acc_scale_curr[:, None]
            
            # Update accumulators
            m_i = m_i_new
            l_i = l_i_new
            k_start += BLOCK_N

    # Phase 1 & 2: Main sliding window blocks up to current query block (causal)
    q_block_end = q_start + BLOCK_M
    max_k_considered = q_block_end
    min_k_considered = q_block_end - WINDOW_SIZE
    min_k_considered = tl.maximum(0, min_k_considered)
    cur_k = tl.maximum(SINK_SIZE, min_k_considered)
    
    while cur_k < max_k_considered:
        # Load K and V blocks
        k_offsets = cur_k + tl.arange(0, BLOCK_N)
        kv_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                  (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_block = tl.load(kv_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                  (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Compute qk^T
        qk = tl.dot(q_block, tl.trans(k_block)) * softmax_scale

        # Apply causal + sliding-window + sink mask
        q_abs = q_start + tl.arange(0, BLOCK_M)
        k_abs = cur_k + tl.arange(0, BLOCK_N)
        q_abs_b = q_abs[:, None]
        k_abs_b = k_abs[None, :]
        causal_mask = k_abs_b <= q_abs_b
        window_mask = k_abs_b >= (q_abs_b - (WINDOW_SIZE - 1))
        sink_mask = k_abs_b < SINK_SIZE
        full_mask = causal_mask & (window_mask | sink_mask)
        qk = tl.where(full_mask, qk, -float('inf'))

        # Numerically stable online softmax accumulation
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        qk_shifted_prev = tl.exp2((m_i - m_i_new) * qk_scale)
        qk_shifted_curr = tl.exp2((qk - m_i_new[:, None]) * qk_scale)
        l_i_new = l_i * qk_shifted_prev + tl.sum(qk_shifted_curr, axis=1)
        acc_scale_prev = (l_i * qk_shifted_prev) / tl.where(l_i_new == 0, 1.0, l_i_new)
        acc_scale_curr = 1.0 / tl.where(l_i_new == 0, 1.0, l_i_new)
        v_contrib = tl.dot(qk_shifted_curr.to(v_block.dtype), v_block)
        acc = acc * acc_scale_prev[:, None] + v_contrib * acc_scale_curr[:, None]
        
        # Update accumulators
        m_i = m_i_new
        l_i = l_i_new
        cur_k += BLOCK_N
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
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
    return o