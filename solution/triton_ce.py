import torch
import triton
import triton.language as tl


@triton.jit
def _ce_forward_kernel(
    X,
    Y,
    LOSS,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)

    x = tl.load(
        X + row * N_COLS + col,
        col < N_COLS,
        other=-float("inf"),
    ).to(tl.float32)

    maximum = tl.max(x, axis=0)
    log_sum_exp = maximum + tl.log(
        tl.sum(tl.exp(x - maximum), axis=0)
    )

    target = tl.load(Y + row).to(tl.int32)
    target_logit = tl.load(
        X + row * N_COLS + target
    ).to(tl.float32)

    loss = log_sum_exp - target_logit
    tl.store(LOSS + row, loss)


def triton_cross_entropy(logits, targets):
    assert logits.is_cuda
    assert targets.is_cuda
    assert logits.ndim == 2
    assert targets.ndim == 1
    assert logits.shape[0] == targets.shape[0]

    n_rows, n_cols = logits.shape

    losses = torch.empty(
        (n_rows,),
        device=logits.device,
        dtype=torch.float32,
    )

    block = triton.next_power_of_2(n_cols)

    _ce_forward_kernel[(n_rows,)](
        logits,
        targets,
        losses,
        n_rows,
        n_cols,
        block,
        num_warps=4,
    )

    return losses
