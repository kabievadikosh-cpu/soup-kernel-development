import torch
import triton
import triton.language as tl


@triton.jit
def _partial_lse_kernel(
    X,
    PARTIAL_MAX,
    PARTIAL_SUM,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    col = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(
        X + row * N_COLS + col,
        col < N_COLS,
        other=-float("inf"),
    ).to(tl.float32)

    maximum = tl.max(x, axis=0)
    total = tl.sum(tl.exp(x - maximum), axis=0)

    idx = row * CHUNKS + chunk

    tl.store(PARTIAL_MAX + idx, maximum)
    tl.store(PARTIAL_SUM + idx, total)


@triton.jit
def _final_lse_kernel(
    PARTIAL_MAX,
    PARTIAL_SUM,
    Y,
    X,
    LOSS,
    N_COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    FINAL_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.arange(0, FINAL_BLOCK)

    idx = row * CHUNKS + chunk

    maximum = tl.load(
        PARTIAL_MAX + idx,
        chunk < CHUNKS,
        other=-float("inf"),
    )

    total = tl.load(
        PARTIAL_SUM + idx,
        chunk < CHUNKS,
        other=0.0,
    )

    global_max = tl.max(maximum, axis=0)

    global_sum = tl.sum(
        total * tl.exp(maximum - global_max),
        axis=0,
    )

    log_sum_exp = global_max + tl.log(global_sum)

    target = tl.load(Y + row).to(tl.int32)

    target_logit = tl.load(
        X + row * N_COLS + target
    ).to(tl.float32)

    tl.store(LOSS + row, log_sum_exp - target_logit)


def triton_cross_entropy_split(logits, targets, block=4096):
    assert logits.is_cuda
    assert targets.is_cuda
    assert logits.ndim == 2
    assert targets.ndim == 1
    assert logits.shape[0] == targets.shape[0]

    n_rows, n_cols = logits.shape
    chunks = triton.cdiv(n_cols, block)

    partial_max = torch.empty(
        (n_rows, chunks),
        device=logits.device,
        dtype=torch.float32,
    )

    partial_sum = torch.empty_like(partial_max)

    losses = torch.empty(
        (n_rows,),
        device=logits.device,
        dtype=torch.float32,
    )

    _partial_lse_kernel[(n_rows, chunks)](
        logits,
        partial_max,
        partial_sum,
        n_cols,
        block,
        chunks,
        num_warps=4,
    )

    _final_lse_kernel[(n_rows,)](
        partial_max,
        partial_sum,
        targets,
        logits,
        losses,
        n_cols,
        chunks,
        triton.next_power_of_2(chunks),
        num_warps=4,
    )

    return losses

@triton.jit
def _ce_backward_kernel(
    X,
    Y,
    LSE,
    GRAD,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    col = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = col < N_COLS

    target = tl.load(Y + row).to(tl.int32)
    lse = tl.load(LSE + row)

    x = tl.load(
        X + row * N_COLS + col,
        mask,
        other=0.0,
    ).to(tl.float32)

    grad = tl.exp(x - lse) - (col == target).to(tl.float32)
    grad = grad * SCALE

    grad = tl.where(target == IGNORE_INDEX, 0.0, grad)

    tl.store(
        GRAD + row * N_COLS + col,
        grad,
        mask,
    )


def triton_cross_entropy_backward(
    logits,
    targets,
    lse,
    scale,
    block=4096,
    ignore_index=-100,
):
    n_rows, n_cols = logits.shape

    grad = torch.empty(
        logits.shape,
        device=logits.device,
        dtype=torch.float32,
    )

    chunks = triton.cdiv(n_cols, block)

    _ce_backward_kernel[(n_rows, chunks)](
        logits,
        targets,
        lse,
        grad,
        n_cols,
        block,
        ignore_index,
        scale,
        num_warps=4,
    )

    return grad

class TritonCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, targets):
        n_rows, n_cols = logits.shape

        valid = targets != -100
        valid_count = valid.sum().item()

        if valid_count == 0:
            raise ValueError("All targets are ignored")

        losses = triton_cross_entropy_split(
            logits,
            targets,
        )

        loss = losses[valid].mean()

        lse = torch.logsumexp(
            logits.detach().float(),
            dim=1,
        )

        ctx.save_for_backward(
            logits,
            targets,
            lse,
        )

        ctx.scale = 1.0 / valid_count

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, lse = ctx.saved_tensors

        grad = triton_cross_entropy_backward(
            logits,
            targets,
            lse,
            ctx.scale,
        )

        grad = grad * grad_output

        return grad.to(logits.dtype), None


def triton_cross_entropy_autograd(logits, targets):
    return TritonCrossEntropyFunction.apply(logits, targets)
