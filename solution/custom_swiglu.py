import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(
    GATE,
    UP,
    OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    gate = tl.load(GATE + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(UP + offsets, mask=mask, other=0).to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    result = (gate * sigmoid) * up

    tl.store(OUT + offsets, result, mask=mask)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up must have identical shapes")

    if gate.dtype != torch.float16 or up.dtype != torch.float16:
        raise TypeError("This kernel currently supports float16 only")

    if gate.device.type != "cuda" or up.device.type != "cuda":
        raise ValueError("Inputs must be CUDA tensors")

    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("Inputs must be contiguous")

    output = torch.empty_like(gate)
    n = gate.numel()

    silu_mul_
