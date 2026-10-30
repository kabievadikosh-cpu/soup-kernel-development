import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/kaggle/working/repo")

from solution.triton_ce import triton_cross_entropy

torch.manual_seed(42)

N = 32
C = 1024

logits = torch.randn(
    N, C,
    device="cuda",
    dtype=torch.float16,
)

targets = torch.randint(
    0, C,
    (N,),
    device="cuda",
)

reference = F.cross_entropy(
    logits.float(),
    targets,
    reduction="none",
)

result = triton_cross_entropy(logits, targets)

print("PyTorch first 5:", reference[:5])
print("Triton first 5:", result[:5])

print("Max absolute error:",
      (reference - result).abs().max().item())

print("All close:",
      torch.allclose(reference, result, atol=1e-5, rtol=1e-5))