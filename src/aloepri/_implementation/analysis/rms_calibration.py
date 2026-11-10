"""用独立样本量化标量RMSNorm近似误差，不能把校准当成严格等价的证明。"""
from __future__ import annotations
import math
import torch


def report_scalar_rms_error(matrix, calibration_inputs, evaluation_inputs, *, epsilon=1e-6):
    """在校准样本估计kappa，在另一批样本比较原始归一化与标量近似。

    matrix为行向量约定下的P[d,D]；输入是原坐标的激活，不是服务器扩维向量。
    原norm的gamma已被融合到后续权重，比较中只计算除以RMS这一步。
    """
    p, train, test = [torch.as_tensor(x).detach().double().cpu() for x in (matrix, calibration_inputs, evaluation_inputs)]
    if p.ndim != 2 or p.shape[1] < p.shape[0] or not p.numel():
        raise ValueError("P must be a nonempty [original_dim, expanded_dim] matrix")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    for x in (train, test):
        if x.ndim < 2 or x.shape[-1]!=p.shape[0] or not x.numel() or not torch.isfinite(x).all():
            raise ValueError("Activation batches must be finite nonempty tensors in the original dimension")
    if not torch.isfinite(p).all() or torch.linalg.matrix_rank(p)!=p.shape[0]:
        raise ValueError("P must be finite and full row rank")
    train, test = train.reshape(-1,p.shape[0]), test.reshape(-1,p.shape[0])
    if not train.square().sum()>0:
        raise ValueError("Calibration samples must include a nonzero vector")
    transformed = train@p
    kappa = torch.sqrt(transformed.square().sum()/train.square().sum())
    coefficient = kappa*math.sqrt(p.shape[0]/p.shape[1])
    eps_prime = coefficient.square()*epsilon
    observed = test@p
    exact = observed*torch.rsqrt(test.square().mean(-1,keepdim=True)+epsilon)
    approx = observed*torch.rsqrt(observed.square().mean(-1,keepdim=True)+eps_prime)*coefficient
    absolute = (approx-exact).norm(dim=-1)
    norm = exact.norm(dim=-1)
    nonzero = norm>1e-12
    relative = absolute[nonzero]/norm[nonzero]
    if not relative.numel():
        raise ValueError("Evaluation samples must include a nonzero vector")
    singular = torch.linalg.svdvals(p)
    return dict(original_dim=p.shape[0], expanded_dim=p.shape[1], calibration_vectors=len(train),
                evaluation_vectors=len(test), kappa=float(kappa), coefficient=float(coefficient),
                epsilon=epsilon, scalar_epsilon=float(eps_prime),
                relative_error_mean=float(relative.mean()), relative_error_p95=float(torch.quantile(relative,.95)),
                relative_error_max=float(relative.max()), absolute_error_max=float(absolute.max()),
                singular_value_min=float(singular.min()), singular_value_max=float(singular.max()),
                condition_number=float(singular.max()/singular.min()),
                scope="Measured on caller-supplied evaluation activations only; a scalar cannot remove directional norm distortion; this is not an equivalence or downstream-accuracy proof")
