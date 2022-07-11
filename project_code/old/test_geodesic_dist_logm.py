

import torch
from scipy.spatial.transform import Rotation as R
import scipy.linalg

"""
Workaround to get matrix log (logm) to work on pytorch and get correct gradients.
It moves to the CPU and back, so it's expensive, but works!
Ref: https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
​
Example: 
A = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(logm, A)
"""


def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)


def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)


class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return logm_scipy(A)
    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)

# logm = Logm.apply

# -----------------------


def geodesic_dist(mata: torch.Tensor, matb: torch.Tensor) -> torch.Tensor:
    """Calculates the geodesic distance between 2 3x3 rotation matrices
​
​
    Args:
        mata: Rotation Matix. Shape: [B, 3, 3]
        matb: Rotation Matix. Shape: [B, 3, 3]
​
    Returns:
        torch.float: Geodesic distance
​
    Reference:
        Mahendran, Siddharth, Haider Ali, and René Vidal. "3d pose regression using convolutional neural networks."
        Proceedings of the IEEE International Conference on Computer Vision Workshops. 2017.
    """
    assert len(mata.shape) == 3 and mata.shape[1] == 3 and matb.shape[2] == 3
    assert len(matb.shape) == 3 and mata.shape[1] == 3 and matb.shape[2] == 3
    dists = 0
    for mata_, matb_ in zip(mata, matb):
        d2 = Logm.apply(mata_ @ matb_.T)
        dist = torch.linalg.norm(d2.flatten(), ord=2) / torch.sqrt(torch.tensor(2).float())
        dists += dist
    return dists / mata.shape[0]  # div by batch size


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Check the geodesic distance as value increases

    r = R.from_euler("x", 0, degrees=True)
    rot0 = torch.tensor(r.as_matrix()).unsqueeze(0)

    dist_list = []
    for angle in range(0, 360, 5):
        r = R.from_euler("x", angle, degrees=True)
        rot_mat = r.as_matrix()
        rot_mat = torch.tensor(rot_mat).unsqueeze(0)  # Shape: (1, 3, 3)

        dist = geodesic_dist(rot0, rot_mat)  # Input Shape: (1, 3, 3)
        dist_list.append(dist.item())

    plt.plot(range(0, 360, 5), dist_list)
    plt.xlabel("Difference in angle")
    plt.ylabel("Geodesic Distance")
    plt.show()