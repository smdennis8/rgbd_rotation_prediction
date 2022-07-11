import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import scipy.linalg

"""
Workaround to get matrix log (logm) to work on pytorch and get correct gradients.
It moves to the CPU and back, so it's expensive, but works!
Ref: https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620

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


    Args:
        mata: Rotation Matix. Shape: [B, 3, 3]
        matb: Rotation Matix. Shape: [B, 3, 3]

    Returns:
        torch.float: Geodesic distance

    Reference:
        Mahendran, Siddharth, Haider Ali, and René Vidal. "3d pose regression using convolutional neural networks."
        Proceedings of the IEEE International Conference on Computer Vision Workshops. 2017.
    """
    assert len(mata.shape) == 3 and mata.shape[1] == 3 and matb.shape[2] == 3
    assert len(matb.shape) == 3 and mata.shape[1] == 3 and matb.shape[2] == 3
    dists = 0
    for mata_, matb_ in zip(mata, matb):
        d2 = Logm.apply(mata_ @ matb_.T)
        dist = torch.linalg.norm(d2.flatten(), ord=2) / torch.sqrt(torch.tensor(2))
        dists += dist
    return dists / mata.shape[0]  # div by batch size


def geodesic_dist_opt(m1: torch.Tensor, m2: torch.Tensor):
    """Calculates the geodesic distance between 2 3x3 rotation matrices as the difference in angle between them

    Ref:
        Zhou, Yi, et al. "On the continuity of rotation representations in neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
        Sec 5.1
    """
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.ones(batch, requires_grad=True, device=m.device))
    cos = torch.max(cos, torch.ones(batch, requires_grad=True, device=m.device) * -1)

    theta = torch.acos(cos)  # Shape: [B]

    # theta = torch.min(theta, 2*np.pi - theta)
    return theta


class GeodesicDist(nn.Module):
    """Calculate Geodesic Distance between two batches of 3x3 orthonormal rotation matrices.
     Wrapper around `geodesic_dist_opt`.

    Inputs:
        - inputs: Rotation Matrix. Shape: [B, 3, 3]
        - targets: Rotation Matrix. Shape: [B, 3, 3]

    Outputs:
        - Tensor: loss

    Args:
        reduction: How to reduce the loss:
         - "none": No reduction. Output is of shape: [B].
         - "sum": sum of all losses
         - "mean": Mean of all losses

    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        valid_reductions = ["sum", "mean", "none"]
        if reduction not in valid_reductions:
            raise ValueError(f"Invalid reduction: {self.reduction}. Valid value: {valid_reductions}")

        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss_geo = geodesic_dist_opt(inputs, targets)

        if self.reduction == "sum":
            loss = loss_geo.sum()
        elif self.reduction == "mean":
            loss = loss_geo.mean()
        elif self.reduction == "none":
            loss = loss_geo
        else:
            raise NotImplementedError

        return loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Check the geodesic distance as value increases

    r = R.from_euler("x", 0, degrees=True)
    rot0 = torch.tensor(r.as_matrix()).unsqueeze(0).float()


    distances = []  # List of distances from each loss func
    angles = list(range(0, 360, 5))
    for loss_func in [geodesic_dist, geodesic_dist_opt]:
        dist_list = []
        for angle in angles:
            r = R.from_euler("x", angle, degrees=True)
            rot_mat = r.as_matrix()
            rot_mat = torch.tensor(rot_mat).unsqueeze(0).float()  # Shape: (1, 3, 3)

            dist = loss_func(rot0, rot_mat)  # Input Shape: (1, 3, 3)
            dist_list.append(dist.item())
        distances.append(dist_list)
        # plt.plot(range(0, 360, 5), dist_list)
        # plt.xlabel("Difference in angle")
        # plt.ylabel("Geodesic Distance")
        # plt.show()


    fig, axs = plt.subplots(2)
    fig.suptitle('Geodesic Loss between Two 3x3 Rotation Matrices')

    axs[0].plot(angles, distances[0])
    axs[0].set_title('Using matrix log. \nRef: "3d pose regression using convolutional neural networks."')
    plt.xlabel("Difference in angle")
    plt.ylabel("Geodesic Distance")

    axs[1].plot(angles, distances[1])
    axs[1].set_title('Using Cosine of Angles. \nRef: "On the continuity of rotation representations in neural networks"')
    plt.xlabel("Difference in angle")
    plt.ylabel("Geodesic Distance")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
    plt.show()
