import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel(size: int, sigma: float, device='cpu') -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_blur(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    
    # Generate Gaussian kernel
    device = input.device
    kernel = gaussian_kernel(kernel_size, sigma, device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    
    kernel = kernel.repeat(input.shape[1], 1, 1, 1)
    
    blurred = F.conv2d(input, kernel, padding=kernel_size // 2, groups=input.shape[1])
    return blurred

def symmetry_measure(matrix):
    diff_norm = torch.norm(matrix - matrix.transpose(-1, -2), p='fro')
    return diff_norm

class CEBSD(nn.Module):
    def __init__(self, phi=0.3, alpha=0.4, beta=0.4, eps=10, tau=1e-6, sigma=15, k=7):
        super(CEBSD, self).__init__()
        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.tau = tau
        self.sigma = sigma
        self.k = k

    def forward(self, pred, mask):
        probs = torch.sigmoid(pred)

        ce1 = F.binary_cross_entropy_with_logits(pred, mask)
        ce2 = F.binary_cross_entropy_with_logits(pred, gaussian_blur(probs, self.k, self.sigma))
        sym = symmetry_measure(torch.matmul(pred, (pred * mask).permute(0, 1, 3, 2)))
        # kl divergence optimiation by Schulman (2020)
        kl = mask.mean() / (pred.mean() + self.tau) - torch.log(mask.mean() / (pred.mean() + self.tau)) - 1
        # log in sym so loss does not scale too fast
        return self.phi*ce1*ce2 + torch.log(sym + torch.e) / self.eps + self.alpha * kl
