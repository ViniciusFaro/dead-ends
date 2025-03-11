import torch
import torch.nn as nn
import torch.nn.functional as F

class EDLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.00001, sigma, kernel_size):
        super(EDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.cross_entropy = nn.BCELoss()

    def forward(self, pred, label):
        ce = self.cross_entropy(pred, label)
        blured_ref = self.gaussian_blur(pred, self.sigma, self.kernel_size)
        kl = self.kl_divergence(pred, blured_ref) 
        return self.alpha * ce + kl

    def kl_divergence(self, pred, img_blur):
        return self.beta * (((img_blur + 1e-7) / (pred + 1e-7)) - torch.log((img_blur + 1e-7) / (pred + 1e-7)) - 1).sum() / (200 * 200)

    def gaussian_blur(self, batch: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
        
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        
        x = torch.arange(kernel_size, dtype=torch.float, device=batch.device) - (kernel_size - 1) / 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2 + 1e-6))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = torch.outer(gauss_1d, gauss_1d)
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        kernel = gauss_2d.view(1, 1, kernel_size, kernel_size)
        batch_4d = batch
        
        padding = kernel_size // 2
        blurred = F.conv2d(batch_4d, kernel, padding=padding, groups=1)
        
        return blurred
