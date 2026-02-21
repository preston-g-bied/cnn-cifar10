import torch
import math

class ConvolutionalLayer(torch.nn.Module):
    def __init__(self, num_channels: int, num_kernels: int, kernel_size: int):
        """
        Args:
            num_channels: Number of input channels
            num_kernels: Number of kernels to learn (P)
            kernel_size: Spatial size of each kernel (MxM)
        """
        super().__init__()
        self.__kernel = torch.nn.Parameter(
            0.01 * (torch.rand(num_kernels, num_channels, kernel_size, kernel_size,
                               dtype=torch.double) - 0.5)
        )

        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

    def setKernel(self, K: torch.Tensor):
        self.__kernel.data = K

    def getKernel(self) -> torch.Tensor:
        return self.__kernel
    
    @staticmethod
    def crossCorrelate3D(kernel: torch.Tensor, data_in: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D cross-correlation.

        Args:
            kernel: Shape (D, M, M) – one kernel with D channel slices
            data_in: Shape (D, H, W) – one input image with D channels

        Returns:
            Shape (H-M+1, W-M+1) – 2D feature map
        """
        num_channels = kernel.shape[0]
        kernel_size = kernel.shape[1]
        out_rows = data_in.shape[1] - kernel_size + 1
        out_cols = data_in.shape[2] - kernel_size + 1

        data_out = torch.zeros((out_rows, out_cols), dtype=torch.double)

        for chan in range(num_channels):
            for r in range(out_rows):
                for c in range(out_cols):
                    data_out[r][c] += torch.sum(
                        data_in[chan, r:r+kernel_size, c:c+kernel_size]
                        * kernel[chan]
                    )

        return data_out
    
    def forward(self, data_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data_in: Shape (N, D, H, W)

        Returns:
            Shape (N, P, H-M+1, W-M+1)
        """
        num_inputs = data_in.shape[0]
        out_rows = data_in.shape[1] - self.kernel_size + 1
        out_cols = data_in.shape[2] - self.kernel_size + 1

        data_out = torch.zeros(
            (num_inputs, self.num_kernels, out_rows, out_cols),
            dtype=torch.double
        )

        for i in range(num_inputs):
            for k in range(self.num_kernels):
                data_out[i][k] = self.crossCorrelate3D(
                    self.__kernel[k], data_in[i]
                )

        return data_out
    
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int):
        super().__init__()
        bound = 2 * math.sqrt(6 / (size_in + size_out))
        self.__weights = torch.nn.Parameter(
            bound * (torch.rand(size_in, size_out, dtype=torch.double) - 0.5)
        )
        self.__biases = torch.nn.Parameter(
            torch.zeros(1, size_out, dtype=torch.double)
        )

    def forward(self, data_in: torch.Tensor) -> torch.Tensor:
        return (data_in @ self.__weights) + self.__biases
    
class ReLULayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_in: torch.Tensor) -> torch.Tensor:
        return torch.clamp(data_in, min=0)
    
class SoftmaxLayer(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, data_in: torch.Tensor) -> torch.Tensor:
        data_in_shifted = data_in - data_in.max(dim=self.dim, keepdim=True).values
        exp_x = torch.exp(data_in_shifted)
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)