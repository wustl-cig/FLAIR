import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import normalize


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """

        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x_dtype = x.dtype
        x = F.interpolate(x.to(torch.float32), scale_factor=2, mode="bilinear")
        x = x.to(x_dtype)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x


class SuperSloMo(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.flow_estimator = UNet(6, 4)
        self.interp = UNet(20, 5)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))

    def back_warp(self, img, flow):
        H, W = img.shape[2:]
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        gridX, gridY = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        )
        gridX = torch.tensor(
            gridX, requires_grad=False, device=img.device, dtype=flow.dtype
        )
        gridY = torch.tensor(
            gridY, requires_grad=False, device=img.device, dtype=flow.dtype
        )
        x = gridX.unsqueeze(0).expand_as(u) + u
        y = gridY.unsqueeze(0).expand_as(v) + v
        # range -1 to 1
        x = 2 * (x / W - 0.5)
        y = 2 * (y / H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut

    def forward(self, frame0, frame1, factor=1, return_flow=False):
        i0 = normalize((frame0 + 1) / 2, [0.429, 0.431, 0.397], [1] * 3, True)
        i1 = normalize((frame1 + 1) / 2, [0.429, 0.431, 0.397], [1] * 3, True)
        ix = torch.cat([i0, i1], dim=1)

        flow_out = self.flow_estimator(ix)
        f01 = flow_out[:, :2, :, :]
        f10 = flow_out[:, 2:, :, :]

        frame_buffer = []
        for i in range(1, factor):
            t = i / factor
            temp = -t * (1 - t)
            co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

            ft0 = co_eff[0] * f01 + co_eff[1] * f10
            ft1 = co_eff[2] * f01 + co_eff[3] * f10

            gi0ft0 = self.back_warp(i0, ft0)
            gi1ft1 = self.back_warp(i1, ft1)

            iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
            io = self.interp(iy)

            ft0f = io[:, :2, :, :] + ft0
            ft1f = io[:, 2:4, :, :] + ft1
            vt0 = F.sigmoid(io[:, 4:5, :, :])
            vt1 = 1 - vt0

            gi0ft0f = self.back_warp(i0, ft0f)
            gi1ft1f = self.back_warp(i1, ft1f)

            co_eff = [1 - t, t]

            ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / (
                co_eff[0] * vt0 + co_eff[1] * vt1
            )
            ft_p = normalize(ft_p, [-0.429, -0.431, -0.397], [1] * 3, True) * 2 - 1
            frame_buffer.append(ft_p)
        interp = torch.stack(frame_buffer, dim=1)  # B, T, C, H, W
        if return_flow:
            return interp, f01, f10
        return interp
