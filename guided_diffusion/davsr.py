import torch
import torch.nn as nn
import torchvision
import torch.fft
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import warnings

from mmedit.models.common import PixelShufflePack, flow_warp
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv,
    SPyNet,
)

from mmedit.utils import get_root_logger

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d
import imageio
from einops import rearrange

ker_x4 = [
    [
        -6.62296952e-06,
        -1.43531806e-05,
        7.71780906e-05,
        -1.71278414e-04,
        4.48358012e-04,
        -4.35484835e-04,
        6.00204476e-05,
        1.72932487e-04,
        -6.59890880e-04,
        6.63316052e-04,
        -1.29075677e-04,
        -1.32539615e-04,
        6.65061933e-04,
        -6.57583529e-04,
        1.72624437e-04,
        5.85416637e-05,
        -4.35113558e-04,
        4.47460392e-04,
        -1.68691287e-04,
        7.48491948e-05,
        -1.20825425e-05,
        -6.16945181e-06,
        1.40647523e-06,
        -2.46621721e-06,
        -1.89478260e-06,
    ],
    [
        -1.57257091e-05,
        -4.14571550e-05,
        3.42346466e-05,
        -1.73117092e-04,
        2.75364990e-04,
        -3.03023058e-04,
        2.78934094e-05,
        1.25176040e-04,
        -4.78930044e-04,
        3.73299612e-04,
        -1.87901940e-04,
        -1.90068182e-04,
        3.75959906e-04,
        -4.78251721e-04,
        1.18706637e-04,
        3.30086950e-05,
        -3.05971625e-04,
        2.75636732e-04,
        -1.65608712e-04,
        3.10237883e-05,
        -3.31510455e-05,
        -2.40114514e-05,
        -1.54131249e-05,
        -2.07109570e-05,
        -1.25366314e-05,
    ],
    [
        8.79877261e-05,
        4.48250794e-05,
        1.43474914e-04,
        -8.13370716e-05,
        4.46069986e-04,
        -2.51096324e-04,
        1.68041937e-04,
        2.82216643e-04,
        -4.16284049e-04,
        6.57742261e-04,
        5.42777002e-07,
        -3.69401528e-06,
        6.61203521e-04,
        -4.13602858e-04,
        2.84677109e-04,
        1.66339727e-04,
        -2.53320148e-04,
        4.44667967e-04,
        -8.76248087e-05,
        1.30069660e-04,
        5.17768203e-05,
        5.41626141e-05,
        6.42609593e-05,
        4.86184363e-05,
        8.17263572e-05,
    ],
    [
        -1.97389381e-04,
        -1.94303153e-04,
        -1.13850823e-04,
        -3.61691375e-04,
        1.82534612e-04,
        -5.27508499e-04,
        -1.55936519e-04,
        -1.29739608e-04,
        -9.89535823e-04,
        -1.25785678e-04,
        -8.99035716e-04,
        -9.06590605e-04,
        -1.23707752e-04,
        -9.87760257e-04,
        -1.27587555e-04,
        -1.51980901e-04,
        -5.24035189e-04,
        1.87726633e-04,
        -3.42782645e-04,
        -1.10211164e-04,
        -1.84603006e-04,
        -1.53397850e-04,
        -1.49407264e-04,
        -1.39940108e-04,
        -1.75328663e-04,
    ],
    [
        5.53600083e-04,
        3.17559112e-04,
        4.92999156e-04,
        2.56536092e-04,
        9.47497436e-04,
        3.44920816e-04,
        7.36473070e-04,
        5.37106011e-04,
        -9.44029307e-04,
        -7.17143354e-04,
        -2.01520137e-03,
        -2.00945209e-03,
        -7.20593613e-04,
        -9.43417777e-04,
        5.21528418e-04,
        7.37398921e-04,
        3.29374365e-04,
        9.26103967e-04,
        2.47579796e-04,
        4.35025868e-04,
        3.67166678e-04,
        3.02359578e-04,
        3.52910836e-04,
        2.49822013e-04,
        4.81858966e-04,
    ],
    [
        -5.44751005e-04,
        -3.35610297e-04,
        -3.12026648e-04,
        -5.53822261e-04,
        2.57063075e-04,
        -5.57883643e-04,
        -1.78515082e-04,
        -7.74280983e-04,
        -3.02986428e-03,
        -3.41906445e-03,
        -5.31860953e-03,
        -5.32733742e-03,
        -3.41347419e-03,
        -3.02830292e-03,
        -7.65440869e-04,
        -1.71034655e-04,
        -5.48122509e-04,
        2.81811052e-04,
        -5.46014286e-04,
        -2.51284830e-04,
        -3.87486536e-04,
        -2.68345058e-04,
        -3.32601747e-04,
        -2.11314007e-04,
        -4.75095061e-04,
    ],
    [
        7.09585001e-05,
        4.18588024e-05,
        1.67012768e-04,
        -1.25738865e-04,
        7.24959245e-04,
        -1.43978832e-04,
        3.76816170e-04,
        7.09050728e-05,
        -1.63729477e-03,
        -1.36717036e-03,
        -2.86972895e-03,
        -2.86799134e-03,
        -1.36989472e-03,
        -1.63232943e-03,
        6.25513348e-05,
        3.80185200e-04,
        -1.57744900e-04,
        7.22191471e-04,
        -1.45817728e-04,
        1.65092526e-04,
        1.68122333e-05,
        6.79298028e-05,
        4.19494318e-05,
        6.21088911e-05,
        4.45288824e-05,
    ],
    [
        2.14053944e-04,
        1.55249145e-04,
        3.15109006e-04,
        -9.39263991e-05,
        5.47674368e-04,
        -7.34235626e-04,
        9.68308959e-05,
        9.93094640e-04,
        1.74057961e-03,
        5.13418857e-03,
        5.55444276e-03,
        5.54880314e-03,
        5.14267059e-03,
        1.73651369e-03,
        9.93132242e-04,
        1.00239566e-04,
        -7.32561864e-04,
        5.48743410e-04,
        -8.96907441e-05,
        3.12769960e-04,
        1.57679460e-04,
        1.99063375e-04,
        1.75503345e-04,
        1.85952114e-04,
        1.91494139e-04,
    ],
    [
        -8.33386788e-04,
        -5.49032586e-04,
        -5.07539778e-04,
        -1.06966426e-03,
        -1.01934304e-03,
        -3.11078108e-03,
        -1.69244420e-03,
        1.67598017e-03,
        7.88008701e-03,
        1.75587516e-02,
        2.22854838e-02,
        2.22803839e-02,
        1.75612923e-02,
        7.88581278e-03,
        1.67268072e-03,
        -1.68787350e-03,
        -3.11866286e-03,
        -9.99479322e-04,
        -1.08888245e-03,
        -4.38772287e-04,
        -6.55522686e-04,
        -4.78970935e-04,
        -5.76296239e-04,
        -3.98336182e-04,
        -7.76139379e-04,
    ],
    [
        8.47557909e-04,
        4.35938098e-04,
        7.62687647e-04,
        -5.77692408e-05,
        -6.17786020e-04,
        -3.36047029e-03,
        -1.29651721e-03,
        5.17373439e-03,
        1.76013876e-02,
        3.49443704e-02,
        4.45243642e-02,
        4.45200689e-02,
        3.49482298e-02,
        1.75940301e-02,
        5.17483568e-03,
        -1.30213180e-03,
        -3.36014689e-03,
        -6.37859222e-04,
        -3.88037770e-05,
        6.88977481e-04,
        5.45008399e-04,
        4.89238359e-04,
        5.55366103e-04,
        3.87062290e-04,
        7.71155697e-04,
    ],
    [
        -1.69815918e-04,
        -2.18344649e-04,
        -2.21612809e-05,
        -9.38849698e-04,
        -2.00746651e-03,
        -5.37591428e-03,
        -2.87766312e-03,
        5.53244352e-03,
        2.22622342e-02,
        4.45263647e-02,
        5.75122349e-02,
        5.75120337e-02,
        4.45260294e-02,
        2.22679544e-02,
        5.53486682e-03,
        -2.87880329e-03,
        -5.37305139e-03,
        -2.00803089e-03,
        -9.31822578e-04,
        -2.41083799e-05,
        -2.12080122e-04,
        -1.42975681e-04,
        -1.42997713e-04,
        -1.48685474e-04,
        -1.47800747e-04,
    ],
    [
        -1.58008785e-04,
        -2.20213420e-04,
        -2.49124987e-05,
        -9.40598780e-04,
        -2.01905868e-03,
        -5.37448563e-03,
        -2.88040726e-03,
        5.53052593e-03,
        2.22669058e-02,
        4.45207581e-02,
        5.75119779e-02,
        5.75034432e-02,
        4.45257016e-02,
        2.22618692e-02,
        5.52939530e-03,
        -2.88189040e-03,
        -5.37443440e-03,
        -2.00954010e-03,
        -9.36773140e-04,
        -2.21936552e-05,
        -2.14422282e-04,
        -1.45715894e-04,
        -1.54002031e-04,
        -1.48140462e-04,
        -1.54624955e-04,
    ],
    [
        8.41734291e-04,
        4.39786090e-04,
        7.64512457e-04,
        -5.01856593e-05,
        -6.14856894e-04,
        -3.35174706e-03,
        -1.29923481e-03,
        5.18318405e-03,
        1.75996777e-02,
        3.49550471e-02,
        4.45260629e-02,
        4.45347801e-02,
        3.49522084e-02,
        1.76057480e-02,
        5.17884130e-03,
        -1.29504339e-03,
        -3.35397781e-03,
        -6.39995153e-04,
        -2.98883297e-05,
        6.91780238e-04,
        5.46498981e-04,
        4.95493179e-04,
        5.64463960e-04,
        3.90185334e-04,
        7.76677974e-04,
    ],
    [
        -8.44855735e-04,
        -5.48137992e-04,
        -5.17587876e-04,
        -1.06950570e-03,
        -1.03276374e-03,
        -3.11213103e-03,
        -1.69876206e-03,
        1.67016091e-03,
        7.87680782e-03,
        1.75525062e-02,
        2.22780760e-02,
        2.22723745e-02,
        1.75588578e-02,
        7.87489023e-03,
        1.67177862e-03,
        -1.69695402e-03,
        -3.11630010e-03,
        -1.00974995e-03,
        -1.08485261e-03,
        -4.46302700e-04,
        -6.57052209e-04,
        -4.86649194e-04,
        -5.82089007e-04,
        -4.01840021e-04,
        -7.81816256e-04,
    ],
    [
        2.20901216e-04,
        1.54273032e-04,
        3.14960664e-04,
        -8.90729498e-05,
        5.54220926e-04,
        -7.33066408e-04,
        9.89281252e-05,
        9.96466959e-04,
        1.73565838e-03,
        5.14443079e-03,
        5.54698193e-03,
        5.55603346e-03,
        5.14005916e-03,
        1.73831673e-03,
        9.93110589e-04,
        1.04838342e-04,
        -7.32817047e-04,
        5.57456224e-04,
        -1.00352911e-04,
        3.18755949e-04,
        1.53792425e-04,
        2.01663090e-04,
        1.80350034e-04,
        1.85500350e-04,
        1.96015768e-04,
    ],
    [
        6.80312296e-05,
        4.67211248e-05,
        1.67404462e-04,
        -1.26764338e-04,
        7.17099989e-04,
        -1.39176409e-04,
        3.78375495e-04,
        7.70669430e-05,
        -1.63196120e-03,
        -1.36380037e-03,
        -2.86062155e-03,
        -2.86648283e-03,
        -1.36091013e-03,
        -1.63337926e-03,
        7.43566634e-05,
        3.75896314e-04,
        -1.48836014e-04,
        7.13720219e-04,
        -1.29924010e-04,
        1.64013400e-04,
        2.48319393e-05,
        6.64570471e-05,
        4.10560206e-05,
        6.33243035e-05,
        4.05774481e-05,
    ],
    [
        -5.45703631e-04,
        -3.43761698e-04,
        -3.12373304e-04,
        -5.55901264e-04,
        2.58315587e-04,
        -5.63259004e-04,
        -1.89779050e-04,
        -7.85009935e-04,
        -3.04622995e-03,
        -3.42933508e-03,
        -5.33473352e-03,
        -5.33901295e-03,
        -3.41859250e-03,
        -3.04469489e-03,
        -7.76268018e-04,
        -1.82858930e-04,
        -5.51534817e-04,
        2.73532642e-04,
        -5.58369618e-04,
        -2.51840771e-04,
        -3.98721168e-04,
        -2.66345829e-04,
        -3.37610429e-04,
        -2.15057604e-04,
        -4.78444214e-04,
    ],
    [
        5.67275973e-04,
        3.24470515e-04,
        5.02358191e-04,
        2.61400215e-04,
        9.38397250e-04,
        3.65285261e-04,
        7.49175902e-04,
        5.44011069e-04,
        -9.18668928e-04,
        -7.11209315e-04,
        -1.99727667e-03,
        -1.98949291e-03,
        -7.21541233e-04,
        -9.20234655e-04,
        5.39022731e-04,
        7.34888134e-04,
        3.58505407e-04,
        9.16481775e-04,
        2.61462701e-04,
        4.42512159e-04,
        3.73992341e-04,
        3.06948525e-04,
        3.56516335e-04,
        2.54016195e-04,
        4.89348255e-04,
    ],
    [
        -2.13831081e-04,
        -1.93390282e-04,
        -1.25380873e-04,
        -3.53034324e-04,
        1.79654700e-04,
        -5.38106658e-04,
        -1.67460195e-04,
        -1.39865500e-04,
        -1.01759122e-03,
        -1.17336975e-04,
        -9.09379276e-04,
        -9.09572060e-04,
        -1.12089809e-04,
        -1.01564266e-03,
        -1.29799577e-04,
        -1.71492764e-04,
        -5.28148550e-04,
        1.81944168e-04,
        -3.42864310e-04,
        -1.16412935e-04,
        -1.86264180e-04,
        -1.59471107e-04,
        -1.43489378e-04,
        -1.46315157e-04,
        -1.75701032e-04,
    ],
    [
        9.86208252e-05,
        4.58713112e-05,
        1.34083530e-04,
        -7.13232366e-05,
        4.01340396e-04,
        -1.96186505e-04,
        1.66365484e-04,
        2.88085139e-04,
        -3.57542915e-04,
        6.16980193e-04,
        1.15134583e-06,
        7.62918989e-06,
        6.15735538e-04,
        -3.58529476e-04,
        2.83650123e-04,
        1.64830781e-04,
        -1.99025308e-04,
        4.03412967e-04,
        -7.59382310e-05,
        1.28919011e-04,
        5.98033548e-05,
        5.52197635e-05,
        6.18129852e-05,
        4.97701039e-05,
        8.39975401e-05,
    ],
    [
        -1.99451788e-05,
        -3.31935407e-05,
        3.95913230e-05,
        -1.68362312e-04,
        3.14572651e-04,
        -3.49850015e-04,
        1.06234475e-05,
        1.22595913e-04,
        -5.72570891e-04,
        4.58886905e-04,
        -1.84707867e-04,
        -1.90342587e-04,
        4.58817725e-04,
        -5.69112890e-04,
        1.24198428e-04,
        9.29201997e-06,
        -3.51323106e-04,
        3.14522476e-04,
        -1.64944271e-04,
        3.88640401e-05,
        -3.53931464e-05,
        -2.12568339e-05,
        -8.58892872e-06,
        -2.45825486e-05,
        -1.05292356e-05,
    ],
    [
        -9.00222312e-06,
        -1.98616726e-05,
        4.21185614e-05,
        -1.23659498e-04,
        2.51632213e-04,
        -2.23101437e-04,
        5.72576610e-05,
        1.64446668e-04,
        -4.00483987e-04,
        4.06739826e-04,
        -1.18534343e-04,
        -1.12464768e-04,
        4.06901614e-04,
        -4.00473684e-04,
        1.65002872e-04,
        5.49562719e-05,
        -2.20523259e-04,
        2.50615441e-04,
        -1.24957500e-04,
        4.32211928e-05,
        -1.65310466e-05,
        -4.98800637e-06,
        -2.85803117e-06,
        -3.67808548e-06,
        -6.34900243e-06,
    ],
    [
        -1.77956721e-07,
        -1.01589767e-05,
        5.13609484e-05,
        -1.24955608e-04,
        2.97146733e-04,
        -2.79870292e-04,
        3.80024503e-05,
        1.45534897e-04,
        -4.76902613e-04,
        4.59446310e-04,
        -1.26425191e-04,
        -1.23541555e-04,
        4.59672912e-04,
        -4.76756628e-04,
        1.48665858e-04,
        3.31575102e-05,
        -2.75740458e-04,
        2.95688311e-04,
        -1.30191984e-04,
        5.70978700e-05,
        -1.24289973e-05,
        -1.27969145e-06,
        3.63751792e-06,
        6.45288026e-07,
        2.65192944e-06,
    ],
    [
        -2.00507111e-06,
        -1.61474363e-05,
        3.89097513e-05,
        -1.13249087e-04,
        2.08590413e-04,
        -1.84651391e-04,
        5.38416571e-05,
        1.43155528e-04,
        -3.37289792e-04,
        3.32014955e-04,
        -1.22051191e-04,
        -1.14295792e-04,
        3.26063979e-04,
        -3.39099526e-04,
        1.49085303e-04,
        5.05394964e-05,
        -1.89832150e-04,
        2.16426371e-04,
        -1.12529029e-04,
        3.91345893e-05,
        -1.57909017e-05,
        -3.37711867e-06,
        -1.28724935e-06,
        -3.23299014e-06,
        -3.96142786e-06,
    ],
    [
        7.00379701e-07,
        -1.14050572e-05,
        6.83115868e-05,
        -1.42903358e-04,
        3.87923239e-04,
        -3.78625060e-04,
        4.03025588e-05,
        1.51877452e-04,
        -6.00269414e-04,
        5.96299767e-04,
        -1.21738311e-04,
        -1.19786884e-04,
        6.02350628e-04,
        -6.07174530e-04,
        1.57549570e-04,
        3.34923061e-05,
        -3.69006040e-04,
        3.82823084e-04,
        -1.46364197e-04,
        6.91332971e-05,
        -9.03777436e-06,
        -2.87651778e-06,
        2.54614224e-06,
        -4.18355597e-07,
        4.72551346e-06,
    ],
]

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
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


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1,
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """

    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda).
        """

        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1.
t = np.linspace(0.125, 0.875, 7)


def getFlowCoeff(indices, device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda).

    Returns
    -------
        tensor
            coefficients C00, C01, C10, C11.
    """

    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C11 = C00 = -(1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))
    return (
        torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device),
    )


def getWarpCoeff(indices, device):
    """
    Gets coefficients used for calculating final intermediate
    frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda).

    Returns
    -------
        tensor
            coefficients C0 and C1.
    """

    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C0 = 1 - t[ind]
    C1 = t[ind]
    return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(
        device
    ), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device)


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""


class BasicVSRPP(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(
        self,
        img_channels=3,
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        is_low_res_input=True,
        spynet_pretrained=None,
        vsr_pretrained=None,
        cpu_cache_length=100,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.img_channels = img_channels

        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        if isinstance(vsr_pretrained, str):
            load_net = torch.load(vsr_pretrained)
            for k, v in load_net["state_dict"].items():
                if k.startswith("generator."):
                    k = k.replace("generator.", "")
                    load_net[k] = v
                    load_net.pop(k)
            self.load_state_dict(load_net, strict=False)
        elif vsr_pretrained is not None:
            raise TypeError(
                "[vsr_pretrained] should be str or None, "
                f"but got {type(vsr_pretrained)}."
            )

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(
                img_channels, mid_channels, 5
            )
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(img_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deform_groups=16,
                    max_residue_magnitude=max_residue_magnitude,
                )
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks
            )

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5
        )

        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3
        )
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn(
                "Deformable alignment module is not added. "
                "Probably your CUDA is not configured correctly. DCN can only "
                "be used with CUDA enabled. Alignment is skipped now."
            )

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.to(self.input_device)
                feat_prop = feat_prop.to(self.input_device)
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.to(self.input_device)

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.to(self.input_device)

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.to(self.input_device)

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](
                    feat_prop, cond, flow_n1, flow_n2
                )

            # concatenate and residual blocks
            feat = (
                [feat_current]
                + [feats[k][idx] for k in feats if k not in ["spatial", module_name]]
                + [feat_prop]
            )
            if self.cpu_cache:
                feat = [f.to(self.input_device) for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats["spatial"])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.to(self.input_device)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1).permute(0, 2, 1, 3, 4)

    def forward(self, lqs_ab):  # TODO
        # def forward(self, lqs):         #TODO
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        self.input_device = lqs_ab.device
        lqs_ab = lqs_ab.permute(0, 2, 1, 3, 4).contiguous()

        if lqs_ab.shape[2] == 4:
            lqs = lqs_ab[:, :, :-1, :, :]  # TODO
        else:
            lqs = lqs_ab

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic"
            ).view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats["spatial"] = []
            for i in range(0, t):
                if lqs_ab.shape[2] == 4:
                    feat = self.feat_extract(lqs_ab[:, i, :, :, :]).cpu()  # TODO
                else:
                    feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats["spatial"].append(feat)
                torch.cuda.empty_cache()
        else:
            if lqs_ab.shape[2] == 4:
                feats_ = self.feat_extract(lqs_ab.view(n * t, -1, h, w))  # TODO
            else:
                feats_ = self.feat_extract(lqs.view(n * t, -1, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats["spatial"] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            "The height and width of low-res inputs must be at least 64, "
            f"but got {h} and {w}."
        )
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"

                feats[module] = []

                if direction == "backward":
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )


def splits3D(a, sf):
    """split a into sfxsf distinct blocks

    Args:
        a: NxCxTxWxH
        sf: 3x1 split factor

    Returns:
        b: NxCxWxHx(sf0*sf1*sf2)
    """

    b = torch.stack(torch.chunk(a, sf[0], dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf[1], dim=3), dim=5)
    b = torch.cat(torch.chunk(b, sf[2], dim=4), dim=5)

    return b


def p2o(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxTxCxHxW
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:2] + shape).type_as(psf)  # [1, 1, 100, 1, 1]
    otf[:, :, : psf.shape[2], ...].copy_(psf)  # [1, 1, 100, 1, 1]
    # for axis, axis_size in enumerate(psf.shape[2:]):
    otf = torch.roll(otf, -int(psf.shape[2] / 2), dims=2)  # [1, 1, 100, 1, 1]
    otf = torch.fft.fftn(otf, dim=(2))  # [1, 1, 100, 1, 1]
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def ps2ot(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = ps2ot(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:2] + shape).type_as(psf)  # [1, 1, 100, 256, 256]
    otf[:, :, : psf.shape[2], : psf.shape[3], : psf.shape[4]].copy_(
        psf
    )  # [1, 1, 100, 256, 256]
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(
            otf, -int(axis_size / 2), dims=axis + 2
        )  # [1, 1, 100, 256, 256]
    otf = torch.fft.fftn(otf, dim=(-3, -2, -1))  # [1, 1, 100, 256, 256]
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample3D(x, sf=(5, 4, 4)):
    """s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    """
    st = 0
    b, c, t, h, w = x.shape
    z = torch.zeros((b, c, t * sf[0], h * sf[1], w * sf[2])).type_as(x)
    z[:, :, st :: sf[0], st :: sf[1], st :: sf[2]].copy_(x)  #
    return z


def downsample(x, sf=3):
    """s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    """
    st = 0
    return x[..., st::sf, st::sf]


def downsample3D(x, sf=4):
    """s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    """
    st = 0
    return x[:, :, :, st::sf, st::sf]


def compute_flow(lqs, spynet_pretrained, cpu_cache):
    """Compute optical flow using SPyNet for feature alignment.

    Note that if the input is an mirror-extended sequence, 'flows_forward'
    is not needed, since it is equal to 'flows_backward.flip(1)'.

    Args:
        lqs (tensor): Input low quality (LQ) sequence with
            shape (n, t, c, h, w).

    Return:
        tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
            flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for
            backward-time propagation (current to next).
    """

    n, t, c, h, w = lqs.size()
    lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
    lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

    spynet = SPyNet(pretrained=spynet_pretrained)

    flows_backward = spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
    flows_forward = spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

    if cpu_cache:
        flows_backward = flows_backward.cpu()
        flows_forward = flows_forward.cpu()

    return flows_forward, flows_backward


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataNet3D(nn.Module):
    def __init__(self):
        super(DataNet3D, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        FR = FBFy + torch.fft.fftn(alpha * x, dim=(2, 3, 4))  # [1, 3, 100, 256, 256]
        x1 = FB.mul(FR)  # [1, 3, 100, 256, 256]
        if sf == (1, 1, 1):
            FBR = splits3D(x1, sf).squeeze(-1)
            invW = splits3D(F2B, sf).squeeze(-1)
        else:
            FBR = torch.mean(
                splits3D(x1, sf), dim=-1, keepdim=False
            )  # [1, 3, 20, 256, 256]
            invW = torch.mean(
                splits3D(F2B, sf), dim=-1, keepdim=False
            )  # [1, 1, 20, 1, 1]
        invWBR = FBR.div(invW + alpha)  # [1, 3, 20, 256, 256]
        FCBinvWBR = FBC * invWBR.repeat(
            1, 1, sf[0], sf[1], sf[2]
        )  # [1, 3, 100, 256, 256]
        FX = (FR - FCBinvWBR) / alpha  # [1, 3, 100, 256, 256]
        Xest = torch.real(torch.fft.ifftn(FX, dim=(2, 3, 4)))  # [1, 3, 100, 256, 256]

        return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main network
# --------------------------------------------
"""


class DAVSRNet(nn.Module):
    def __init__(
        self,
        n_iter=8,
        h_nc=64,
        img_channels=3,
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        is_low_res_input=True,
        sf=(5, 4, 4),
    ):
        super(DAVSRNet, self).__init__()
        self.d = DataNet3D()
        self.h = HyPaNet(in_nc=3, out_nc=n_iter * 2, channel=h_nc)
        self.n = n_iter

        self.vsr = BasicVSRPP(
            img_channels=img_channels,
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            max_residue_magnitude=max_residue_magnitude,
            is_low_res_input=is_low_res_input,
            spynet_pretrained=None,
            vsr_pretrained=None,
        )

        self.img_channels = img_channels

        self.sf = sf

        # data and slomo
        mean = [0.429, 0.431, 0.397]
        mea0 = [-m for m in mean]
        std = [1] * 3
        self.trans_forward = transforms.Compose(
            [transforms.Normalize(mean=mean, std=std)]
        )
        self.trans_backward = transforms.Compose(
            [transforms.Normalize(mean=mea0, std=std)]
        )
        self.flow = UNet(6, 4)
        self.interp = UNet(20, 5)

    def interpolate_batch(self, frame0, frame1, factor, flow, interp, back_warp):
        # frame0 = torch.stack(frames[:-1])
        # frame1 = torch.stack(frames[1:])

        i0 = frame0
        i1 = frame1
        ix = torch.cat([i0, i1], dim=1)

        flow_out = flow(ix)
        f01 = flow_out[:, :2, :, :]
        f10 = flow_out[:, 2:, :, :]

        frame_buffer = []
        for i in range(1, factor):
            t = i / factor
            temp = -t * (1 - t)
            co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

            ft0 = co_eff[0] * f01 + co_eff[1] * f10
            ft1 = co_eff[2] * f01 + co_eff[3] * f10

            gi0ft0 = back_warp(i0, ft0)
            gi1ft1 = back_warp(i1, ft1)

            iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
            io = interp(iy)

            ft0f = io[:, :2, :, :] + ft0
            ft1f = io[:, 2:4, :, :] + ft1
            vt0 = F.sigmoid(io[:, 4:5, :, :])
            vt1 = 1 - vt0

            gi0ft0f = back_warp(i0, ft0f)
            gi1ft1f = back_warp(i1, ft1f)

            co_eff = [1 - t, t]

            ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / (
                co_eff[0] * vt0 + co_eff[1] * vt1
            )

            frame_buffer.append(ft_p)

        return frame_buffer

    def forward(self, x, pre_pad=None, post_pad=None):
        """
        x: tensor, NxCxTxWxH
        k: tensor, Nx(1,3)Txwxh
        sf: integer, 3
        sigma: tensor, Nx1x1x1x1
        """

        # reset sf by config
        sf = self.sf
        # initialization & pre-calculation

        b, t, c, w, h = x.shape  # [B, 20, 3, 64, 64]
        x = x.permute(0, 2, 1, 3, 4)  # [B, 3, 20, 64, 64]

        k = torch.tensor(ker_x4, device=x.device).repeat(b, 1, 5, 1, 1) / 5
        # k = (torch.ones(b, 1, 5, 25, 25)/5).cuda()  # TODO

        FB = ps2ot(k, (sf[0] * t, sf[1] * w, sf[2] * h))  # [1, 1, 100, 256, 256]
        FBC = torch.conj(FB)  # [1, 1, 100, 256, 256]
        F2B = torch.pow(torch.abs(FB), 2)  # [1, 1, 100, 256, 256]
        STy = upsample3D(x, sf)  # [1, 3, 100, 256, 256]
        FBFy = FBC * torch.fft.fftn(STy, dim=(2, 3, 4))  # [1, 3, 100, 256, 256]

        back_warp = backWarp(h, w, x.device)

        x0 = x.permute(0, 2, 1, 3, 4).view(-1, c, w, h)
        x0 = self.trans_forward(x0).view(b, t, c, w, h)

        frame0 = x0[:, :-1, :, :, :].reshape(-1, c, w, h)
        frame1 = x0[:, 1:, :, :, :].reshape(-1, c, w, h)
        x_inter = self.interpolate_batch(
            frame0, frame1, sf[0], self.flow, self.interp, back_warp
        )
        x_inter = torch.stack(x_inter, dim=1).view(-1, c, w, h)  # [20, 3, 64, 64]
        x_inter = self.trans_backward(x_inter).view(
            b, t - 1, sf[0] - 1, c, w, h
        )  # [b, 4, 5, 3, 64, 64]
        x0 = self.trans_backward(x0.view(-1, c, w, h)).view(b, t, c, w, h)

        out_x = []
        if pre_pad is not None:
            out_x.append(pre_pad.to(x.device))
        else:
            out_x.append(x0[:, 0, :, :, :].unsqueeze(1).repeat(1, 2, 1, 1, 1))
        for i in range(t - 1):
            out_x.append(x0[:, i, :, :, :].unsqueeze(1))
            out_x.append(x_inter[:, i, ...])
        out_x.append(x0[:, -1, :, :, :].unsqueeze(1))
        if post_pad is not None:
            out_x.append(post_pad.to(x.device))
        else:
            out_x.append(x0[:, -1, :, :, :].unsqueeze(1).repeat(1, 2, 1, 1, 1))
        x = torch.cat(out_x, dim=1)  # [b, 25, 3, 64, 64]
        x = nn.functional.interpolate(
            x.view(-1, c, w, h),
            scale_factor=sf[1:],
            mode="bilinear",
            align_corners=True,
        )
        x = x.view(b, t * sf[0], c, w * sf[1], h * sf[2])
        x_init = x
        x = x.permute(0, 2, 1, 3, 4)  # [B, 3, 20, 64, 64]
        # hyper-parameter, alpha & beta
        ab = self.h(
            torch.cat(
                (
                    torch.zeros(b, 1, 1, 1, 1, device=x.device, dtype=x.dtype),
                    torch.tensor(sf[0], device=x.device, dtype=x.dtype).expand(
                        b, 1, 1, 1, 1
                    ),
                    torch.tensor(sf[1], device=x.device, dtype=x.dtype).expand(
                        b, 1, 1, 1, 1
                    ),
                ),
                dim=1,
            )
        )

        # unfolding
        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i : i + 1, ...], sf)
            x = self.vsr(x).to(x.device)

        x = x.permute(0, 2, 1, 3, 4)

        return x
