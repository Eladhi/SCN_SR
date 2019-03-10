import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib
matplotlib.use('agg')
import numpy as np


use_cuda = torch.cuda.is_available()

# original block from the paper - no theta
class SCN(nn.Module):
    def __init__(self, cfg):
        super(SCN, self).__init__()
        my = cfg['my']
        sy = cfg['sy']
        n = cfg['n']
        mx = cfg['sx'] * cfg['sx']
        sg = cfg['sg']
        self.k = cfg['k']
        self.H = nn.Conv2d(1, my, sy, padding=int(sy/2))
        self.W = nn.utils.weight_norm(nn.Linear(my, n, bias=False), dim=1)
        self.S = nn.utils.weight_norm(nn.Linear(n, n, bias=False), dim=1)
        self.Dx = nn.utils.weight_norm(nn.Linear(n, mx, bias=False), dim=1)
        self.G = nn.Conv2d(mx, 1, sg, padding=int(sg/2))
        self.T = nn.Conv2d(1, 1, sy, padding=int(sy/2))
        self.mx = mx

        # initialization values
        # general
        C = 5
        L = 5
        Dy = torch.empty(my, n)
        init.normal_(Dy, 0, 0.1)
        # H - patch extraction
        init.uniform_(self.H.weight, a=-0.1, b=0.1)
        for i in range(int(my/4)):
            row = int(i/(sy-2))
            col = np.mod(i, sy-2)
            self.H.weight[4 * i, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]))
            self.H.weight[4 * i + 1, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]))
            self.H.weight[4 * i + 2, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]))
            self.H.weight[4 * i + 3, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]]))
        # W, S, Dx - linear layers
        self.W.weight.data = torch.t(Dy) * C
        self.S.weight.data = (torch.eye(n) - torch.mm(torch.t(Dy), Dy))
        init.normal_(self.Dx.weight, 0, 0.1)
        self.Dx.weight.data = self.Dx.weight.data / torch.tensor(L * C)
        # G - patch aggregation
        init.constant_(self.G.weight, 1/(sg*sg))
        # T - patch average extraction
        init.constant_(self.T.weight,1/(sy*sy))
        self.T.weight.requires_grad = False

    def forward(self, y):
        means = self.T(y)
        y = y - means
        y = self.H(y)
        b, c, w, h = y.size()
        y = y.view(b, c, w*h)
        y = self.W(y.permute(0,2,1))
        z = y
        for i in range(self.k):
            z = F.softshrink(z, lambd=1)
            z = self.S(z)
            z = z + y
        z = F.softshrink(z, lambd=1)
        z = self.Dx(z)
        z = z.view(b, w, h, self.mx)
        z = self.G(z.permute(0,3,1,2))
        z = z + means
        return z


# original block from the paper - with theta
class SCN_theta(nn.Module):
    def __init__(self, cfg):
        super(SCN_theta, self).__init__()
        my = cfg['my']
        sy = cfg['sy']
        n = cfg['n']
        mx = cfg['sx'] * cfg['sx']
        sg = cfg['sg']
        self.k = cfg['k']
        self.H = nn.Conv2d(1, my, sy, padding=int(sy/2))
        self.W = nn.utils.weight_norm(nn.Linear(my, n, bias=False), dim=1)
        self.S = nn.utils.weight_norm(nn.Linear(n, n, bias=False), dim=1)
        self.Dx = nn.utils.weight_norm(nn.Linear(n, mx, bias=False), dim=1)
        self.G = nn.Conv2d(mx, 1, sg, padding=int(sg/2))
        self.T = nn.Conv2d(1, 1, sy, padding=int(sy/2))
        self.theta = nn.Parameter(torch.ones((1, n)))
        self.mx = mx

        # initialization values
        # general
        C = 5
        L = 5
        Dy = torch.empty(my, n)
        init.normal_(Dy, 0, 0.1)
        # H - patch extraction
        init.uniform_(self.H.weight, a=-0.1, b=0.1)
        for i in range(int(my/4)):
            row = int(i/(sy-2))
            col = np.mod(i, sy-2)
            self.H.weight[4 * i, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]))
            self.H.weight[4 * i + 1, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]))
            self.H.weight[4 * i + 2, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]))
            self.H.weight[4 * i + 3, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]]))
        # W, S, Dx - linear layers
        self.W.weight.data = torch.t(Dy) * C
        self.S.weight.data = (torch.eye(n) - torch.mm(torch.t(Dy), Dy))
        init.normal_(self.Dx.weight, 0, 0.1)
        self.Dx.weight.data = self.Dx.weight.data / torch.tensor(L * C)
        # G - patch aggregation
        init.constant_(self.G.weight, 1/(sg*sg))
        # T - patch average extraction
        init.constant_(self.T.weight,1/(sy*sy))
        self.T.weight.requires_grad = False

    def forward(self, y):
        means = self.T(y)
        y = y - means
        y = self.H(y)
        b, c, w, h = y.size()
        y = y.view(b, c, w*h)
        y = self.W(y.permute(0,2,1))
        y = torch.div(y, self.theta.repeat(b, w * h, 1))
        z = y
        for i in range(self.k):
            z = F.softshrink(z, lambd=1)
            z = torch.mul(z, self.theta.repeat(b, w * h, 1))
            z = self.S(z)
            z = torch.div(z, self.theta.repeat(b, w * h, 1))
            z = z + y
        z = F.softshrink(z, lambd=1)
        z = torch.mul(z, self.theta.repeat(b, w * h, 1))
        z = self.Dx(z)
        z = z.view(b, w, h, self.mx)
        z = self.G(z.permute(0,3,1,2))
        z = z + means
        return z


# with Dy instead of W and S
class SCN_Dy(nn.Module):
    def __init__(self, cfg):
        super(SCN_Dy, self).__init__()
        my = cfg['my']
        sy = cfg['sy']
        n = cfg['n']
        mx = cfg['sx'] * cfg['sx']
        sg = cfg['sg']
        self.k = cfg['k']
        self.H = nn.Conv2d(1, my, sy, padding=int(sy/2))
        self.Dy = nn.utils.weight_norm(nn.Linear(n, my, bias=False), dim=1)
        self.Dx = nn.utils.weight_norm(nn.Linear(n, mx, bias=False), dim=1)
        self.G = nn.Conv2d(mx, 1, sg, padding=int(sg/2))
        self.T = nn.Conv2d(1, 1, sy, padding=int(sy/2))
        self.theta = nn.Parameter(torch.ones((1, n)))
        self.mx = mx
        self.my = my
        self.n = n

        # initialization values
        # general
        self.C = 5
        self.L = 5
        # H - patch extraction
        init.uniform_(self.H.weight, a=-0.1, b=0.1)
        for i in range(int(my/4)):
            row = int(i/(sy-2))
            col = np.mod(i, sy-2)
            self.H.weight[4 * i, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]))
            self.H.weight[4 * i + 1, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]))
            self.H.weight[4 * i + 2, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]))
            self.H.weight[4 * i + 3, :, col:col + 3, row:row + 3].data.copy_(torch.tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]]))
        # Dy, Dx - linear layers
        init.normal_(self.Dy.weight, 0, 0.1)
        init.normal_(self.Dx.weight, 0, 0.1)
        self.Dx.weight.data = self.Dx.weight.data / torch.tensor(self.L * self.C)
        # G - patch aggregation
        init.constant_(self.G.weight, 1/(sg*sg))
        # T - patch average extraction
        init.constant_(self.T.weight,1/(sy*sy))
        self.T.weight.requires_grad = False
        # Create W and S according to Dy
        self.W = nn.Linear(self.my, self.n, bias=False)
        self.W.weight.requires_grad = False
        self.S = nn.Linear(self.n, self.n, bias=False)
        self.S.weight.requires_grad = False

    def forward(self, y):
        #update W and S
        self.W.weight.copy_(torch.t(self.Dy.weight.data) * self.C)
        self.S.weight.copy_((torch.eye(self.n) - torch.mm(torch.t(self.Dy.weight.data), self.Dy.weight.data)))
        # forward pass
        means = self.T(y)
        y = y - means
        y = self.H(y)
        b, c, w, h = y.size()
        y = y.view(b, c, w*h)
        y = self.W(y.permute(0,2,1))
        y = torch.div(y, self.theta.repeat(b, w * h, 1))
        z = y
        for i in range(self.k):
            z = F.softshrink(z, lambd=1)
            z = torch.mul(z, self.theta.repeat(b, w * h, 1))
            z = self.S(z)
            z = torch.div(z, self.theta.repeat(b, w * h, 1))
            z = z + y
        z = F.softshrink(z, lambd=1)
        z = torch.mul(z, self.theta.repeat(b, w * h, 1))
        z = self.Dx(z)
        z = z.view(b, w, h, self.mx)
        z = self.G(z.permute(0,3,1,2))
        z = z + means
        return z
