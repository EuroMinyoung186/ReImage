import torch
import torch.nn as nn
import torch.nn.functional as F

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class SEAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(SEAttention2D, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // reduction, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se(x) * x
        return x

class BottleneckBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8, stride=1, attention=True):
        super(BottleneckBlock1D, self).__init__()

        # skip connection(지름길) 경로가 필요할 경우(채널 수 변경 or stride != 1)
        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm1d(out_channels)
            )

        # 메인 경로
        self.left = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(out_channels)
        )

        # Squeeze-and-Excitation Attention
        if attention:
            self.attention = SEAttention1D(out_channels, out_channels, reduction=reduction)
        else:
            self.attention = nn.Identity()  # attention을 끄고 싶으면 Identity로 대체

    def forward(self, x):
        identity = x
        out = self.left(x)
        out = self.attention(out)

        if self.change is not None:
            identity = self.change(identity)

        out = out + identity
        out = F.relu(out)
        return out

class ResBlock1D(nn.Module):
    """
    blocks: 이 블록 안에 몇 개의 BottleneckBlock1D(또는 BasicBlock 등)를 쌓을지 결정
    """
    def __init__(self, in_channels, out_channels, blocks=1, block_type="BottleneckBlock1D",
                 reduction=8, stride=1, attention=True):
        super(ResBlock1D, self).__init__()
        layers = []

        # 첫 번째 블록(채널이 바뀌거나 stride가 필요할 수 있으므로)
        if blocks > 0:
            layers.append(eval(block_type)(in_channels, out_channels, reduction, stride, attention=attention))

        # 이후 블록(채널/stride 동일)
        for _ in range(blocks - 1):
            layers.append(eval(block_type)(out_channels, out_channels, reduction, 1, attention=attention))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class BottleneckBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, stride, attention=None):
        super(BottleneckBlock2D, self).__init__()

        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=stride, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        self.attention = SEAttention2D(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        

    def forward(self, x):
        identity = x
        x = self.left(x)
        x = self.attention(x)

        if self.change is not None:
            identity = self.change(identity)

        x += identity
        x = F.relu(x)
        return x


class ResBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=1, block_type="BottleneckBlock2D", reduction=8, stride=1, attention=None):
        super(ResBlock2D, self).__init__()

        layers = [eval(block_type)(in_channels, out_channels, reduction, stride, attention=attention)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = eval(block_type)(out_channels, out_channels, reduction, 1, attention=attention)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
