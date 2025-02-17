import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class SelfAttention(nn.Module):
    def __init__(self, semantic):
        super(SelfAttention, self).__init__()

        # overlapping embedding (query, key, value)
        self.query = nn.Conv2d(in_channels=semantic, out_channels=semantic, kernel_size=3, stride=1, padding=1)
        self.key = nn.Conv2d(in_channels=semantic, out_channels=semantic, kernel_size=3, stride=1, padding=1)
        self.value = nn.Conv2d(in_channels=semantic, out_channels=semantic, kernel_size=3, stride=1, padding=1)

        # normalization constant
        self.normalizer = sqrt(semantic * 4)

        self.flatten = nn.Flatten(2, 3)  # flatten for the attention calculation

    def forward(self, x):
        b, c, h, w = x.size()

        # Apply query, key, and value convolutions
        q = self.flatten(self.query(x))
        k = self.flatten(self.key(x))
        v = self.flatten(self.value(x))

        # Compute scaled dot-product attention
        scaled = torch.bmm(q, k.permute(0, 2, 1)) / self.normalizer

        # Attention output reshaped back into original size
        return torch.bmm(F.softmax(scaled, dim=-1), v).reshape(b, c, h, w)


class DCA(nn.Module):
    def __init__(self, ic, oc):
        super(DCA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=oc)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(oc)
        
        self.attention = SelfAttention(semantic=oc)
        
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.attention(x)
        x = self.relu2(x)
        return x


class ExpansionNet(nn.Module):
    def __init__(self):
        super(ExpansionNet, self).__init__()

        # DCA-Head
        self.dcah = DCA(ic=1, oc=8)

        # Expansion-Head
        self.expansion_transpose = nn.ConvTranspose2d(
                                    in_channels=1,
                                    out_channels=8,
                                    kernel_size=(160, 40),
                                    stride=(1, 60),
                                    padding=(0, 0),
                                    output_padding=(0, 0)
                                )
        self.expansion_attention = SelfAttention(8)

        # encoder
        self.dca1 = DCA(ic=16, oc=32)
        self.dca2 = DCA(ic=32, oc=64)
        self.dca3 = DCA(ic=64, oc=64)
        self.dca4 = DCA(ic=64, oc=64)
        self.dca5 = DCA(ic=64, oc=64)

        # bottleneck
        self.bottom_conv = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.bottom_norm = nn.BatchNorm2d(num_features=64)
        self.unity_conv = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

        # decoder
        self.transpose1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.dca6 = DCA(ic = 128, oc=64)
        
        self.transpose2 = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0,
                                             output_padding=0)
        self.dca7 = DCA(ic=128, oc=64)

        self.transpose3 = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2, padding=0,
                                             output_padding=0)
        self.dca8 = DCA(ic=128, oc=64)

        self.transpose4 = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0,
                                             output_padding=0)
        self.dca9 = DCA(ic=128, oc=64)

        self.dca10 = DCA(ic=64, oc=32)

        self.final_conv = nn.Conv2d(in_channels=32,
                                    out_channels=1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, img, vec):
        img = self.dcah(img)
        expanded = self.relu(self.expansion_attention(self.expansion_transpose(vec)))
        x = torch.cat((img, expanded), dim=1)

        # encoder
        enc1 = self.dca1(x)
        enc2 = self.pool(self.dca2(enc1))
        enc3 = self.pool(self.dca3(enc2))
        enc4 = self.pool(self.dca4(enc3))
        enc5 = self.pool(self.dca5(enc4))

        # bottleneck
        bottom1 = self.relu(self.bottom_norm(self.bottom_conv(enc5)))
        bottom2 = self.relu(self.bottom_norm(self.unity_conv(bottom1)))

        # decoder
        dec1 = self.dca6(torch.cat((enc5, bottom2), dim=1))
        dec2 = self.dca7(torch.cat((enc4, self.transpose1(dec1)), dim=1))
        dec3 = self.dca8(torch.cat((enc3, self.transpose2(dec2)), dim=1))
        dec4 = self.dca9(torch.cat((enc2, self.transpose3(dec3)), dim=1))
        dec5 = self.dca10(torch.cat((enc1, self.transpose4(dec4)), dim=1))

        return self.final_conv(dec5)
