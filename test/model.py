import torch, math
import torch.nn as nn
from torchvision.models import resnet101
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        resnet = resnet101()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.reduce_conv5 = Con1x1WithBnRelu(2048, 64)
        self.reduce_conv4 = Con1x1WithBnRelu(1024, 64)
        self.reduce_conv3 = Con1x1WithBnRelu(512, 64)
        self.reduce_conv2 = Con1x1WithBnRelu(256, 64)

    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out1_ = self.maxpool(out1)
        out2 = self.layer1(out1_)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        out2 = self.reduce_conv2(out2)
        out3 = self.reduce_conv3(out3)
        out4 = self.reduce_conv4(out4)
        out5_ = self.reduce_conv5(out5)

        return [out2, out3, out4, out5_, out5]

class UpsampleModule(nn.Module):
    def __init__(self, in_ch):
        super(UpsampleModule, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch,
                                  kernel_size=3,
                                  padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input, assign):
        output = F.interpolate(input, [assign[0], assign[1]], mode='bilinear')
        output = self.conv(output)
        return self.relu(self.bn(output))


class Con3x3WithBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Con3x3WithBnRelu, self).__init__()
        self.con1x1 = nn.Conv2d(in_ch, out_ch,
                                kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        return self.relu(self.bn(self.con1x1(input)))


class Con1x1WithBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Con1x1WithBnRelu, self).__init__()
        self.con1x1 = nn.Conv2d(in_ch, out_ch,
                                kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        return self.relu(self.bn(self.con1x1(input)))



class KeyValue_Q(torch.nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue_Q, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)

class KeyValue_M(torch.nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue_M, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x).unsqueeze(2), self.value_conv(x).unsqueeze(2)

class fuse(torch.nn.Module):
    def __init__(self, indim_h, indim_l):
        super(fuse, self).__init__()
        self.conv_h = torch.nn.Conv2d(indim_h, 1, kernel_size=1)
        self.conv_l = torch.nn.Conv2d(indim_l, indim_h, kernel_size=3, padding=1, stride=1)
        self.fc = torch.nn.Linear(64, 64)

    def forward(self, l, h):
        # Eq.3 at the paper, h has been upsampled at the previous step
        S = self.conv_h(h)* self.conv_l(l)
        c = self.fc(F.adaptive_avg_pool2d(S, (1, 1)).squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        return S + c * S


class MemoryReader(torch.nn.Module):
    def __init__(self):
        super(MemoryReader, self).__init__()
        self.memory_reduce = Con1x1WithBnRelu(256, 64)

    def forward(self, K_M, V_M, K_Q, V_Q):  # shape: B,C,N,H,W, Eq.2 in the paper.
        B, C_K, N, H, W = K_M.size()
        _, C_V, _, _, _ = V_M.size()

        K_M = K_M.view(B, C_K, N * H * W)
        K_M = torch.transpose(K_M, 1, 2)
        K_Q = K_Q.view(B, C_K, H * W)

        w = torch.bmm(K_M, K_Q)
        w = w / math.sqrt(C_K)
        w = F.softmax(w, dim=1)
        V_M = V_M.view(B, C_V, N * H * W)

        mem = torch.bmm(V_M, w)
        mem = mem.view(B, C_V, H, W)

        E_t = torch.cat([mem, V_Q], dim=1)


        return self.memory_reduce(E_t)


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # D0-->D1
        self.upsample_D0 = UpsampleModule(64)
        self.memory4 = UpsampleModule(64)
        self.fuse_Q4 = fuse(64, 64)
        self.fuse_M4 = fuse(64, 64*2)
        self.D1 = Con3x3WithBnRelu(64*3, 64)

        # D1-->D2
        self.upsample_D1 = UpsampleModule(64)
        self.memory3 = UpsampleModule(64)
        self.fuse_Q3 = fuse(64, 64)
        self.fuse_M3 = fuse(64, 64*2)
        self.D2 = Con3x3WithBnRelu(64*3, 64)

        # D2-->D3
        self.upsample_D2 = UpsampleModule(64)
        self.memory2 = UpsampleModule(64)
        self.fuse_Q2 = fuse(64, 64)
        self.fuse_M2 = fuse(64, 64*2)
        self.D3 = Con3x3WithBnRelu(64*3, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64*3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,1,3,1,1)
        )



    def forward(self, Q_convs,M_convs, memory, H, W):

        Q_conv5, Q_conv4, Q_conv3, Q_conv2 = Q_convs
        M_conv4, M_conv3, M_conv2 = M_convs


        upsample_D0 = self.upsample_D0(Q_conv5, Q_conv4[0, 0].size())
        memory4 = self.memory4(memory, Q_conv4[0, 0].size())
        fuse_Q4 = self.fuse_Q4(Q_conv4, upsample_D0)
        fuse_M4 = self.fuse_M4(M_conv4, memory4)
        D1 = self.D1(torch.cat((upsample_D0, fuse_Q4, fuse_M4), dim=1))

        upsample_D1 = self.upsample_D1(D1, Q_conv3[0, 0].size())
        memory3 = self.memory3(memory4, Q_conv3[0, 0].size())
        fuse_Q3 = self.fuse_Q3(Q_conv3, upsample_D1)
        fuse_M3 = self.fuse_M3(M_conv3, memory3)
        D2 = self.D2(torch.cat((upsample_D1, fuse_Q3, fuse_M3), dim=1))

        upsample_D2 = self.upsample_D2(D2, Q_conv2[0, 0].size())
        memory2 = self.memory2(memory3, Q_conv2[0, 0].size())
        fuse_Q2 = self.fuse_Q2(Q_conv2, upsample_D2)
        fuse_M2 = self.fuse_M2(M_conv2, memory2)
        D3 = self.D3(torch.cat((upsample_D2, fuse_Q2, fuse_M2), dim=1))

        up3 = F.interpolate(D3, size=[H,W], mode="bilinear")
        up2 = F.interpolate(D2, size=[H,W], mode="bilinear")
        up1 = F.interpolate(D1, size=[H,W], mode="bilinear")

        D1_D2_D3 = torch.cat([up3,up2,up1],dim=1)
        pred = self.final_conv(D1_D2_D3)

        return pred


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.encoder = ResNet101()
        self.kv_memory_r5 = KeyValue_M(2048, keydim=64, valdim=128)
        self.kv_query_r5 = KeyValue_Q(2048, keydim=64, valdim=128)
        self.memory = MemoryReader()
        self.decoder = Decoder()


    def forward(self, x):
        B, N, C, input_H, input_W = x.size()
        x = x.view(B*N,C,input_H,input_W)
        features = self.encoder(x)
        for i in range(len(features)):
            _, C, H, W = features[i].size()
            features[i] = features[i].view(B,N,C,H,W)
        # features list [conv2, conv3, conv4, conv5, conv5_ori]
        current_memory_frames = [[t - 1 if t - 1 >= 0 else 0, t + 1 if t + 1 <= N - 1 else N - 1] for t in range(N)]
        preds = []
        for query_t in range(N):
            K_Q, V_Q = self.kv_query_r5(features[4][:,query_t])
            Q_convs = [features[j][:,query_t] for j in range(3,-1,-1)]
            for i, memory_t in enumerate(current_memory_frames[query_t]):
                memory_t_key, memory_t_value = self.kv_memory_r5(features[4][:,memory_t])
                memory_convs = [features[j][:,memory_t] for j in range(2,-1,-1)]
                if i == 0:
                    K_M, V_M = memory_t_key, memory_t_value
                    M_convs = memory_convs
                else:
                    K_M = torch.cat([K_M, memory_t_key], dim=2)
                    V_M = torch.cat([V_M, memory_t_value], dim=2)
                    for j in range(len(M_convs)):
                        M_convs[j] = torch.cat([M_convs[j],memory_convs[j]], dim=1)

            memory = self.memory(K_M, V_M, K_Q, V_Q)

            pred= self.decoder(Q_convs, M_convs, memory, H=input_H, W=input_W)
            preds.append(pred)

        preds = [torch.sigmoid(mask.unsqueeze(1)) for mask in preds]
        preds = torch.cat(preds, dim=1)

        return preds


