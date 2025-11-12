import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck
from mmcv.ops import DeformConv2d  
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from functools import partial
import math


class GumbelSparseGate(nn.Module):
    """
    GumbelSparseGate module: A differentiable gating mechanism for sparse expert selection.

    Function:
    - Dynamically predicts the importance (weights) of each expert based on input features (e.g., F4).
    - Uses Gumbel-Softmax to achieve sparse selection while maintaining gradient flow.
    - Fuses multiple expert outputs by weighted summation to produce the final output.

    """

    def __init__(self, in_channels, num_experts, temperature=1.0):

        super().__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),                    # [B, C, 1, 1] -> [B, C]
            nn.Linear(in_channels, num_experts)  # [B, C] -> [B, num_experts]
        )

        self.temperature = temperature  

    def forward(self, feat, expert_outputs, hard=False):

        logits = self.fc(feat)  # [B, num_experts]

        gate_weights = F.gumbel_softmax(logits, tau=self.temperature, hard=hard)  # [B, num_experts]

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, D]
        gated_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # [B, D]

        return gated_output
        #return gated_output, gate_weights
class ProjectionHead(nn.Module):
    def __init__(self, in_channels, proj_dim=128, hidden_dim=512):
        """
        :param in_channels: Number of input channels (e.g., 256, 512, 1024)
        :param proj_dim: Output embedding dimension used for contrastive learning 
        :param hidden_dim: Hidden layer dimension in the projection head
        """
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
  
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
    
            nn.AdaptiveAvgPool2d(1),  # [B, hidden_dim, 1, 1]
            nn.Flatten(),             # [B, hidden_dim]
         
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim)  
        )

    def forward(self, x):
        return self.net(x)  # shape: [B, proj_dim]
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='leakyrelu', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs

class MSCB(nn.Module):
    """
    Multi-Scale Convolution Block (MSCB):
    Expands channels, applies depthwise convolutions with different kernel sizes (MSDC),
    and then compresses channels to extract multi-scale features.
    """

    def __init__(self, in_channels, out_channels, stride,
                 kernel_sizes=[1, 3, 5], expansion_factor=2,
                 dw_parallel=True, add=True, activation='leakyrelu'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels                      
        self.out_channels = out_channels                   
        self.stride = stride                             
        self.kernel_sizes = kernel_sizes                
        self.expansion_factor = expansion_factor        
        self.dw_parallel = dw_parallel                    
        self.add = add                                    
        self.activation = activation                        
        self.n_scales = len(self.kernel_sizes)           

        assert self.stride in [1, 2]

        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # Pointwise 1x1
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )

        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride,
                         self.activation, dw_parallel=self.dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)

        msdc_outs = self.msdc(pout1) 

        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)

        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))

        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out  
        else:
            return out  #
        
# Multi-scale Convolution Block (MSCB) 
def MSCBLayer(in_channels, out_channels, n=1, stride=1,
              kernel_sizes=[1, 3, 5], expansion_factor=2,
              dw_parallel=True, add=True, activation='leakyrelu'):
    """
    Create a sequence of multiple MSCB modules (an MSCB layer).
    
    Args:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.
    - n: Number of stacked MSCB modules.
    - stride: Stride of the first module (stride=2 can be used for downsampling).
    - kernel_sizes: List of kernel sizes for multi-scale convolutions, e.g., [1, 3, 5].
    - expansion_factor: Channel expansion factor.
    - dw_parallel: Whether to apply multi-scale depthwise convolutions in parallel 
                   (True for parallel, False for sequential with residual connection).
    - add: Fusion mode for multi-scale results; True for additive fusion, 
           False for channel concatenation.
    - activation: Type of activation function, e.g., 'relu6'.
    
    """
    convs = []

    mscb = MSCB(
        in_channels, out_channels, stride,
        kernel_sizes=kernel_sizes,
        expansion_factor=expansion_factor,
        dw_parallel=dw_parallel,
        add=add,
        activation=activation
    )
    convs.append(mscb)

    if n > 1:
        for i in range(1, n):
            mscb = MSCB(
                out_channels, out_channels, 1,
                kernel_sizes=kernel_sizes,
                expansion_factor=expansion_factor,
                dw_parallel=dw_parallel,
                add=add,
                activation=activation
            )
            convs.append(mscb)

    conv = nn.Sequential(*convs)
    return conv

class ACE(nn.Module):
    """
    Downsample shallow features to the target number of channels 
    using a combination of depthwise and pointwise convolutions.  
    The number of channels is doubled at each step until reaching out_channels.
    """

    def __init__(self, in_channels, out_channels, down_times=1, activation='leakyrelu'):
        super(ACE, self).__init__()
        layers = []

        mid_channels = [in_channels * (2 ** i) for i in range(down_times + 1)]

        if mid_channels[-1] != out_channels:
            mid_channels[-1] = out_channels

        for i in range(down_times):
            inch = mid_channels[i]
            outch = mid_channels[i + 1]
            
            layers.append(nn.Sequential(
                nn.Conv2d(inch, inch, kernel_size=3, stride=2, padding=1, groups=inch, bias=False),
                nn.BatchNorm2d(inch),
                act_layer(activation, inplace=True),
                nn.Conv2d(inch, outch, kernel_size=1, bias=False),
                nn.BatchNorm2d(outch),
                act_layer(activation, inplace=True)
            ))

        self.blocks = nn.Sequential(*layers)

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.init_weights('normal')

    def init_weights(self, scheme='normal'):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.blocks(x)
        x = self.out_proj(x)
        return x

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='leakyrelu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class GateNet(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GateNet, self).__init__()
        # in_channels = shallow_feat_channels + deep_out_channels
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)  

class AdaptiveLossWeight(nn.Module):
    def __init__(self, feature_dim):
        """
        feature_dim: Feature dimension input to the gate network, 
        representing the intermediate layer output size.
        """
        super(AdaptiveLossWeight, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()  # 0~1
        )

    def forward(self, loss1, loss2, features):

        # shape: [B, 1]
        gate = self.gate_net(features)  # 0~1

        total_loss_per_sample = gate.squeeze(1) * loss1 + (1 - gate.squeeze(1)) * loss2
        total_loss = total_loss_per_sample.mean()

        return total_loss
    
class SEMC_Net(nn.Module):
    def __init__(self, num_experts=3, num_classes=7, use_norm=False):
        super(SEMC_Net, self).__init__()
        self.s = 30 if use_norm else 1
        self.num_experts = num_experts
        self.K=2048
        base = resnet50(pretrained=True)

        # shared stem
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # shared layers
        self.layer1 = base.layer1  # output: 256
        self.layer2 = base.layer2  # output: 512
        self.layer3 = base.layer3  # output: 1024

        expert_dims = [1024, 1024, 1024]
        expert_planes = [512, 512, 512]
        self.experts = nn.ModuleList([
            self._make_layer(Bottleneck, in_planes=dim, planes=plane, blocks=3, stride=2)
            for dim, plane in zip(expert_dims, expert_planes)
        ])

        self.conv_proj_heads = ProjectionHead(in_channels=2048)


        self.edbc = nn.ModuleList([ACE(in_channels=256, out_channels=2048, down_times=3),
                                   ACE(in_channels=512, out_channels=2048, down_times=2),
                                   ACE(in_channels=1024, out_channels=2048, down_times=1)])

        #MSCB
        self.mscb = MSCBLayer(2048, 2048, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, activation='relu6')

        self.gate =GateNet(2048)
     #   self.gate = ChannelGate(4096)
        #CAB
        self.cab = CAB(2048)
        #SAB
        self.sab = SAB()

        self.classifiers = nn.ModuleList([nn.Linear(2048, num_classes),
                                          nn.Linear(2048, num_classes),
                                          nn.Linear(2048, num_classes)
            
        ])

        #  fusion_convs
        self.fusion_convs = nn.ModuleList([
         nn.Conv2d(2048 * 2, 2048, kernel_size=1)
            for _ in range(self.num_experts)
        ])
        
        self.register_buffer("queue", torch.randn(2048, 128))
        self.queue = F.normalize(self.queue, dim=0)  # L2 

        #  supervised contrastive loss
        self.register_buffer("queue_l", torch.randint(0, num_classes, (2048,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _make_layer(self, block, in_planes, planes, blocks, stride):
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion)
        )
        layers = [block(in_planes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """
        Enqueue the current batch of key features and their labels 
        into the queue (as negative samples).
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)  
        assert self.K % batch_size == 0  

        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_l[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, x, labels):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # three layers and three shallow features
        out1 = self.layer1(x)     # (B, 256, H/4, W/4)
        out2 = self.layer2(out1)  # (B, 512, H/8, W/8)
        out3 = self.layer3(out2)  # (B, 1024, H/16, W/16)

        shallow_feats = [out1, out2, out3]
        shared_feats = [out3, out3, out3]  # inputs for each expert

        logits_list = []
        feature_expert = []
        deep_expert = []
        for i in range(self.num_experts):
            deep_out = self.experts[i](shared_feats[i])
            deep_expert.append(deep_out)
            shallow_feat = self.edbc[i](shallow_feats[i])
            
            logits = shallow_feat + deep_out
            #logits = deep_out

            logits =self.cab(logits)*logits
            logits =self.sab(logits)*logits
            logits =self.mscb(logits)
         
        
            proj_feat = self.conv_proj_heads(logits)       # [B, 128]
            proj_feat = F.normalize(proj_feat, dim=1)             
            feature_expert.append(proj_feat)
            logits =F.adaptive_avg_pool2d(logits, 1).view(logits.size(0), -1)
            proj_feat_single=proj_feat
          
            logits =self.classifiers[i](logits)
            logits_list.append(logits)
        #return logits_list
        
        feat_q = feature_expert[0]   # [B, D]
        feat_k1 = feature_expert[1]  # [B, D]
        feat_k2 = feature_expert[2]  # [B, D]

        features = torch.cat([feat_q, feat_k1, feat_k2, self.queue.clone().detach()], dim=0)  # [3B+K, D]

        # supervised part
        logits_q = logits_list[0]  # [B, C]
        logits_k1 = logits_list[1]
        logits_k2 = logits_list[2]
        sup_logits = torch.cat([logits_q, logits_k1, logits_k2], dim=0)  # [3B, C]

        target = labels.view(-1, 1)
        sup_target = labels.view(-1, 1)
        target = torch.cat([target, target, target, self.queue_l.clone().detach().view(-1, 1)], dim=0)  # [3B+K, 1]
        sup_target = torch.cat([sup_target, sup_target, sup_target], dim=0) 

        k_all = torch.cat([feat_k1, feat_k2], dim=0)
        label_all = torch.cat([labels, labels], dim=0)
        self._dequeue_and_enqueue(k_all, label_all)

        logits = torch.cat([logits_q, logits_k1, logits_k2], dim=0)  # [3B, C]
        return features, target, sup_logits, logits_list,torch.stack(deep_expert, dim=0).mean(dim=0),proj_feat_single
        #return features, target, logits, sup_logits,sup_target,logits_list,torch.stack(deep_expert, dim=0).mean(dim=0),proj_feat_single
   


def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params:", total_params)
    print("Total layers:", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))



if __name__ == "__main__":
    moe32 = SEMC_Net()
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    model = SEMC_Net() 
    count_parameters(model)

