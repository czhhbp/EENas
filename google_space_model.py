import torch
import torch.nn as nn
import math
import random


OPS = {
    'avg_pool_3x3': lambda C, stride, affine: AvgPool2d(C, stride=stride, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: MaxPool2d(C, stride=stride, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class MaxPool2d(nn.Module):
    def __init__(self, C, stride, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.BatchNorm2d(C, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class AvgPool2d(nn.Module):
    def __init__(self, C, stride, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            nn.BatchNorm2d(C, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class Edge(nn.Module):

    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self.ops.append(op)

    def forward(self, x, choice):
        return self.ops[choice](x)


class Node(nn.Module):

    def __init__(self, node_id, C, reduction):
        super().__init__()
        self.node_id = node_id  # id 从2开始，id=0,1分别表示所属cell的两个输入
        self.edges = nn.ModuleList()
        for i in range(self.node_id):
            stride = 2 if reduction and i < 2 else 1
            edge = Edge(C, stride)
            self.edges.append(edge)

    def forward(self, x, choice):
        return sum(self.edges[in_node](x[in_node], edge) for in_node, edge in choice)

class NormalCell(nn.Module):
    NormalCell_gene = []

    def __init__(self, node_num, C_prev_prev, C_prev, C, reduction_prev):
        """
        :param node_num:一个cell中中间操作节点数目，不包含两个固定的输入节点和一个固定的输出节点
        :param C_prev_prev:
        :param C_prev:
        :param C:
        :param reduction_prev:前一个cell是否位Reduction Cell
        """
        super().__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.node_num = node_num
        self.reduction = False
        self.nodes = nn.ModuleList()
        for i in range(self.node_num):
            node = Node(i + 2, C, reduction=self.reduction)
            self.nodes.append(node)

    def forward(self, x0, x1):
        x0 = self.preprocess0(x0)
        x1 = self.preprocess1(x1)
        states = [x0, x1]
        for i in range(self.node_num):
            x = self.nodes[i](states, self.NormalCell_gene[i])
            states.append(x)
        return torch.cat(states[-self.node_num:], dim=1)


class ReductionCell(nn.Module):
    ReductionCell_gene = []

    def __init__(self, node_num, C_prev_prev, C_prev, C):
        super().__init__()
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.node_num = node_num
        self.reduction = True
        self.nodes = nn.ModuleList()
        for i in range(self.node_num):
            node = Node(i + 2, C, reduction=self.reduction)
            self.nodes.append(node)

    def forward(self, x0, x1):
        x0 = self.preprocess0(x0)
        x1 = self.preprocess1(x1)
        states = [x0, x1]
        for i in range(self.node_num):
            x = self.nodes[i](states, self.ReductionCell_gene[i])

            states.append(x)
        return torch.cat(states[-self.node_num:], dim=1)


class Classifier(nn.Module):
    def __init__(self, inp, class_num):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inp, class_num, bias=True)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Model(nn.Module):

    def __init__(self, C=16, class_num=10, cell_num=8, node_num=4, stem_multiplier=3):
        super().__init__()
        self.C = C
        self.class_num = class_num
        self.cell_num = cell_num
        self.node_num = node_num
        self.stem_multiplier = stem_multiplier
        self.reduction_layer = [(cell_num - 2) // 3, 2 * (cell_num - 2) // 3 + 1]
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in  range(cell_num):
            if i not in self.reduction_layer:
                cell = NormalCell(self.node_num, C_prev_prev, C_prev, C_curr, reduction_prev)
                self.cells.append(cell)
                reduction_prev = False
            else:
                C_curr = C_curr * 2
                cell = ReductionCell(self.node_num, C_prev_prev, C_prev, C_curr)
                self.cells.append(cell)
                reduction_prev = True
            C_prev_prev, C_prev = C_prev, C_curr * self.node_num

        self.classifier = Classifier(C_prev, self.class_num)
        self._initialize_weights()

    def forward(self, x):
        s0 = s1 = self.stem(x)
        # i=1
        for cell in self.cells:
            # print('cell', i)
            s0, s1 = s1, cell(s0, s1)
        return self.classifier(s1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def set_gene(self, gene):
        NormalCell.NormalCell_gene = gene[0]
        ReductionCell.ReductionCell_gene = gene[1]













