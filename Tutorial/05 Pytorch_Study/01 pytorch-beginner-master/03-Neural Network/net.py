from torch import nn
import torch.nn.functional as F
# 基本的网络构建类模板
class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
        # 可以添加各种网络层
        self.conv1 = nn.Conv2d(3, 10, 3)
        # 具体每种层的参数可以去查看文档
        
    def forward(self, x):
        # 定义向前传播
        out = self.conv1(x)
        return out