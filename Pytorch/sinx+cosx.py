from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch

# 1.准备数据
x = np.linspace(-2*np.pi, 2*np.pi, 400)     # 400个从-2PI到2PI之间的x,以及对应sinx(y)值
y = np.sin(x)+np.cos(x)
X = x.reshape(400, -1)  # 升为二维
Y = y.reshape(400, -1)

dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)      # 数据封装，总共4批


# 2.构建网络：神经网络主要结构，使用简单的线性结构进行拟合
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Linear()全连接层,参数应该是二维张量
        # ReLU()激活函数
        # 常用激活函数：Sigmoid、tanh、ReLU、LeakyReLU、pReLU、ELU、maxout
        # self.net = nn.Sequential(
        #     nn.Linear(in_features=1, out_features=10), nn.Sigmoid(),
        #     nn.Linear(10, 100), nn.Sigmoid(),
        #     nn.Linear(100, 10), nn.Sigmoid(),
        #     nn.Linear(10, 1)
        # )
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=10), nn.ReLU(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, input):     # 前向计算input:torch.FloatTensor
        return self.net(input)


net = Net()
# 定义优化器和损失函数
optim = torch.optim.Adam(Net.parameters(net), lr=0.005)     # lr学习率的设定太大效果会很差
Loss = nn.MSELoss()     # 均方损失函数
# 自适应学习率优化算法：AdaGrad、RMSProp、AdaDelta、Adam


cnt = 500   # 训练次数越大拟合越好
for i in range(cnt):
    loss = None
    for batch_x, batch_y in dataloader:
        y_predict = net(batch_x)
        loss = Loss(y_predict, batch_y)
        optim.zero_grad()   # 梯度置零，把loss关于weight的导数变成0
        loss.backward()     # 反向传播，计算梯度
        optim.step()    # 参数更新
    # 输出拟合情况loss
    if i % 50 == 0:
        print("step: {0} , loss: {1}".format(i, loss.item()))


# 使用训练好的模型进行预测
predict = net(torch.tensor(X, dtype=torch.float))

# 绘图展示预测的和真实数据之间的差异
plt.plot(x, y, label="真实值")
plt.plot(x, predict.detach().numpy(), label="预测值")
plt.title("sin+cosx")
plt.xlabel("x")
plt.ylabel("sin(x)+cos(x)")
plt.legend()
plt.show()
