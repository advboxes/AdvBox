#coding=utf-8

# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CNN on mnist data using pytorch
"""
from __future__ import print_function
from __future__ import division

from builtins import range
from past.utils import old_div
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data

#查看cuda版本
#cat /usr/local/cuda/version.txt
#当cuda版本为8时
#pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
#pip install torchvision
#


train_data = torchvision.datasets.MNIST(
 './mnist-pytorch/data', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(
 './mnist-pytorch/data', train=False, transform=torchvision.transforms.ToTensor()
)
print("train_data:", train_data.train_data.size())
print("train_labels:", train_data.train_labels.size())
print("test_data:", test_data.test_data.size())


#批大小
batch_size=128
#训练的批次数
epochs=10

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        #return torch.nn.functional.log_softmax(out, dim=1)
        return out


"""
参考了 以梦为马_Sun 的实现
---------------------
作者：以梦为马_Sun
来源：CSDN
原文：https: // blog.csdn.net / sunqiande88 / article / details / 80089941?utm_source = copy

"""

def main():

    # 自适应使用GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()


    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size)

    for epoch in range(epochs):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = model(inputs)

            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 50 == 0:
                print('epoch=%d, batch=%d loss: %.04f'
                      % (epoch + 1, i, old_div(sum_loss, 100)))
                sum_loss = 0.0
        # 每跑完一次epoch测试一下准确率 进入测试模式 禁止梯度传递
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('epoch=%d accuracy=%.02f%%' % (epoch + 1, (old_div(100 * correct, total))))

    torch.save(model.state_dict(), './mnist-pytorch/net.pth')




if __name__ == '__main__':
    main()
