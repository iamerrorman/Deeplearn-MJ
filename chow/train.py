import os
from data_set import Dataset_chow
from torch.utils.data import DataLoader
from model import *
from torch.optim import AdamW
from torch import nn
import torch
from data_process import *
from base_data_process import *
import os


device = torch.device("cuda")
# 获取数据集
ds_train = Dataset_chow("D:\\project\\python\\MJDecisionAI-DeepLearn\\data\\output\\chow", "train", action_to_id)
data_train = DataLoader(ds_train, batch_size=100)
ds_vel = Dataset_chow("D:\\project\\python\\MJDecisionAI-DeepLearn\\data\\output\\chow", "test", action_to_id)
data_vel = DataLoader(ds_vel, batch_size=100)
# 构建网络
my_model = Model_v1(71, 64)
my_model.load_state_dict(torch.load("D:\\project\\python\\MJDecisionAI-DeepLearn\\chow\\mode_param\\ep0_acc=0.7129.pth"))
my_model.to(device)

# 学习率
lr = 1e-3

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 定义优化器
optimer = AdamW(my_model.parameters(), lr=lr, weight_decay=0.01)

# epoch
EPOCH = 50

# 开始训练
for i in range(1, EPOCH):

    print(f"---epoch-{i}-start---")
    loss_sum = 0
    my_model.train()
    for idx, (intputs1, intputs2, mask, targets) in enumerate(data_train):
        # print(inputs.shape)
        intputs1, intputs2, mask, targets = intputs1.to(device), intputs2.to(device), mask.to(device), targets.to(device)
        outputs = my_model(intputs1, intputs2)
        loss = loss_fn(outputs, targets)
        loss_sum += loss
        if idx % 5 == 1:
            print(f"epoch-{i}-train-{idx}-loss:{loss}")
        optimer.zero_grad()
        loss.backward()
        optimer.step()

    print(f"epoch-{i}-loss_sum:{loss_sum}")

    my_model.eval()
    print(f"--test-{i}-start--")
    with torch.no_grad():
        corret_sum = 0
        sample_sum = 0
        for intputs1, intputs2, mask, targets in data_vel:
            intputs1, intputs2, mask, targets = intputs1.to(device), intputs2.to(device), mask.to(device), targets.to(
                device)
            outputs = my_model.predict(intputs1, intputs2, mask)
            corret_sum += (outputs.argmax(dim=1) == targets).sum()
            sample_sum += targets.shape[0]
        acc = corret_sum / sample_sum
        print(f"--test--{i}--acc:{acc:0.4f}")
        torch.save(my_model.state_dict(), os.path.join("D:\\project\\python\\MJDecisionAI-DeepLearn\\chow"
                                                       "\\mode_param", f"ep{i}_acc={acc:0.4f}.pth"))

