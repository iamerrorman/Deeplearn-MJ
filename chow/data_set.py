import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from feature_extract import get_feature
from data_process import *

class Dataset_chow(Dataset):
    def __init__(self, root, mode, action_dict):
        super(Dataset_chow, self).__init__()
        self.root = root
        self.action_dict = action_dict
        file_names = os.listdir(root)  # 获取标签文件名
        # self.sample_list = {}
        self.train_sample = []  # 制作训练数据文件
        self.test_sample = []  # 制作测试数据文件
        for item in file_names:
            temp = [os.path.join(item, i) for i in os.listdir(os.path.join(self.root, item))]
            if item == "pass":
                temp = temp[: int(len(temp) * 0.5)]  # pass就取一半来训练，另一半用来测试
            self.train_sample += temp[:int(len(temp) * 0.8)]
            self.test_sample += temp[int(len(temp) * 0.8):]

        # 再将合并的文件进行打乱
        random.shuffle(self.train_sample)
        random.shuffle(self.test_sample)

        if mode == "train":
            self.sample = self.train_sample
        else:
            self.sample = self.test_sample

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        df = open(os.path.join(self.root, self.sample[idx]), encoding="utf-8")
        data = json.load(df)
        try:
            features1, features2 = get_feature(data)
            return features1, features2, torch.tensor(data["action_mask"]), torch.tensor(
                self.action_dict[data["label"]])
        except KeyError:
            print(os.path.join(self.root, self.sample[idx]))
            return 0


if __name__ == '__main__':
    root = "D:\\project\\python\\MJDecisionAI-DeepLearn\\data\\output\\chow"
    ds_train = Dataset_chow(root, "train", action_to_id)
    data_train = DataLoader(ds_train, batch_size=100)
    for x1, x2, mask, targets in data_train:
        print(x1.shape)
        print(x2.shape)
        print(mask.shape)
        print(targets.shape)
        exit()