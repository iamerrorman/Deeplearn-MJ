import os
import torch
from data_set import Dataset_chow
from torch.utils.data import DataLoader
from feature_extract import *
from model.model_chow import *
from torch.optim import AdamW
from torch import nn
import torch
from data_process import *
from base_data_process import *
import os

# 构建网络
my_model = Model_v1(71, 64)
my_model.load_state_dict(torch.load("D:\\project\\python\\MJDecisionAI-DeepLearn\\chow\\mode_param\\ep0_acc=0.7129.pth"))

my_model.eval()
print(f"--test-start--")
data = {
    "handcards": [
        23,
        24,
        33,
        38,
        49,
        49,
        49,
        39,
        5,
        6
    ],
    "discards_real": [
        [
            53,
            51,
            41,
            41,
            34,
            19
        ],
        [
            53,
            50,
            25,
            22,
            9,
            8
        ],
        [
            55,
            50,
            25,
            8,
            1,
            38,
            21,
            6
        ],
        [
            53,
            51,
            52,
            17,
            20,
            18,
            2,
            4
        ]
    ],
    "fulu_chow": [
        [
            [
                1,
                2,
                3
            ]
        ],
        [
            [
                17,
                18,
                19
            ]
        ],
        [
            [
                20,
                21,
                22
            ]
        ],
        [
            [
                6,
                7,
                8
            ],
            [
                4,
                5,
                6
            ]
        ]
    ],
    "fulu_peng": [
        [],
        [],
        [
            [
                52,
                52,
                52
            ]
        ],
        []
    ],
    "fulu_kong": [
        [],
        [],
        [],
        []
    ],
    "pos": [
        2,
        3,
        0,
        1
    ],
    "king_card": 33,
    "round": 8,
    "last_discard": 4,
    "operate_card": 4,
    "action_type": "C",
    "action_mask": [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "label": "left_chow_3"
}
x1, x2 = get_feature(data)
x1 = x1.reshape(1, 71, 3, 9)
x2 = x2.reshape(1, 71, 1, 7)
mask = torch.tensor(data["action_mask"]).reshape(1, 64)
print(my_model.predict(x1, x2, mask).argmax(dim=1))

