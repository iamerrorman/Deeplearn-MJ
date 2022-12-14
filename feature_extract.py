
import torch
from base_data_process import *


def make_4x9x4_tensor(tiles, mode=None):
    """
    用于将牌集合数据变成特征
    :param tiles: 牌集合
    :param mode:  king模式下，表示tiles只有一张牌
    :return: 非字牌编码 fz_tensor（4*3*9）字牌编码 z_tensor（4*1*7）
    :author: 刘振杰
    """
    if mode == "king":
        tiles = [tiles]
    tiles = [translate3(i) for i in tiles]
    fz_tensor = torch.randint(1, (4, 3, 9), dtype=torch.float)
    z_tensor = torch.randint(1, (4, 1, 7), dtype=torch.float)
    un_tile = list(set(tiles))

    if mode == "king":
        fz_tensor = torch.randint(1, (1, 3, 9), dtype=torch.float)
        z_tensor = torch.randint(1, (1, 1, 7), dtype=torch.float)

    for item in un_tile:
        if item == 255:
            continue
        if 0 <= item <= 26:
            x = tiles.count(item) - 1
            while x != -1:
                fz_tensor[x][item // 9][item % 9] = 1.0
                x -= 1
        else:
            x = tiles.count(item) - 1
            while x != -1:
                z_tensor[x][0][item % 7] = 1.0
                x -= 1
    return fz_tensor, z_tensor


def get_feature(data):
    """
    将制作好的json文件内的数据变成我们想要的特征编码
    :param data: 牌局信息
    :return: 非字牌特征（tensor_fz）， 字牌特征（tensor_z）
    :author: 刘振杰
    """
    handcards = data["handcards"]
    discards_real = data["discards_real"]
    fulu_chow = data["fulu_chow"]
    fulu_peng = data["fulu_peng"]
    fulu_kong = data["fulu_kong"]
    king_card = data["king_card"]
    round = data["round"]
    last_discard = data["last_discard"]
    tensor_fz, tensor_z = make_4x9x4_tensor(handcards)

    for item in discards_real:  # 各个玩家的丢牌
        fz, z = make_4x9x4_tensor(item)
        tensor_fz = torch.cat((tensor_fz, fz), dim=0)
        tensor_z = torch.cat((tensor_z, z), dim=0)

    for item in fulu_chow:  # 各个玩家的吃牌
        item_ = []
        for it in item:
            item_ += it
        fz, z = make_4x9x4_tensor(item_)
        tensor_fz = torch.cat((tensor_fz, fz), dim=0)
        tensor_z = torch.cat((tensor_z, z), dim=0)

    for item in fulu_peng:  # 各个玩家的碰
        item_ = []
        for it in item:
            item_ += it
        fz, z = make_4x9x4_tensor(item_)
        tensor_fz = torch.cat((tensor_fz, fz), dim=0)
        tensor_z = torch.cat((tensor_z, z), dim=0)

    for item in fulu_kong:  # 各个玩家的杠牌
        item_ = []
        for it in item:
            item_ += it
        fz, z = make_4x9x4_tensor(item_)
        tensor_fz = torch.cat((tensor_fz, fz), dim=0)
        tensor_z = torch.cat((tensor_z, z), dim=0)

    # 添加宝牌信息
    fz, z = make_4x9x4_tensor(king_card, "king")
    tensor_fz = torch.cat((tensor_fz, fz), dim=0)
    tensor_z = torch.cat((tensor_z, z), dim=0)

    # 添加上一个玩家打出的牌
    fz, z = make_4x9x4_tensor(last_discard, "king")
    tensor_fz = torch.cat((tensor_fz, fz), dim=0)
    tensor_z = torch.cat((tensor_z, z), dim=0)

    # 添加轮次信息
    fz, z = make_4x9x4_tensor(round, "king")
    tensor_fz = torch.cat((tensor_fz, fz), dim=0)
    tensor_z = torch.cat((tensor_z, z), dim=0)

    return tensor_fz, tensor_z


if __name__ == '__main__':
    # hands = [5, 22, 34, 34, 36, 37, 38, 39, 40, 41, 6, 23, 41, 1]
    # hands = 1
    # make_4x9x4_tensor(hands)
    #
    # pass
    df = open("D:\\project\\python\\MJDecisionAI-DeepLearn\\data\\output\\hu\\hu\\0.json", encoding="utf-8")
    data = json.load(df)
    tensor_fz, tensor_z = get_feature(data)
    print(tensor_fz.shape)
    print(tensor_z.shape)
