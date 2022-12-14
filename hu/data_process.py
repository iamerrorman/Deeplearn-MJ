import random

from base_data_process import *
import os
from hu.shangraoMJ_v5 import get_score_dict


def get_action_label():
    """
    用于生成碰牌所有标签，和对应的反对应（用于实现后面的多模型决策）
    :return: 生成决策对应id, 以及id对应的标签 action_to_id, id_to_action
    :author: 刘振杰
    """
    action_list = ["pass", "hu"]  # 除了胡牌还有过这个操作
    action_to_id = {}
    for item in action_list:
        action_to_id[item] = len(action_to_id.keys())

    id_to_action = {}
    id_to_action = {value: key for key, value in action_to_id.items()}
    return action_to_id, id_to_action


action_to_id, id_to_action = get_action_label() # 先生成决策对应id, 以及id对应的标签 action_to_id, id_to_action

def get_action_mask(feature_json):  # 获取行为掩码
    """
    用于生成行为掩码，来控制神经网络的决策方向。
    :param feature_json: 该参数是已经处理好的标签json数据
    :return: 返回该标签对应的行为掩码， action_mask
    :author: 刘振杰
    """
    action_mask = [0] * 2  # 行为掩码，因为是胡牌所以有2（加过这个操作）
    action_mask[0] = 1  # 因为过是一定可以操作的
    handcards = feature_json["handcards"]  # 高玩手牌
    actions = []  # 操作集合
    high_player_fulu = feature_json["fulu_peng"][0] + feature_json["fulu_chow"][0] + feature_json["fulu_kong"][0]
    king_card = feature_json["king_card"]
    xts = get_score_dict(handcards, high_player_fulu, king_card, 0)
    if xts == 0:  # 所需的牌为零，则可以胡牌
        actions.append("hu")
    for item in actions:  # 制作action_mask
        action_mask[action_to_id[item]] = 1
    return action_mask


def get_label(battle_info_item):
    """
    用于获取json数据标签的的label
    :param battle_info_item: 整场信息中一个回合的数据
    结构如下：
    {
        "seat_id": 3,   # 当前玩家的座位id
        "discards": [[49, 33, 38, 9], [50, 35, 6, 38, 37, 23], [34, 37, 38, 5, 35], [34, 1, 55, 50, 55]],   # 所有玩家的出牌(不包括被别人吃碰杠的牌)
        "discards_real": [[51, 49, 33, 38, 9, 22], [50, 35, 6, 38, 37, 23], [34, 37, 39, 38, 3, 5, 35], [34, 1, 2, 55, 50, 24, 55]],   # 所有玩家的出牌(包括被别人吃碰杠的牌)
        "discards_op": [[[2, 3, 4]], [[24, 24, 24], [21, 22, 23]], [], [[51, 51, 51], [39, 40, 41], [3, 4, 5]]],   # 所有玩家的副露
        "handcards": [[17, 20, 54, 17, 21], [17, 20, 54, 17, 21], [17, 20, 54, 17, 21], [17, 20, 54, 17, 21]],   # 所有玩家的手牌
        “action_type”: D,  # 动作
        "operate_card": 21,   # 当前操作的是哪张牌
        “passivity_action_site”: 255,  # 被吃碰杠的玩家的座位id
        “combine_cards”:   # 如果action_type是吃碰杠，则表示吃碰杠后形成的副露
        "round": 7  # 当前是第几轮。目前定的是每次到庄家出牌则round+1
        ”last_discard“: 2  # 上一个玩家丢弃的牌
    },
    :return: 返回该行为对应的label
    :author: 刘振杰
    """
    action_type = battle_info_item["action_type"]
    if action_type == "G":  # 如果传过来的数据的操作是摸牌就是制作过的标签
        return "pass"
    else:
        return "hu"


def generate_json(battle_info_item, king_card, pos, json_file_root, file_num_dict):  # 对每个x和y都生成一个json文件
    """
    用于生成json数据标签
    :param battle_info_item: 整场信息中一个回合的数据
    同上
    :param king_card: 宝牌id
    :param pos: 调整好的玩家位置，保证 位置放置为[当前玩家id, 下家id, 对家id, 上家id]
    :param json_file_root: 要存放数据的位置
    :param file_num_dict: 用于控制处理文件个数，不用管
    :return:
    :author: 刘振杰
    """
    feature_json = {}  # 制作json数据文件
    feature_json["handcards"] = battle_info_item["handcards"][pos[0]]  # 获取手牌信息，pos[0]高玩
    feature_json["discards_real"] = convert_item(battle_info_item["discards_real"], pos)  # 获取各个玩家丢弃的手牌
    feature_json["fulu_chow"], feature_json["fulu_peng"], feature_json["fulu_kong"] = get_fulu_classify(
        battle_info_item["discards_op"], pos)  # 获取各个玩家副露
    feature_json["pos"] = pos  # 获取各个玩家调整好的位置
    feature_json["king_card"] = king_card  # 获取宝牌信息
    feature_json["round"] = battle_info_item["round"]  # 获取轮次信息
    feature_json["last_discard"] = battle_info_item["last_discard"]  # 获取上一个玩家家丢弃的牌
    feature_json["operate_card"] = battle_info_item["operate_card"]  # 获取操作牌id
    feature_json["action_type"] = battle_info_item["action_type"]  # 获取操作类型
    feature_json["action_mask"] = get_action_mask(feature_json)  # 获取行为掩码
    feature_json["label"] = get_label(battle_info_item)  # 获取标签
    # feature_json["old"] = battle_info_item  # 做验证，可去

    # 下面代码可看可不看，就是把上面做好的字典写入json文件中

    file_label_root = os.path.join(json_file_root, str(feature_json["label"]))

    if not os.path.exists(file_label_root):  # 标签文件目录不存在
        os.mkdir(file_label_root)  # 创建标签文件目录

    file_num = file_num_dict.get(str(feature_json["label"]))  # 获取存好的文件数量

    if file_num is None:  # 若没有就获取一下文件数量
        file_num = len(os.listdir(file_label_root))
        file_num_dict[str(feature_json["label"])] = file_num

    with open(os.path.join(file_label_root, f"{file_num}.json"), "w", encoding="utf-8") as file_json:  # 将数据写入文件中
        file_json.write(json.dumps(feature_json, indent=4))

    file_num_dict[str(feature_json["label"])] = file_num_dict[str(feature_json["label"])] + 1  # 文件数量加一

    return file_num_dict


def is_can_hu(battle_info_item, high_score_player_pos, king_card):
    """
    用于判断当前高玩是否能执行胡牌操作
    :param king_card: 宝牌信息
    :param battle_info_item: 整场信息中一个回合的数据
    结构同上
    :param high_score_player_pos: 高玩座位id
    :return: 返回高玩是否能执行胡牌操作，能则True，否则False
    :author: 刘振杰
    """
    handcards = battle_info_item["handcards"][high_score_player_pos]  # 获取高手玩家的手牌
    fulu = battle_info_item["discards_op"][high_score_player_pos]  # 获取高手玩家的副露
    xts = get_score_dict(handcards, fulu, king_card, 0)
    if xts == 0:
        return True
    else:
        return False


def process_file(file_intput_root, file_name, file_output_root):
    """
    用于处理原始的麻将数据文件，生成对应的数据标签
    :param file_intput_root: 原始文件数据位置根目录
    :param file_name: 原始文件名
    :param file_output_root: 处理好的文件放置位置
    :return:
    :author: 刘振杰
    """
    df = open(os.path.join(file_intput_root, file_name), encoding="utf-8")  # 打开原始的麻将文件数据
    data = json.load(df)   # 使用json读取原始的麻将文件数据
    high_score_player_pos = data["players_id"].index(data["high_score_player_id"])  # 获取高手玩家的位置
    king_card = data["king_card"]  # 获取宝牌id
    battle_info = data["battle_info"]  # 获取对局的整场信息
    pos = get_pos(high_score_player_pos)  # 获取调整好的玩家id，[高玩id, 下家id, 对家id, 上家id]
    file_num_dict = {}  # 控制文件数量

    # 遍历整场信息，获取高手决策点
    for idx, item in enumerate(battle_info):
        battle_info_item = copy.deepcopy(item)
        # 胡牌，注意上饶麻将中只能通过自摸胡牌
        if item["seat_id"] == pos[0] and item["action_type"] == "A":  # 当前玩家是高手玩家，并且是胡牌操作
            battle_info_item["last_discard"] = 255  # 胡牌，上一个玩家丢的牌没用
            # 制作胡牌标签
            file_num_dict = generate_json(battle_info_item, king_card, pos, file_output_root, file_num_dict)

        # 过
        # 当前玩家是高手玩家，并且是摸牌操作，并且他可以胡牌，但他没有胡牌，也就是下一个操作不是胡牌
        if idx < len(battle_info) - 1 and item["seat_id"] == pos[0] and item["action_type"] == "G" and is_can_hu(
                item, pos[0], king_card) and battle_info[idx + 1]["action_type"] != "A":
            battle_info_item["last_discard"] = 255  # 胡牌，上一个玩家丢的牌没用
            # 制作过标签
            file_num_dict = generate_json(battle_info_item, king_card, pos, file_output_root, file_num_dict)


if __name__ == '__main__':
    # # print(get_action_label()[1])
    file_root = "D:\\project\\python\\high_score_json"
    dir_name = os.listdir(file_root)
    for item in dir_name:
        file_label_path = os.path.join(file_root, item)
        file_name = os.listdir(file_label_path)
        for it in file_name:
            process_file(file_label_path, it, r"..\data\output\hu")
        print(f"{item}文件完成！")
