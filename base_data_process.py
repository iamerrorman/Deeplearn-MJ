import copy
import os
import json


def get_pos(player_id):  # 获取当前转换位置, [当前玩家id, 下家id, 对家id, 上家id]
    """
    获取当前玩家转换位置
    :param player_id: 当前玩家id
    :return: [当前玩家id, 下家id, 对家id, 上家id]
    :author: 毛芊蕙，张阳阳
    """
    if player_id == 0:
        return [0, 1, 2, 3]
    elif player_id == 1:
        return [1, 2, 3, 0]
    elif player_id == 2:
        return [2, 3, 0, 1]
    else:
        return [3, 0, 1, 2]


def convert_item(items, pos):  # 用于调整列表的位置
    """
    用于按照玩家位置调整列表位置，比如[[1,.], [2,..], [3,..], [4,..]]现在的位置是0, 1, 2, 3  我们要调整为pos对应的位置
    比如[3, 0, 1, 2], 返回[[4,.], [1,..], [2,..], [3,..]]
    :param items: 需要调整的列表
    :param pos: 需要调整的成的位置
    :return: 返回调整好的列表
    :author: 毛芊蕙，张阳阳
    """
    res = []
    for i in pos:
        res.append(items[i])
    return res


def get_fulu_classify(fulu_, pos):  # 通过副露，分类各个玩家的吃、碰、杠的牌
    """
    通过所给的各个玩家的副露，提取每个玩家的吃碰杠的牌，位置就是pos的位置
    :param fulu_: 各个玩家的副露，原始位置是0， 1， 2， 3
    :param pos: 需要调整的成的位置
    :return: 各个玩家吃的牌，各个玩家碰的牌，各个玩家杠的牌（都是调整好的位置）
    :author: 毛芊蕙，张阳阳
    """
    player_chow = [[], [], [], []]  # 用于存储各个玩家吃的牌
    player_peng = [[], [], [], []]  # 用于存储各个玩家碰的牌
    player_kong = [[], [], [], []]  # 用于存储各个玩家杠的牌

    for idx, item in enumerate(fulu_):  # 遍历每个玩家的副露
        for i in item:  # 遍历当前玩家的副露
            if len(i) == 3:  # 如果当前的组成长度为3，则表示是顺子或者刻子，杠子是四个
                if i[0] != i[1]:  # 如果当前的组合是前一个和后一个不相等就是顺子，否则是刻子
                    player_chow[idx].append(i)  # 是顺子放入player_chow中该玩家对应的位置
                else:
                    player_peng[idx].append(i)  # 是刻子放入player_peng中该玩家对应的位置

            if len(i) == 4:
                player_kong[idx].append(i)  # 是杠子放入player_kong中该玩家对应的位置

    return convert_item(player_chow, pos), convert_item(player_peng, pos), convert_item(player_kong, pos)


# def get_self_king_num(battle_info_item, king_card):  # 用于获取自身的宝牌数
#     seat_id = battle_info_item["seat_id"]  # 获取当前玩家id
#     self_handcards = battle_info_item["handcards"][seat_id]  # 获取当前玩家手牌信息
#     cout = 0  # 用于记录宝牌数量
#     for item in self_handcards:  # 统计手牌中的宝牌数量
#         if item == king_card:
#             cout += 1
#     return cout


def translate33_16(i):  # 将下标转换成16进制的牌
    """
    将0-34的手牌转换成16进制手牌
    :param i: 需要转换的手牌
    :return: 转换好的手牌
    :author: 毛芊蕙，张阳阳
    """
    if 0 <= i < 9:
        return i + 1
    elif 9 <= i < 18:
        return i + 8
    elif 18 <= i < 27:
        return i + 15
    elif 27 <= i < 34:
        return i + 22
    else:
        print("[INFO_ZW]:INPUT ERROR")


def translate3(op_card):  # 16进制op_card转换到 0-33 34转换
    """
    与上面的函数功能相反
    :param op_card: 需要转换的手牌
    :return: 转换好的手牌
    :author: 毛芊蕙，张阳阳
    """
    if 1 <= op_card <= 9:
        op_card = op_card - 1
    elif 17 <= op_card <= 25:
        op_card = op_card - 8
    elif 33 <= op_card <= 41:
        op_card = op_card - 15
    elif 49 <= op_card <= 55:
        op_card = op_card - 22
    elif op_card == 255:  # 非法字符，返回一个不存在的牌
        op_card = 34
    return op_card


# def is_can_cpkh(battle_info_item, high_score_player_pos, action_types):  # 判断能不能cpk
#     handcards = battle_info_item["handcards"][high_score_player_pos]  # 获取高手玩家的手牌
#     fulu = battle_info_item["discards_op"][high_score_player_pos]  # 获取高手玩家的副露
#     last_discard = battle_info_item["operate_card"]  # 获取上一个玩家的丢弃的牌
#     if handcards.count(last_discard) >= 2:  # 能碰
#         return True
#
#     if handcards.count(last_discard) >= 3:  # 能明杠
#         return True
#
#     for item in fulu:
#         if len(item) == 3 and item[0] == item[1] and item[0] == last_discard:  # 能补杠
#             return True
#
#     if action_types["discard"] == 1:
#         if handcards.count(last_discard + 1) > 0 and handcards.count(last_discard + 2) > 0:  # 能左吃
#             return True
#         if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard + 1) > 0:  # 能中吃
#             return True
#         if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard - 2) > 0:  # 能右吃
#             return True
#
#     # 胡牌
#
#     return False  # 不能cpkA


# def process_file(file_intput_root, file_name, file_output_root):
#     df = open(os.path.join(file_intput_root, file_name), encoding="utf-8")
#     data = json.load(df)
#     high_score_player_pos = data["players_id"].index(data["high_score_player_id"])  # 获取高手玩家的位置
#     king_card = data["king_card"]  # 获取宝牌id
#     battle_info = data["battle_info"]  # 获取对局的整场信息
#     pos = get_pos(high_score_player_pos)  # 获取调整好的玩家id，[高玩id, 下家id, 对家id, 上家id]
#     file_num_dict = {}
#
#     # 遍历整场信息，获取高手决策点
#     for idx, item in enumerate(battle_info):
#         action_types = {"discard": 0, "discard": 0, "peng": 0, "ming_gang": 0, "an_gang": 0, "zi_bu_gang": 0,
#                         "bu_gang": 0, "fu_pai": 0, "pass": 0}  # 丢、吃、碰、明杠、暗杠、自摸补杠、补杠，胡牌, 过
#         battle_info_item = copy.deepcopy(item)
#
#         # if item["seat_id"] == pos[0] and item["action_type"] in ["t", "k", "k"]:
#         #     print(item["action_type"])
#
#         # 丢牌
#         if item["seat_id"] == pos[0] and item["action_type"] == "d":
#             battle_info_item["last_discard"] = 255  # 丢牌，上一个玩家丢的牌没用
#             battle_info_item["mode"] = "d"  # 丢牌模式
#             action_types["discard"] = 1  # 能丢
#             # 上一个操作是自己摸牌
#             if idx >= 1 and battle_info[idx - 1]["seat_id"] == pos[0] and battle_info[idx - 1]["action_type"] == "G":
#                 action_types["an_gang"] = 1  # 能暗杠
#                 action_types["zi_bu_gang"] = 1  # 能自摸补杠
#                 action_types["fu_pai"] = 1  # 能胡牌
#
#             # # 情况二，上一个操作是cp
#             # if idx >= 1 and battle_info[idx - 1]["seat_id"] == pos[0] and (
#             #         battle_info["action_type"] in ["C", "N"]):
#             battle_info_item["action_types"] = action_types
#             # 制作丢牌标签
#             file_num_dict = generate_json(battle_info_item, king_card, pos, file_output_root, file_num_dict)
#
#         # 暗杠
#         if item["seat_id"] == pos[0] and item["action_type"] == "K":
#             action_types["discard"] = 1  # 能丢
#             action_types["an_gang"] = 1  # 能暗杠
#             action_types["zi_bu_gang"] = 1  # 能自摸补杠
#             action_types["fu_pai"] = 1  # 能胡牌
#             battle_info_item["last_discard"] = 255  # 丢牌，上一个玩家丢的牌没用
#             battle_info_item["mode"] = "K"  # 暗杠牌模式
#             battle_info_item["action_types"] = action_types
#             # 制作暗杠标签
#             file_num_dict = generate_json(battle_info_item, king_card, pos, file_output_root, file_num_dict)
#
#         # 自摸补杠
#         if item["seat_id"] == pos[0] and item["action_type"] == "t" and idx >= 1 and battle_info[idx - 1][
#             "seat_id"] == pos[0] and battle_info[idx - 1]["action_type"] == "G":
#             action_types["discard"] = 1  # 能丢
#             action_types["an_gang"] = 1  # 能暗杠
#             action_types["zi_bu_gang"] = 1  # 能自摸补杠
#             action_types["fu_pai"] = 1  # 能胡牌
#             battle_info_item["last_discard"] = 255  # 丢牌，上一个玩家丢的牌没用
#             battle_info_item["mode"] = "t"  # 暗杠牌模式
#             battle_info_item["action_types"] = action_types
#             # 制作暗杠标签
#             file_num_dict = generate_json(battle_info_item, king_card, pos, file_output_root, file_num_dict)
#
#         # 吃、碰，杠，胡牌
#         if item["seat_id"] != pos[0] and item["action_type"] == "d":  # 不是高手玩家丢的牌
#
#             action_types["peng"] = 1  # 能碰
#             action_types["ming_gang"] = 1  # 能明杠
#             action_types["bu_gang"] = 1  # 能补杠
#             action_types["fu_pai"] = 1  # 能胡牌
#             action_types["pass"] = 1  # 能过
#             if item["seat_id"] == pos[3]:  # 如果上一个玩家是高玩的上家，还可以吃
#                 action_types["discard"] = 1  # 能吃
#
#             # 情况一，下一个玩家是高玩，且是吃牌
#             if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1][
#                 "action_type"] == "C":
#                 battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
#                 battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
#                 battle_info_item_["mode"] = "C"  # 吃牌模式
#                 battle_info_item_["action_types"] = action_types
#                 # 制作吃牌标签
#                 file_num_dict = generate_json(battle_info_item_, king_card, pos, file_output_root, file_num_dict)
#
#             # 情况二，下一个玩家是高玩，且是碰牌
#             if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1][
#                 "action_type"] == "N":
#                 battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
#                 battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
#                 battle_info_item_["mode"] = "N"  # 碰牌模式
#                 battle_info_item_["action_types"] = action_types
#                 # 制作碰牌标签
#                 file_num_dict = generate_json(battle_info_item_, king_card, pos, file_output_root, file_num_dict)
#
#             # 情况三，下一个玩家是高玩， 且是明杠
#             if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1][
#                 "action_type"] == "k":
#                 battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
#                 battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
#                 battle_info_item_["mode"] = "k"  # 明杠模式
#                 battle_info_item_["action_types"] = action_types
#                 # 制作明杠标签
#                 file_num_dict = generate_json(battle_info_item_, king_card, pos, file_output_root, file_num_dict)
#
#             # 情况四，下一个玩家是高玩， 且是补杠
#             if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1][
#                 "action_type"] == "t":
#                 battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
#                 battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
#                 battle_info_item_["mode"] = "t"  # 补杠牌模式
#                 battle_info_item_["action_types"] = action_types
#                 # 制作补杠标签
#                 file_num_dict = generate_json(battle_info_item_, king_card, pos, file_output_root, file_num_dict)
#
#             # 情况五，下一个玩家不是高玩，且操作不是吃、碰、杠、胡，且高玩手牌中能吃，或者能碰，或者能杠、胡牌
#             if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] != pos[0] and battle_info[idx + 1][
#                 "action_type"] not in ["C", "t", "K", "A"] and is_can_cpkh(battle_info_item, pos[0], action_types):
#                 battle_info_item["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
#                 battle_info_item["mode"] = "pass"  # 过模式
#                 battle_info_item["action_types"] = action_types
#                 # 制作过的标签
#                 file_num_dict = generate_json(battle_info_item, king_card, pos, file_output_root, file_num_dict)
#         # 胡牌


if __name__ == '__main__':
    # discard, peng, kong = get_fulu_classify([[[52, 52, 52]],
    # [],
    # [[39, 39, 39], [33, 34, 35]],
    # [[49, 49, 49], [7, 8, 9], [22, 23, 24]]], [1, 2, 3, 0])
    # print(discard)
    # print(peng)
    # print(kong)
    # print(get_action_label())

    pass