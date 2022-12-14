import random

from base_data_process import *
import os


def get_action_label():
    """
    用于生成碰牌所有标签，和对应的反对应（用于实现后面的多模型决策）
    :return: 生成决策对应id, 以及id对应的标签 action_to_id, id_to_action
    :author: 曹梦阳
    """
    action_list = ["pass"]  # 除了杠还有过这个操作
    for i in range(34):
        action_list.append(f"ming_gang_{i}")
        action_list.append(f"an_gang_{i}")
        action_list.append(f"bu_gang_{i}")
    action_to_id = {}
    for item in action_list:  # 生成标签对应的id
        action_to_id[item] = len(action_to_id.keys())

    id_to_action = {}
    id_to_action = {value: key for key, value in action_to_id.items()}
    return action_to_id, id_to_action

action_to_id, id_to_action = get_action_label()


def get_action_mask(feature_json):  # 获取行为掩码
    """
    用于生成行为掩码，来控制神经网络的决策方向。
    :param feature_json: 该参数是已经处理好的标签json数据
    :return: 返回该标签对应的行为掩码， action_mask
    :author: 曹梦阳
    """
    action_mask = [0] * 103  # 行为掩码，因为是杠牌所以有103（加过这个操作）
    action_mask[0] = 1  # 因为过是一定可以操作的
    last_discard = feature_json["last_discard"]  # 上一个玩家丢弃的牌
    handcards = feature_json["handcards"]  # 高玩手牌
    high_player_fulu_peng = feature_json["fulu_peng"][0]  # 获取高玩碰的牌
    actions = []  # 操作集合
    if handcards.count(last_discard) >= 3:  # 明杠
        actions.append(f"ming_gang_{translate3(last_discard)}")

    un_handcards = list(set(handcards))
    for i in un_handcards:  # 暗杠
        if handcards.count(i) == 4:  # 能暗杠
            actions.append(f"an_gang_{translate3(i)}")

    # 补杠
    for item in high_player_fulu_peng:
        if len(item) == 3 and item[0] == item[1] and handcards.count(item[0]) == 1:  # 能自摸补杠
            actions.append(f"bu_gang_{translate3(item[0])}")


    for item in actions:  # 制作action_mask
        action_mask[action_to_id[item]] = 1
    return action_mask


def get_label(battle_info_item):
    """
    用于获取json数据标签的的label
    :param battle_info_item: 整场信息中一个回合的数据
    :return: 返回该行为对应的label
    :author: 曹梦阳
    """
    action_type = battle_info_item["action_type"]
    if action_type == "d":
        return "pass"
    operate_card = battle_info_item["operate_card"]  # 获取操作牌
    operate_card_label = translate3(operate_card)  # 获取操作手牌label
    action_type = battle_info_item["action_type"]  # 获取操作类型
    action = ""
    if action_type == "K":
        action = f"an_gang_{operate_card_label}"
    if action_type == "k":
        action = f"ming_gang_{operate_card_label}"
    if action_type == "t":
        action = f"bu_gang_{operate_card_label}"

    return action


def generate_json(battle_info_item, king_card, pos, json_file_root, file_num_dict):  # 对每个x和y都生成一个json文件
    """
    用于生成json数据标签
    :param battle_info_item: 整场信息中一个回合的数据
    :param king_card: 宝牌id
    :param pos: 调整好的玩家位置，保证 位置放置为[当前玩家id, 下家id, 对家id, 上家id]
    :param json_file_root: 要存放数据的位置
    :param file_num_dict: 用于控制处理文件个数
    :return:
    :author: 曹梦阳
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


def is_can_gang(last_discard, battle_info_item, high_score_player_pos):
    """
    用于判断当前高玩是否能执行碰牌操作
    :param last_discard: 上一个玩家丢弃的牌
    :param battle_info_item: 整场信息中一个回合的数据
    :param high_score_player_pos: 高玩座位id
    :return:
    :author: 曹梦阳
    """
    handcards = battle_info_item["handcards"][high_score_player_pos]  # 获取高手玩家的手牌
    fulu = battle_info_item["discards_op"][high_score_player_pos]  # 获取高手玩家的副露
    if handcards.count(last_discard) >= 3:  # 能明杠
        return True

    for item in fulu:
        if len(item) == 3 and item[0] == item[1] and item[0] == last_discard:  # 能补杠
            return True
    return False


def convert(x, flag19, flag1725, flag_wt, flag_tt):
    """
    用于数据增强
    :param x: 需要转变的数
    :param flag19: 是否需要19对调
    :param flag_wt: 是否需要万条对调
    :param flag1725: 是否需要17,25对调
    :param flag_tt: 是否需要条筒对调
    :return: 转变好的数
    :author: 曹梦阳
    """
    if x == 255:
        return  255
    # 19对调
    if flag19:
        if x == 1:
            x = 9
        elif x == 9:
            x = 1
    # 19对调
    if flag1725:
        if x == 17:
            x = 25
        elif x == 25:
            x = 17
    # 万条对调
    if flag_wt:
        if 1 <= x <= 9:
            x = 17 + x - 1
        elif 17 <= x <= 25:
            x = 1 + x - 17
    # 条筒对调
    if flag_tt:
        if 17 <= x <= 25:
            x = 33 + x - 17
        elif 33 <= x <= 41:
            x = 17 + x - 33
    return x


def data_augment(battle_info_item):
    """
    用于数据增强
    :param battle_info_item: 整场信息中一个回合的数据
    :return: 返回增强完的数据
    :author: 曹梦阳
    """
    battle_info_item_ = copy.deepcopy(battle_info_item)
    # 19对调
    flag19 = True if random.randint(0, 10) > 4 else False
    # 1725对调
    flag1725 = True if random.randint(0, 10) > 4 else False
    # 万条对调
    flag_wt = True if random.randint(0, 10) > 4 else False
    # 条筒对调
    flag_tt = True if random.randint(0, 10) > 4 else False
    discards = battle_info_item_["discards"]
    discards_real = battle_info_item_["discards_real"]
    discards_op = battle_info_item_["discards_op"]
    handcards = battle_info_item_["handcards"]
    operate_card = battle_info_item_["operate_card"]
    last_discard = battle_info_item_["last_discard"]
    for i in range(0, len(discards)):  # 对弃牌集合操作(包括cpg)
        for j in range(0, len(discards[i])):
            discards[i][j] = convert(discards[i][j], flag19, flag1725, flag_wt, flag_tt)

    for i in range(0, len(discards_real)):  # 对弃牌集合操作(不包括cpg)
        for j in range(0, len(discards_real[i])):
            discards_real[i][j] = convert(discards_real[i][j], flag19, flag1725, flag_wt, flag_tt)

    for i in range(0, len(discards_op)):  # 对副露操作
        for j in range(0, len(discards_op[i])):
            for k in range(0, len(discards_op[i][j])):
                discards_op[i][j][k] = convert(discards_op[i][j][k], flag19, flag1725, flag_wt, flag_tt)

    for i in range(0, len(handcards)):  # 对手牌集合操作
        for j in range(0, len(handcards[i])):
            handcards[i][j] = convert(handcards[i][j], flag19, flag1725, flag_wt, flag_tt)

    operate_card = convert(operate_card, flag19, flag1725, flag_wt, flag_tt)
    last_discard = convert(last_discard, flag19, flag1725, flag_wt, flag_tt)
    battle_info_item_["discards"] = discards
    battle_info_item_["discards_real"] = discards_real
    battle_info_item_["discards_op"] = discards_op
    battle_info_item_["handcards"] = handcards
    battle_info_item_["operate_card"] = operate_card
    battle_info_item_["last_discard"] = last_discard
    return battle_info_item_


def process_file(file_intput_root, file_name, file_output_root):
    """
    用于处理原始的麻将数据文件，生成对应的数据标签
    :param file_intput_root: 原始文件数据位置根目录
    :param file_name: 原始文件名
    :param file_output_root: 处理好的文件放置位置
    :return:
    :author: 曹梦阳
    """
    df = open(os.path.join(file_intput_root, file_name), encoding="utf-8")
    data = json.load(df)
    high_score_player_pos = data["players_id"].index(data["high_score_player_id"])  # 获取高手玩家的位置
    king_card = data["king_card"]  # 获取宝牌id
    battle_info = data["battle_info"]  # 获取对局的整场信息
    pos = get_pos(high_score_player_pos)  # 获取调整好的玩家id，[高玩id, 下家id, 对家id, 上家id]
    file_num_dict = {}  # 控制文件数量

    # 遍历整场信息，获取高手决策点
    for idx, item in enumerate(battle_info):
        battle_info_item = copy.deepcopy(item)
        # 杠牌
        # 暗杠
        if item["seat_id"] == pos[0] and item["action_type"] == "K":
            battle_info_item["last_discard"] = 255  # 暗杠，上一个玩家丢的牌没用
            # 制作暗杠标签
            for i in range(0, 15):
                file_num_dict = generate_json(data_augment(battle_info_item), king_card, pos, file_output_root, file_num_dict)

        # 自摸补杠，不会触发
        if item["seat_id"] == pos[0] and item["action_type"] == "t" and idx >= 1 and battle_info[idx - 1][
            "seat_id"] == pos[0] and battle_info[idx - 1]["action_type"] == "G":
            battle_info_item["last_discard"] = 255  # 丢牌，上一个玩家丢的牌没用
            # 制作自摸补杠
            for i in range(0, 15):
                file_num_dict = generate_json(data_augment(battle_info_item), king_card, pos, file_output_root,
                                              file_num_dict)

        if item["seat_id"] != pos[0] and item["action_type"] == "d":  # 不是高手玩家丢的牌
            # 情况一，下一个玩家是高玩， 且是明杠
            if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1][
                "action_type"] == "k":
                battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
                battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
                # 制作明杠标签
                for i in range(0, 15):
                    file_num_dict = generate_json(data_augment(battle_info_item_), king_card, pos, file_output_root,
                                                  file_num_dict)

            # 情况二，下一个玩家是高玩， 且是补杠
            if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1][
                "action_type"] == "t":
                battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
                battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
                # 制作补杠标签
                for i in range(0, 15):
                    file_num_dict = generate_json(data_augment(battle_info_item_), king_card, pos, file_output_root,
                                                  file_num_dict)
            # 情况三，下一个玩家不是高玩，且操作不是杠，且高玩手牌中能杠
            if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] != pos[0] and battle_info[idx + 1][
                "action_type"] not in ["t", "K", "k"] and is_can_gang(item["operate_card"], battle_info_item, pos[0]):
                battle_info_item["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
                # 制作过的标签
                file_num_dict = generate_json(battle_info_item, king_card, pos, file_output_root, file_num_dict)

if __name__ == '__main__':
    # print(get_action_label()[0])
    file_root = "D:\\project\\python\\high_score_json"
    dir_name = os.listdir(file_root)
    for item in dir_name:
        file_label_path = os.path.join(file_root, item)
        file_name = os.listdir(file_label_path)
        for it in file_name:
            process_file(file_label_path, it, r"..\data\output\gang")
        print(f"{item}文件完成！")
