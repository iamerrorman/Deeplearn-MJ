from base_data_process import *
import os


def get_action_label():
    """
    用于生成吃牌所有标签，和对应的反对应（用于实现后面的多模型决策）
    :return: 生成决策对应id, 以及id对应的标签 action_to_id, id_to_action
    :author: 毛芊蕙
    """
    action_list = ["pass"]  # 除了左中右吃还有过这个操作
    for i in range(27):  # 吃牌决策，因为只有非字牌能吃，只有循环到27就行
        if i % 9 == 0:  # 非字牌中的1万，1条，1筒只能被左吃
            action_list.append(f"left_chow_{i}")
        elif i % 9 == 1:  # 非字牌中的2万，2条，2筒只能被左吃，中吃
            action_list.append(f"left_chow_{i}")
            action_list.append(f"mid_chow_{i}")
        elif i % 9 == 7:  # 非字牌中的8万，8条，8筒只能被右吃，中吃
            action_list.append(f"right_chow_{i}")
            action_list.append(f"mid_chow_{i}")
        elif i % 9 == 8:  # 非字牌中的9万，9条，9筒只能被右吃
            action_list.append(f"right_chow_{i}")
        else:  # 非字牌中其他牌能被左吃，中吃，右吃
            action_list.append(f"left_chow_{i}")
            action_list.append(f"mid_chow_{i}")
            action_list.append(f"right_chow_{i}")

    action_to_id = {}  # 用于存储决策对应id，使用字典的方式例如,{"left_chow_0": 0, "left_chow_1": 1, .....}
    for item in action_list:  # 生成标签对应的id
        action_to_id[item] = len(action_to_id.keys())

    id_to_action = {}  # 用于存储决策对应id，使用字典的方式例如,{0: "left_chow_0", 1: "left_chow_1", .....}
    id_to_action = {value: key for key, value in action_to_id.items()}
    return action_to_id, id_to_action


action_to_id, id_to_action = get_action_label()  # 先生成决策对应id, 以及id对应的标签 action_to_id, id_to_action


def get_action_mask(feature_json):  # 获取行为掩码
    """
    用于生成行为掩码，来控制神经网络的决策方向。什么叫行为掩码？
    就是当你准备弃牌时，你的神经网络是不是只能选择手牌中有的牌去丢弃，行为掩码就是控制当前你能执行的行为，1表示可以执行，0就是不能执行
    注意：所有文件中的数据中牌都是16进制牌，而在特征中的都是0-33，所以在制作决策字典的时候注意转换
    :param feature_json: 该参数是已经处理好的标签json数据，他是一个字典数据，里面有什么字段呢？看函数generate_json(就在这个文件中)
    :return: 返回该标签对应的行为掩码， action_mask
    :author: 毛芊蕙
    """
    action_mask = [0] * 64  # 行为掩码，因为是吃牌所以有64（加过这个操作），为什么是64？因为各个牌的左中右吃有63种情况，加一个过就64啦
    action_mask[0] = 1  # 因为过是一定可以操作的
    last_discard = feature_json["last_discard"]  # 上一个玩家丢弃的牌
    handcards = feature_json["handcards"]  # 高玩手牌
    actions = []  # 操作集合
    if last_discard < 27:  # 吃牌的话，只有万条筒才可以被吃，字牌不能被吃
        if handcards.count(last_discard + 1) > 0 and handcards.count(last_discard + 2) > 0:  # 能左吃
            actions.append(f"left_chow_{translate3(last_discard)}")
        if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard + 1) > 0:  # 能中吃
            actions.append(f"mid_chow_{translate3(last_discard)}")
        if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard - 2) > 0:  # 能右吃
            actions.append(f"right_chow_{translate3(last_discard)}")

    for item in actions:  # 制作action_mask
        action_mask[action_to_id[item]] = 1  # 将所有能做的操作对应的id置为1
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
    :author: 毛芊蕙
    """
    action_type = battle_info_item["action_type"]  # 获取当前玩家操作动作
    if action_type == "G":  # 如果当前决策是摸牌，则表示要制作的标签是过操作
        return "pass"
    operate_card = battle_info_item["operate_card"]  # 获取操作牌
    combine_cards = battle_info_item["combine_cards"]  # 获取经过这个操作后形成的组合
    operate_card_label = translate3(operate_card)  # 将16进制牌转成0-27的牌
    idx = combine_cards.index(operate_card)  # 获取该操作牌在形成的组合的位置，通过位置判断是什么吃
    if idx == 0:
        action = f"left_chow_{operate_card_label}"  # 左吃
    elif idx == 1:
        action = f"mid_chow_{operate_card_label}"  # 中吃
    else:
        action = f"right_chow_{operate_card_label}"  # 右吃
    return action


def generate_json(battle_info_item, king_card, pos, json_file_root, file_num_dict):  # 对每个x和y都生成一个json文件
    """
    用于生成json数据标签，生成的json文件字段如函数体中所示
    :param battle_info_item: 整场信息中一个回合的数据，它是没有经过位置调整的，其结构如上一个函数中一样
    :param king_card: 宝牌id
    :param pos: 调整好的玩家位置，保证 位置放置为[当前玩家id, 下家id, 对家id, 上家id]
    :param json_file_root: 要存放数据的位置
    :param file_num_dict: 用于控制处理文件个数
    :return: 不用管
    :author: 毛芊蕙
    """
    feature_json = {}  # 制作json数据
    feature_json["handcards"] = battle_info_item["handcards"][pos[0]]  # 获取手牌信息，pos[0]高玩
    feature_json["discards_real"] = convert_item(battle_info_item["discards_real"], pos)  # 获取各个玩家丢弃的手牌（通过convert_item调整后的）
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

    # 下面代码可看可不看，就是把上面做好的字典写入json文件中，主要太麻烦，不想写注释

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


def is_can_chow(last_discard, battle_info_item, high_score_player_pos):
    """
    用于判断当前高玩是否能执行吃操作
    :param last_discard: 上一个玩家丢弃的牌
    :param battle_info_item: 整场信息中一个回合的数据，结构同上
    :param high_score_player_pos: 高玩座位id
    :return: 高玩是否能执行吃操作， 能返回True，否则False
    :author: 毛芊蕙
    """
    if last_discard >= 27:  # 如果上一个玩家丢的不是非字牌，就不能吃返回false
        return False
    handcards = battle_info_item["handcards"][high_score_player_pos]  # 获取高手玩家的手牌
    if handcards.count(last_discard + 1) > 0 and handcards.count(last_discard + 2) > 0:  # 能左吃
        return True
    if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard + 1) > 0:  # 能中吃
        return True
    if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard - 2) > 0:  # 能右吃
        return True
    return False


def process_file(file_intput_root, file_name, file_output_root):
    """
    用于处理原始的麻将数据文件，生成对应的数据标签
    :param file_intput_root: 原始文件数据位置根目录
    :param file_name: 原始文件名
    :param file_output_root: 处理好的文件放置位置
    :return:
    :author: 毛芊蕙
    """
    df = open(os.path.join(file_intput_root, file_name), encoding="utf-8")  # 打开原始的麻将文件数据
    data = json.load(df)  # 使用json读取原始的麻将文件数据
    high_score_player_pos = data["players_id"].index(data["high_score_player_id"])  # 获取高手玩家的位置
    king_card = data["king_card"]  # 获取宝牌id
    battle_info = data["battle_info"]  # 获取对局的整场信息
    pos = get_pos(high_score_player_pos)  # 获取调整好的玩家id，[高玩id, 下家id, 对家id, 上家id]
    file_num_dict = {}  # 控制文件数量，不要管

    # 遍历整场信息，获取高手决策点
    for idx, item in enumerate(battle_info):
        battle_info_item = copy.deepcopy(item)  # 使用深拷贝，防止后面误操作修改原始数据
        # 吃牌
        # 情况一，下一个玩家是高玩，且是吃牌
        if idx < len(battle_info) - 1 and battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1][
            "action_type"] == "C":
            battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
            battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
            # 为什么上面代码没有判断高手玩家上一个玩家是否是高玩的上家呢？因为如果是高手玩家吃牌的话上一个操作一定是高玩上家丢牌
            # 制作吃牌标签
            file_num_dict = generate_json(battle_info_item_, king_card, pos, file_output_root, file_num_dict)

        # 情况二，当前玩家为高玩上家且为丢牌，下一个玩家为高玩摸牌
        if idx < len(battle_info) - 1 and item["seat_id"] == pos[3] and item["action_type"] == "d":
            if battle_info[idx + 1]["seat_id"] == pos[0] and battle_info[idx + 1]["action_type"] == "G":
                battle_info_item_ = copy.deepcopy(battle_info[idx + 1])  # 获取高玩对战信息
                battle_info_item_["last_discard"] = item["operate_card"]  # 上一个玩家丢的牌(相对高玩来说)
                if is_can_chow(battle_info_item_["last_discard"], battle_info_item_, pos[0]):
                    # 制作不吃（过）标签
                    file_num_dict = generate_json(battle_info_item_, king_card, pos, file_output_root, file_num_dict)


if __name__ == '__main__':
    file_root = "D:\\project\\python\\high_score_json"
    dir_name = os.listdir(file_root)
    for item in dir_name:
        file_label_path = os.path.join(file_root, item)
        file_name = os.listdir(file_label_path)
        for it in file_name:
            process_file(file_label_path, it, r"..\data\output\chow")
        print(f"{item}文件完成！")
