from model import Model_v1
import torch
import discard.data_process as dis_p
import chow.data_process as ch_p
import gang.data_process as ga_p
import peng.data_process as pe_p
import hu.data_process as hu_p
import copy
import random
from feature_extract import *


class Player:
    # 导入丢牌模型
    discard_model = Model_v1(71, 34)
    discard_model.load_state_dict(
        torch.load("D:\\project\\python\\MJDecisionAI-DeepLearn\\discard\\mode_param\\ep2_acc=0.6585.pth"))

    # 导入吃牌模型
    chow_model = Model_v1(71, 64)
    chow_model.load_state_dict(
        torch.load("D:\\project\\python\\MJDecisionAI-DeepLearn\\chow\\mode_param\\ep5_acc=0.8273.pth"))

    # 导入碰牌模型
    peng_model = Model_v1(71, 35)
    peng_model.load_state_dict(
        torch.load("D:\\project\\python\\MJDecisionAI-DeepLearn\\peng\\mode_param\\ep0_acc=0.9776.pth"))

    # 导入杠牌模型
    gang_model = Model_v1(71, 103)
    gang_model.load_state_dict(
        torch.load("D:\\project\\python\\MJDecisionAI-DeepLearn\\gang\\mode_param\\ep0_acc=0.9996.pth"))

    # 导入胡牌模型
    hu_model = Model_v1(71, 2)
    hu_model.load_state_dict(
        torch.load("D:\\project\\python\\MJDecisionAI-DeepLearn\\hu\\mode_param\\ep25_acc=0.8152.pth"))

    def __init__(self, handcards, discards_real=None, fulu_chow=None, fulu_peng=None, fulu_kong=None):
        if fulu_kong is None:
            fulu_kong = []
        if fulu_peng is None:
            fulu_peng = []
        if fulu_chow is None:
            fulu_chow = []
        if discards_real is None:
            discards_real = []
        self.handcards = handcards  # 玩家手牌
        self.discards_real = discards_real  # 玩家弃牌集合
        self.fulu_chow = fulu_chow  # 玩家的顺子
        self.fulu_peng = fulu_peng  # 玩家的刻子
        self.fulu_kong = fulu_kong  # 玩家的杠子

    def zip_data_mask(self, status):
        feature_data = copy.deepcopy(status)
        # 制作弃牌行为掩码
        feature_data["discard_mask"] = torch.tensor(dis_p.get_action_mask(feature_data)).reshape((1, -1))
        # if feature_data["discard_mask"].count(1) == 0:  # 如果不能丢弃任何牌，那就剥夺该决策
        #     feature_data["action_types"].remove("discard")

        # 制作吃牌行为掩码
        feature_data["chow_mask"] = torch.tensor(ch_p.get_action_mask(feature_data)).reshape((1, -1))
        # if feature_data["chow_mask"].count(1) == 1:  # 如果不能吃任何牌，那就剥夺该决策
        #     feature_data["action_types"].remove("chow")

        # 制作碰牌行为掩码
        feature_data["peng_mask"] = torch.tensor(pe_p.get_action_mask(feature_data)).reshape((1, -1))
        # if feature_data["peng_mask"].count(1) == 1:  # 如果不能碰任何牌，那就剥夺该决策
        #     feature_data["action_types"].remove("peng")

        # 制作杠牌行为掩码
        feature_data["gang_mask"] = torch.tensor(ga_p.get_action_mask(feature_data)).reshape((1, -1))
        # if feature_data["gang_mask"].count(1) == 1:  # 如果不能杠任何牌，那就剥夺该决策
        #     feature_data["action_types"].remove("gang")

        # 制作胡牌行为掩码
        feature_data["hu_mask"] = torch.tensor(hu_p.get_action_mask(feature_data)).reshape((1, -1))
        return feature_data

    def get_actions(self, status):
        # 用于存储可以执行的操作，discard_action_list：用于存储可以执行弃牌的行为决策， chow_action_list：用于存储可以执行吃牌的行为决策
        # peng_action_list: 用于存储可以执行碰牌的行为决策, gang_action_list: 用于存储可以执行杠牌的行为决策
        actions = {"discard_action_list": [], "chow_action_list": [], "peng_action_list": [], "gang_action_list": [],
                   "pass": [], "hu": []}
        feature_ = self.zip_data_mask(status)  # 制作行为掩码，为了后面选取决策动作
        discard_mask = [item.item() for item in feature_["discard_mask"][0]]
        chow_mask = [item.item() for item in feature_["chow_mask"][0]]
        peng_mask = [item.item() for item in feature_["peng_mask"][0]]
        gang_mask = [item.item() for item in feature_["gang_mask"][0]]
        hu_mask = [item.item() for item in feature_["hu_mask"][0]]
        action_types = feature_["action_types"]
        if "discard" in action_types:  # 如果该状态下可以进行丢牌操作，则将可以执行的丢牌操作放入决策集合中
            for idx, item in enumerate(discard_mask):
                if item == 1:  # 说明该操作可以执行
                    temp = dis_p.id_to_action[idx].split("_")  # 通过id获取操作类型，使用_分割字符串获取行为和操作牌
                    action = temp[0]  # 获取行为
                    operate_card = translate33_16(int(temp[1]))  # 获取操作牌
                    actions["discard_action_list"].append((action, operate_card))

        if len(actions["discard_action_list"]) == 0:  # 说明不能执行丢牌操作，那么可以执行吃碰杠胡操作，那么一定有的操作就是pass
            actions["pass"].append(("pass", 255))

        if "chow" in action_types and chow_mask.count(1) != 1:  # 如果该状态下可以进行吃牌操作，则将可以执行的吃牌操作放入决策集合中
            # 如果行为掩码中就只有一个1，则表示不能执行任何吃牌操作（因为pass一定是1）
            for idx, item in enumerate(chow_mask):
                if item == 1 and idx != 0:  # 如果该操作可以执行且不是pass
                    temp = ch_p.id_to_action[idx].split("_")  # 通过id获取操作类型，使用_分割字符串获取行为和操作牌
                    action = temp[0] + "_" + temp[1]  # 获取行为
                    operate_card = translate33_16(int(temp[2]))  # 获取操作牌
                    actions["chow_action_list"].append((action, operate_card))

        if "peng" in action_types and peng_mask.count(1) != 1:  # 如果该状态下可以进行碰牌操作，则将可以执行的碰牌操作放入决策集合中
            # 如果行为掩码中就只有一个1，则表示不能执行任何碰牌操作（因为pass一定是1）
            for idx, item in enumerate(peng_mask):
                if item == 1 and idx != 0:  # 如果该操作可以执行且不是pass
                    temp = pe_p.id_to_action[idx].split("_")  # 通过id获取操作类型，使用_分割字符串获取行为和操作牌
                    action = temp[0]  # 获取行为
                    operate_card = translate33_16(int(temp[1]))  # 获取操作牌
                    actions["peng_action_list"].append((action, operate_card))

        if "gang" in action_types and gang_mask.count(1) != 1:  # 如果该状态下可以进行杠牌操作，则将可以执行的杠牌操作放入决策集合中
            # 如果行为掩码中就只有一个1，则表示不能执行任何碰牌操作（因为pass一定是1）
            for idx, item in enumerate(gang_mask):
                if item == 1 and idx != 0:  # 如果该操作可以执行且不是pass
                    temp = ga_p.id_to_action[idx].split("_")  # 通过id获取操作类型，使用_分割字符串获取行为和操作牌
                    action = temp[0] + "_" + temp[1]  # 获取行为
                    operate_card = translate33_16(int(temp[2]))  # 获取操作牌
                    actions["gang_action_list"].append((action, operate_card))

        if "hu" in action_types and hu_mask.count(1) != 1:  # 如果该状态下可以进行胡牌操作，则将可以执行的胡牌操作放入决策集合中
            # 如果行为掩码中就只有一个1，则表示不能执行胡牌操作（因为pass一定是1）
            for idx, item in enumerate(hu_mask):
                if item == 1 and idx != 0:  # 如果该操作可以执行且不是pass
                    temp = hu_p.id_to_action[idx].split("_")  # 通过id获取操作类型，使用_分割字符串获取行为和操作牌
                    action = temp[0]  # 获取行为
                    actions["hu"].append((action, 255))

        return actions

    def get_actions_list(self, status):
        actions = self.get_actions(status)
        actions_list = []  # 存储能执行的一系列操作，为了下面的判断
        for key in actions.keys():
            if len(actions[key]) == 0:
                continue
            for item in actions[key]:
                actions_list.append(item)
        return actions_list

    def play_by_human(self, status):
        actions = self.get_actions(status)  # 获取当前状态下能执行的操作
        actions_list = []  # 存储能执行的一系列操作，为了下面的判断
        print("------当前你能执行的操作------")
        for key in actions.keys():
            if len(actions[key]) == 0:
                continue
            for item in actions[key]:
                actions_list.append(item)
                print(item)  # 打印能执行的操作

        # print(actions_list)
        while True:
            action = input("请输入你选择的操作:")
            operate_card = int(input("请输入你选择的操作牌:"))
            if (action, operate_card) not in actions_list:
                print("你选择的操作有误，请重新输入")
            else:
                break

        if action in ["hu", "pass"]:  # 如果选择的行为是胡牌或者是过操作的话就直接放回决策（返回值一定要和ai返回值一样）
            return action

        return action + "_" + str(translate3(operate_card))

    def play_by_ai(self, status):
        action = []
        feature_ = self.zip_data_mask(status)
        action_types = feature_["action_types"]
        x1, x2 = get_feature(feature_)
        x1 = x1.reshape((1, 71, 3, 9))
        x2 = x2.reshape((1, 71, 1, 7))
        for item in action_types:
            if item == "discard":  # 如可以丢牌
                res = Player.discard_model.predict(x1, x2, feature_["discard_mask"])
                idx = res.argmax(dim=1).item()
                sorce = res[0][idx].item()  # 预测行为的分数
                ac = dis_p.id_to_action[idx]  # 预测行为
                action.append((ac, sorce))

            if item == "chow":  # 如果可以吃牌
                res = Player.chow_model.predict(x1, x2, feature_["chow_mask"])
                idx = res.argmax(dim=1).item()
                sorce = res[0][idx].item()  # 预测行为的分数
                ac = ch_p.id_to_action[idx]  # 预测行为
                action.append((ac, sorce))

            if item == "peng":  # 如果可以碰牌
                res = Player.peng_model.predict(x1, x2, feature_["peng_mask"])
                idx = res.argmax(dim=1).item()
                sorce = res[0][idx].item()  # 预测行为的分数
                ac = pe_p.id_to_action[idx]  # 预测行为
                action.append((ac, sorce))

            if item == "gang":  # 如果可以杠牌
                res = Player.gang_model.predict(x1, x2, feature_["gang_mask"])
                idx = res.argmax(dim=1).item()
                sorce = res[0][idx].item()  # 预测行为的分数
                ac = ga_p.id_to_action[idx]  # 预测行为
                action.append((ac, sorce))

            if item == "hu":  # 如果可以胡牌
                res = Player.hu_model.predict(x1, x2, feature_["hu_mask"])
                idx = res.argmax(dim=1).item()
                sorce = res[0][idx].item()  # 预测行为的分数
                ac = hu_p.id_to_action[idx]  # 预测行为
                action.append((ac, sorce))


        res_action_list = []  # 用于存储除了过操作的决策行为
        for item in action:  # 在所有的行为决策中找到不是过的决策行为
            if item[0] != "pass":
                res_action_list.append(item)

        # if quick_hu:  # 快速胡牌模式
        #     for item in action:  # 在所有的行为决策中找到不是过的决策行为
        #         if item[0] == "hu":
        #             return "hu"


        if len(res_action_list) == 0:  # 如果所有的决策都是过的话就返回过
            return "pass"

        for idx, item in enumerate(res_action_list):  # 分数的前缀和，用于后面进行随机决策
            if idx > 0:
                res_action_list[idx] = (res_action_list[idx][0], res_action_list[idx][1] + res_action_list[idx - 1][1])

        # 所有的决策分数总和
        sum_sorce = res_action_list[len(res_action_list) - 1][1]

        # 生成随机数，用于进行随机决策
        x = min(random.random() * sum_sorce, sum_sorce)
        res_action_id = len(res_action_list) - 1  # 存储选择到的决策索引
        for idx, item in enumerate(res_action_list):
            if x <= item[1]:  # 如属于该范围就就选择该行为
                res_action_id = idx
                break

        return res_action_list[res_action_id][0]  # 返回随机到的决策

    def get_fulu(self):
        return self.fulu_kong + self.fulu_chow + self.fulu_peng

    def update_status_by_op(self, action, operate_card):
        part_act = action.split("_")  # 分割行为字符串
        part_act.append(str(translate3(operate_card)))
        if len(part_act) == 2 and part_act[0] == "discard":  # 该行为为弃牌
            operate_card = translate33_16(int(part_act[1]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除丢弃的牌
            self.discards_real.append(operate_card)  # 弃牌集合中加入丢弃的牌
            return "discard", operate_card

        if len(part_act) == 2 and part_act[0] == "peng":  # 该行为为碰牌
            operate_card = translate33_16(int(part_act[1]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除组成刻子的牌
            self.handcards.remove(operate_card)  # 手牌中移除组成刻子的牌
            self.fulu_peng.append([operate_card, operate_card, operate_card])  # 加入碰牌集合
            return "peng", operate_card

        if len(part_act) == 3 and part_act[0] == "bu" and part_act[1] == "gang":  # 该行为为补杠
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除补杠的牌
            self.fulu_peng.remove([operate_card, operate_card, operate_card])  # 移除碰副露中的牌
            self.fulu_kong.append([operate_card, operate_card, operate_card, operate_card])  # 杠牌副露中加入杠的牌
            return "gang", operate_card

        if len(part_act) == 3 and part_act[0] == "ming" and part_act[1] == "gang":  # 该行为为明杠
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除明杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除明杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除明杠的牌
            self.fulu_kong.append([operate_card, operate_card, operate_card, operate_card])  # 杠牌副露中加入杠的牌
            return "gang", operate_card

        if len(part_act) == 3 and part_act[0] == "an" and part_act[1] == "gang":  # 该行为为暗杠
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.fulu_kong.append([operate_card, operate_card, operate_card, operate_card])  # 杠牌副露中加入杠的牌
            return "gang", operate_card

        if len(part_act) == 3 and part_act[0] == "left" and part_act[1] == "chow":  # 该行为为左吃
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card + 1)  # 从手牌中移除组成顺子的牌
            self.handcards.remove(operate_card + 2)  # 从手牌中移除组成顺子的牌
            self.fulu_chow.append([operate_card, operate_card + 1, operate_card + 2])  # 吃牌副露中加入顺子
            return part_act[0] + "_" + part_act[1], operate_card

        if len(part_act) == 3 and part_act[0] == "mid" and part_act[1] == "chow":  # 该行为为中吃
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card - 1)  # 从手牌中移除组成顺子的牌
            self.handcards.remove(operate_card + 1)  # 从手牌中移除组成顺子的牌
            self.fulu_chow.append([operate_card - 1, operate_card, operate_card + 1])  # 吃牌副露中加入顺子
            return part_act[0] + "_" + part_act[1], operate_card

        if len(part_act) == 3 and part_act[0] == "right" and part_act[1] == "chow":  # 该行为为右吃
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card - 1)  # 从手牌中移除组成顺子的牌
            self.handcards.remove(operate_card - 2)  # 从手牌中移除组成顺子的牌
            self.fulu_chow.append([operate_card - 2, operate_card - 1, operate_card])  # 吃牌副露中加入顺子
            return part_act[0] + "_" + part_act[1], operate_card

        if len(part_act) == 1 and part_act[0] == "hu":  # 该行为为胡牌
            return "hu", 255

        return "pass", 255

    def update_status(self, status, is_human=False):
        action = None
        if is_human:
            action = self.play_by_human(status)  # 获取人的决策
        else:
            action = self.play_by_ai(status)  # 获取神经网络输出
        part_act = action.split("_")  # 分割行为字符串
        if len(part_act) == 2 and part_act[0] == "discard":  # 该行为为弃牌
            operate_card = translate33_16(int(part_act[1]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除丢弃的牌
            self.discards_real.append(operate_card)  # 弃牌集合中加入丢弃的牌
            return "discard", operate_card

        if len(part_act) == 2 and part_act[0] == "peng":  # 该行为为碰牌
            operate_card = translate33_16(int(part_act[1]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除组成刻子的牌
            self.handcards.remove(operate_card)  # 手牌中移除组成刻子的牌
            self.fulu_peng.append([operate_card, operate_card, operate_card])  # 加入碰牌集合
            return "peng", operate_card

        if len(part_act) == 3 and part_act[0] == "bu" and part_act[1] == "gang":  # 该行为为补杠
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除补杠的牌
            self.fulu_peng.remove([operate_card, operate_card, operate_card])  # 移除碰副露中的牌
            self.fulu_kong.append([operate_card, operate_card, operate_card, operate_card])  # 杠牌副露中加入杠的牌
            return "gang", operate_card

        if len(part_act) == 3 and part_act[0] == "ming" and part_act[1] == "gang":  # 该行为为明杠
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除明杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除明杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除明杠的牌
            self.fulu_kong.append([operate_card, operate_card, operate_card, operate_card])  # 杠牌副露中加入杠的牌
            return "gang", operate_card

        if len(part_act) == 3 and part_act[0] == "an" and part_act[1] == "gang":  # 该行为为暗杠
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.handcards.remove(operate_card)  # 手牌中移除暗杠的牌
            self.fulu_kong.append([operate_card, operate_card, operate_card, operate_card])  # 杠牌副露中加入杠的牌
            return "gang", operate_card

        if len(part_act) == 3 and part_act[0] == "left" and part_act[1] == "chow":  # 该行为为左吃
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card + 1)  # 从手牌中移除组成顺子的牌
            self.handcards.remove(operate_card + 2)  # 从手牌中移除组成顺子的牌
            self.fulu_chow.append([operate_card, operate_card + 1, operate_card + 2])  # 吃牌副露中加入顺子
            return part_act[0] + "_" + part_act[1], operate_card

        if len(part_act) == 3 and part_act[0] == "mid" and part_act[1] == "chow":  # 该行为为中吃
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card - 1)  # 从手牌中移除组成顺子的牌
            self.handcards.remove(operate_card + 1)  # 从手牌中移除组成顺子的牌
            self.fulu_chow.append([operate_card - 1, operate_card, operate_card + 1])  # 吃牌副露中加入顺子
            return part_act[0] + "_" + part_act[1], operate_card

        if len(part_act) == 3 and part_act[0] == "right" and part_act[1] == "chow":  # 该行为为右吃
            operate_card = translate33_16(int(part_act[2]))  # 获取操作牌，并转换为16进制手牌
            self.handcards.remove(operate_card - 1)  # 从手牌中移除组成顺子的牌
            self.handcards.remove(operate_card - 2)  # 从手牌中移除组成顺子的牌
            self.fulu_chow.append([operate_card - 2, operate_card - 1, operate_card])  # 吃牌副露中加入顺子
            return part_act[0] + "_" + part_act[1], operate_card

        if len(part_act) == 1 and part_act[0] == "hu":  # 该行为为胡牌
            return "hu", 255

        return "pass", 255

    def show_data(self):
        print("手牌: " + str(sorted(self.handcards)))
        print("弃牌: " + str(self.discards_real))
        print("顺子: " + str(self.fulu_chow))
        print("刻子: " + str(self.fulu_peng))
        print("杠子: " + str(self.fulu_kong))


if __name__ == '__main__':
    data = {
    "handcards": [
        17,
        17,
        17,
        18,
        18,
        18,
        18,
        19,
        20,
        34,
        24
    ],
    "discards_real": [
        [
            25,
            53,
            1,
            38,
            40
        ],
        [
            53,
            51,
            1,
            25,
            9,
            7
        ],
        [
            55,
            54,
            40,
            20,
            8,
            41
        ],
        [
            55,
            9,
            35,
            8,
            21,
            51
        ]
    ],
    "fulu_chow": [
        [
            [
                6,
                7,
                8
            ]
        ],
        [],
        [
            [
                5,
                6,
                7
            ]
        ],
        []
    ],
    "fulu_peng": [
        [
            [
                24,
                24,
                24
            ]
        ],
        [],
        [],
        []
    ],
    "fulu_kong": [
        [],
        [],
        [],
        []
    ],
    "pos": [
        1,
        2,
        3,
        0
    ],
    "king_card": 49,
    "round": 7,
    "last_discard": 255,
    "operate_card": 255,
    "action_type": "A",
    "action_mask": [
        1,
        0
    ],
    "label": "hu"
}
    feature = {
        "handcards": data["handcards"],
        "king_card": data["king_card"],
        "round": data["round"]
    }
    feature["discards_real"] = data["discards_real"]
    feature["fulu_chow"], feature["fulu_peng"], feature["fulu_kong"] = data["fulu_chow"], data["fulu_peng"], data["fulu_kong"]
    feature["last_discard"] = data["last_discard"]
    feature["action_types"] = ["hu", "discard", "gang"]
    player = Player(feature["handcards"], feature["discards_real"][0], feature["fulu_chow"][0],
                    feature["fulu_peng"][0],
                    feature["fulu_kong"][0])
    player.show_data()
    action, operate_card = player.update_status(feature, True)
    print(f"{action}: {operate_card}")
    player.show_data()
