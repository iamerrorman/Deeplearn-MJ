import random
import time

from player import Player
from base_data_process import *
from majiang_fina_2l import  *
import chow.data_process as ch_p
import gang.data_process as ga_p
import peng.data_process as pe_p


class Game:
    def __init__(self):
        super().__init__()
        self.tiles = []
        # 生成牌墙数据
        for i in range(1, 10):
            self.tiles += [i, i, i, i]
            self.tiles += [i + 16, i + 16, i + 16, i + 16]
            self.tiles += [i + 32, i + 32, i + 32, i + 32]
        for i in range(1, 8):
            self.tiles += [i + 48, i + 48, i + 48, i + 48]
        random.shuffle(self.tiles)  # 打乱牌墙数据

        # 随机生成宝牌信息
        lists = [random.randint(1, 9), random.randint(17, 25), random.randint(33, 41), random.randint(49, 55)]
        random.shuffle(lists)
        self.king_card = lists[0]

        # 庄家id
        self.leader_id = random.randint(0, 3)

        # 生成玩家信息
        self.player_s = [None, None, None, None]
        for i in range(0, 4):
            if self.leader_id == i:
                self.player_s[i] = Player(self.tiles[0: 14])
                self.tiles = self.tiles[14: len(self.tiles)]
            else:
                self.player_s[i] = Player(self.tiles[0: 13])
                self.tiles = self.tiles[13: len(self.tiles)]

        # 轮次信息
        self.round = 0

        # 当前决策玩家
        self.current_id = self.leader_id

        # 当前玩家的决策种类
        self.action_types = ["discard", "hu", "gang"]

        # 上一个玩家丢弃的牌
        self.last_discard = 255

        # 获胜玩家id
        self.win = -1

        # 是否是流局
        self.liu = False

        # 被人类操控的玩家id
        self.human_player_id = None

    def zip_feature(self, current_id):
        feature = {
            "handcards": self.player_s[current_id].handcards,
            "king_card": self.king_card,
            "round": self.round,
        }
        pos = get_pos(current_id)
        feature["discards_real"] = convert_item([self.player_s[i].discards_real for i in range(0, 4)], pos)
        feature["fulu_chow"], feature["fulu_peng"], feature["fulu_kong"] = get_fulu_classify(
            [self.player_s[i].get_fulu() for i in range(0, 4)], pos)  # 获取各个玩家副露
        feature["last_discard"] = self.last_discard
        feature["action_types"] = self.action_types
        return feature

    def is_peng(self, play_id, last_discard):
        handcards = self.player_s[play_id].handcards
        if handcards.count(last_discard) >= 2:  # 能碰
            return True
        return False

    def is_gang(self, play_id, last_discard):
        handcards = self.player_s[play_id].handcards
        if handcards.count(last_discard) == 3:  # 能杠
            return True
        return False

    def is_chow(self, play_id, last_discard):

        if last_discard >= 27:
            return False
        handcards = self.player_s[play_id].handcards  # 获取玩家的手牌
        if handcards.count(last_discard + 1) > 0 and handcards.count(last_discard + 2) > 0:  # 能左吃
            return True
        if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard + 1) > 0:  # 能中吃
            return True
        if handcards.count(last_discard - 1) > 0 and handcards.count(last_discard - 2) > 0:  # 能右吃
            return True

        return False

    def G_pai(self, player_id):
        if len(self.tiles) == 0:
            self.liu = True
        else:
            get_tile = self.tiles.pop(0)
            self.player_s[player_id].handcards.append(get_tile)

    def other_peng_gang_chow(self, last_discard):
        id_action = {0: [], 1: [], 2: [], 3: []}  # 存储每个玩家在当前别人丢弃牌之后的决策动作
        for i in range(0, 4):
            if i == self.current_id:  # 如果是弃牌人，就不做考虑
                continue
            if self.current_id == (i + 3) % 4 and self.is_chow(i, last_discard):  # 如果该玩家是弃牌者的下家，他也能执行吃操作
                id_action[i].append("chow")
            if self.is_peng(i, last_discard):
                id_action[i].append("peng")
            if self.is_gang(i, last_discard):
                id_action[i].append("gang")

        ids = list(id_action.keys())
        random.shuffle(ids)  # 为了能模拟真实游戏环境，需要打乱位置

        for item in ids:
            if len(id_action[item]) == 0:  # 如果不能做任何动作，就下一位
                continue

            feature = self.zip_feature(item)  # 如果能做决策，就获取基于该玩家的游戏状态
            feature["last_discard"] = last_discard
            feature["action_types"] = id_action[item]

            if self.human_player_id == item:  # 如果当前玩家为人类玩家
                self.player_s[item].show_data()
                action, operate_card = self.player_s[item].update_status(feature, is_human=True)  # 做出决策并更新状态
                # self.player_s[item].update_status(action, operate_card)  # 通过决策和决策牌更新状态
            else:
                action, operate_card = self.player_s[item].update_status(feature)  # 做出决策并更新状态

            if action == "pass":  # 如果该玩家还是选择了过，就下一位
                continue
            else:
                # 如果已经有人做出决策，那么下一个操作就是该玩家丢牌，则需要把当前玩家指向该玩家，行为决策也应该转变
                # 并且上一个玩家丢弃的牌应该从他的弃牌集合中移除
                self.player_s[self.current_id].discards_real.remove(last_discard)
                self.current_id = item
                if action == "gang":  # 如果该玩家决策为杠牌的话， 则需要摸一张牌
                    self.G_pai(item)
                    # 摸牌后可以进行的操作为弃牌，胡牌，杠牌
                    self.action_types = ["discard", "hu", "gang"]
                else:  # 除了杠牌的其他操作后就只能弃牌
                    self.action_types = ["discard"]
                self.show_data(self.current_id, action, operate_card)
                # self.update_ui_sleep(self.zip_total_data(), 0, False)  # 更新界面， 0为全展开
                return item, action, operate_card  # 已经有人做出决策动作，返回决策人、决策动作和操作牌

        return None, "pass", None  # 没有人做出决策动作，返回过

    # def update_ui_sleep(self, kwargs, AiOrPeople, is_sleep=True):
    #
    #     self.update(kwargs, AiOrPeople)
    #     if is_sleep:
    #         time.sleep(2)

    def add_humans(self):
        while True:
            id = int(input("请输入你要控制的0到3的玩家id:"))
            if id not in [0, 1, 2, 3]:
                print("你输入的id不合法")
            else:
                self.human_player_id = id
                break

    def start(self):
        while True:
            self.show_global_data()
            if self.liu:
                print("该对局为流局！")
                break
            # print(self.zip_total_data())
            # self.update_ui_sleep(self.zip_total_data(), 0)  # 更新界面， 0为全展开
            if self.current_id == self.leader_id:
                self.round += 1

            feature_ = self.zip_feature(self.current_id)
            if self.current_id == self.human_player_id:
                self.player_s[self.current_id].show_data()
                action, operate_card = self.player_s[self.current_id].update_status(feature_, is_human=True)
            else:
                action, operate_card = self.player_s[self.current_id].update_status(self.zip_feature(self.current_id))

            if action == "hu":  # 如果当前玩家决策为胡牌就结束游戏，设置游戏胜利者为当前玩家
                self.win = self.current_id
                self.show_data(self.current_id, action, operate_card)
                break

            if action == "discard":  # 如果当前玩家决策为丢牌，则需要看看其它玩家的决策
                self.show_data(self.current_id, action, operate_card)
                # self.update_ui_sleep(self.zip_total_data(), 0, False)  # 更新界面， 0为全展开
                n_player, n_aciton, n_operate_card = self.other_peng_gang_chow(operate_card)
                if n_aciton == "pass":  # 如果其他玩家不进行任何操作的话，游戏按顺时针继续
                    self.current_id = (self.current_id + 1) % 4
                    self.G_pai(self.current_id)
                    self.action_types = ["discard", "hu", "gang"]

            if action == "gang":  # 如果当前玩家为杠牌操作， 则该玩家需要摸一张牌
                self.show_data(self.current_id, action, operate_card)
                # self.update_ui_sleep(self.zip_total_data(), 0, False)  # 更新界面， 0为全展开
                self.G_pai(self.current_id)

    def show_global_data(self):
        print("-----------全局属性-----------")
        print(f"宝牌: {self.king_card}")
        print(f"庄家id: {self.leader_id}")

    def show_data(self, player_id, action, operate_card):  # 打印对局信息
        self.show_global_data()
        print(f"当前轮次: {self.round}")
        print("-----------玩家属性-----------")
        print("-----------0号玩家------------")
        self.player_s[0].show_data()
        print("-----------1号玩家------------")
        self.player_s[1].show_data()
        print("-----------2号玩家------------")
        self.player_s[2].show_data()
        print("-----------3号玩家------------")
        self.player_s[3].show_data()
        print("-----------------------------")
        print("-----------当前操作------------")
        print(f"操作者: {player_id}号玩家")
        print(f"决策动作: {action}")
        print(f"操作牌: {operate_card}")
        print("=============================")
        print()

    def zip_total_data(self, player_id=None, action=None, operate_card=None):
        zip_data = {
            "king_card": self.king_card,   # 宝牌
            "round": self.round,  # 当前是第几轮。目前定的是每次到庄家出牌则round+1
            "zhuang_id": self.leader_id,  # 庄家座位id
            "current_id": self.current_id,  # 当前玩家
            "player_id": player_id,
            "action": action,
            "operate_card": operate_card,
            "play_0": {
                "handcards": self.player_s[0].handcards,  # 玩家的手牌
                "discards": self.player_s[0].discards_real,  # 玩家的出牌(不包括被别人吃碰杠的牌)
                "fulu": self.player_s[0].get_fulu(),  # 所有玩家的副露
            },
            "play_1": {
                "handcards": self.player_s[1].handcards,  # 玩家的手牌
                "discards": self.player_s[1].discards_real,  # 玩家的出牌(不包括被别人吃碰杠的牌)
                "fulu": self.player_s[1].get_fulu(),  # 所有玩家的副露
            },
            "play_2": {
                "handcards": self.player_s[2].handcards,  # 玩家的手牌
                "discards": self.player_s[2].discards_real,  # 玩家的出牌(不包括被别人吃碰杠的牌)
                "fulu": self.player_s[2].get_fulu(),  # 所有玩家的副露
            },
            "play_3": {
                "handcards": self.player_s[3].handcards,  # 玩家的手牌
                "discards": self.player_s[3].discards_real,  # 玩家的出牌(不包括被别人吃碰杠的牌)
                "fulu": self.player_s[3].get_fulu(),  # 所有玩家的副露
            }
        }
        # print(zip_data)
        return zip_data


if __name__ == '__main__':
    game = Game()
    # game.add_humans()
    print(game)
    game.start()








