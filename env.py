import torch
import random
import csv
import numpy as np
from driver import Driver
from order import Order


class Env:
    def __init__(self, args):
        self.args = args
        self.time_slot = 0
        self.max_time_slot = args.episode_limit  # max number of steps
        self.region_graph = args.region_graph

        self.use_gnn = args.use_gnn

        self.n = args.agent_n  # 智能体的数量
        self.region_n = args.region_n  # 地图上划分的区域数量

        # 定义维度
        self.observation_space = 8  # [7 for _ in range(agent_n)]
        self.action_space = 10  # [10 for _ in range(agent_n)]
        #

        self.drivers = []  # a list to record all drivers
        self.orders = []  # a list to record all orders
        self.PoI_data = None
        self.AoI_data = None  # an array,used to record the AoI of PoI data

        self.regions_neighbor = []  # 保存每一个区域的邻居
        for i in range(self.region_n):
            neigh = []
            index = 0
            for node_index in self.region_graph.edges()[0]:
                if node_index.item() == i:
                    neigh.append(self.region_graph.edges()[1][index].item())
                index += 1
            self.regions_neighbor.append(neigh.copy())

        # 一些需要统计的数据
        self.order_number = 0  # 目前已产生的订单数量
        self.accepted_order_number = 0  # 已经被接受的order数量
        self.overdue_order_number = 0  # 过期的order数量
        self.data_collected = 0  # 收集到的信息量
        self.ava_AoI = 0  # 收集到的数据的平均信息年龄
        self.data_q = 0  # 收集到的数据的质量
        #

        # 一些参数
        self.collect_unit = 100  # 每次可以收集100MB数据
        #

        # 加载order数据集
        self.order_dataset = []
        with open(args.order_data_path) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                self.order_dataset.append(row)  # 选择某一列加入到data数组中
            del (self.order_dataset[0])
        #

    def init_poi_data(self):
        """
        初始化POI数据和信息年龄
        :return:
        """
        self.PoI_data = np.zeros(self.region_n)  # 一个numpy数组
        self.PoI_data[3] = 20000
        self.PoI_data[2] = 11000
        self.PoI_data[13] = 25000
        self.PoI_data[16] = 15000
        self.PoI_data[23] = 10000
        self.PoI_data[28] = 20000
        self.PoI_data[30] = 22000
        self.PoI_data[25] = 9000
        self.AoI_data = np.ones(self.region_n)

    def init_drivers(self):
        """
        初始化若干个司机
        :return:
        """
        for i in range(self.n):
            init_location = np.random.randint(0, self.region_n)
            self.drivers.append(Driver(driver_id=i, init_location=init_location))

    def init_orders(self):
        # 先随机初始化若干个订单
        number_of_orders = 50
        region_item = range(self.region_n)
        # prob = [0.01, 0.01, 0.06, 0.10, 0.15, 0.07, 0.02, 0.12, 0.05, 0.005, 0.02, 0.281, 0.04, 0.001, 0.01, 0.01,
        #         0.008, 0.012, 0.01, 0.013]
        prob = [0.02, 0.06, 0.08, 0.02, 0, 0, 0,
                0.08, 0.13, 0.1, 0, 0, 0.01, 0.1,
                0.1, 0.01, 0.03, 0, 0, 0, 0.02,
                0.06, 0.01, 0.03, 0, 0, 0.02, 0,
                0.03, 0.03, 0.01, 0, 0.05, 0, 0]
        region = np.random.choice(region_item, size=number_of_orders, replace=True, p=prob)
        for i, data in enumerate(region):
            travel_time = np.random.randint(4, 9)
            destination = np.random.randint(0, self.region_n)
            self.orders.append(
                Order(order_id=self.order_number, init_location=data,
                      generate_time=self.time_slot, waiting_time=15,
                      price=10, travel_time=travel_time,
                      destination=destination)
            )
            self.order_number += 1

    def create_orders(self):
        """
        根据order数据集在每个时隙产生order
        :return:
        """
        for data in self.order_dataset:
            s = data[2].split(' ')
            s1 = s[1].split(':')

            # 目前暂定地图每0.01经纬度进行划分，74.00~-73.86  40.70~40.80一共划分35个区域
            if (float(s1[0])) * 60 + float(s1[1]) == self.time_slot:  # 数据集中order的产生时间
                init_location = [float(data[5]), float(data[6])]
                drop_off_location = [float(data[7]), float(data[8])]
                if init_location[0] <= -74.00 or init_location[0] >= -73.86 or \
                        init_location[1] <= 40.70 or init_location[1] >= 40.80 or \
                        drop_off_location[0] <= -74.00 or drop_off_location[0] >= -73.86 or \
                        drop_off_location[1] <= 40.70 or drop_off_location[1] >= 40.80:
                    continue
                else:
                    init_region_location = int((init_location[0] + 74.00) / 0.02) + \
                                           int((40.80 - init_location[1]) / 0.02) * 7
                    drop_off_region_location = int((drop_off_location[0] + 74.00) / 0.02) + \
                                               int((40.80 - drop_off_location[1]) / 0.02) * 7
                    order_price = 0.02 * float(data[10])
                    travel_time_cost = float(data[10])
                    self.orders.append(
                        Order(order_id=self.order_number, init_location=init_region_location,
                              generate_time=self.time_slot, waiting_time=15, price=order_price,
                              travel_time=travel_time_cost, destination=drop_off_region_location)
                    )
                    self.order_number += 1

    def init_env(self):
        self.time_slot = 0
        self.init_drivers()
        self.init_poi_data()
        self.init_orders()
        self.create_orders()

        self.order_number = 0  # 目前已产生的订单数量
        self.accepted_order_number = 0  # 已经被接受的order数量
        self.overdue_order_number = 0  # 过期的order数量
        self.data_collected = 0  # 收集到的信息量
        self.ava_AoI = 0  # 收集到的数据的平均信息年龄
        self.data_q = 0

    def get_idle_order(self):
        """
        返回每个区域内的待服务的订单数量
        :return:
        """
        result = np.zeros(self.region_n)
        for order in self.orders:
            if not order.is_matched and not order.is_overdue:
                result[order.init_location] += 1
        return result

    def get_idle_driver(self):
        """
        返回每个区域内空闲的车辆数量
        :return:
        """
        result = np.zeros(self.region_n)
        for driver in self.drivers:
            if not driver.is_serving and not driver.is_dispatched and not driver.is_collecting_data:
                result[driver.now_location] += 1
        return result

    def get_data_vol(self):
        """
        返回每个区域的目前未采集的数据量
        :return:
        """
        return self.PoI_data

    def get_vehicle_observation(self):
        obs = np.zeros((self.n, self.observation_space))  # n * obs_dim
        idle_order = self.get_idle_order()
        idle_drivers = self.get_idle_driver()
        data_vol = self.get_data_vol()
        for i in range(self.n):
            obs[i, 0] = idle_order[self.drivers[i].now_location]  # driver i 所在区域的空闲订单数量
            obs[i, 1] = idle_drivers[self.drivers[i].now_location]  # driver i 所在区域的空闲车辆数量
            obs[i, 2] = data_vol[self.drivers[i].now_location]  # driver i 所在区域的PoI数据量
            obs[i, 3] = self.drivers[i].data_vol  # driver i 收集的数据量
            obs[i, 4] = i  # driver i 的编号
            obs[i, 5] = 1 if self.drivers[i].is_serving else 0  # driver i 是否正在服务订单
            obs[i, 6] = self.AoI_data[self.drivers[i].now_location]  # driver i 所在区域的数据信息年龄
            obs[i, 7] = self.drivers[i].now_location

        return obs

    def reset(self):
        self.time_slot = 0
        self.drivers.clear()
        self.orders.clear()

        self.init_env()

        self.accepted_order_number = 0  # 已经被接受的order数量
        self.overdue_order_number = 0  # 过期的order数量
        self.data_collected = 0  # 收集到的信息量
        self.ava_AoI = 0  # 收集到的数据的平均信息年龄
        self.data_q = 0
        return self.get_vehicle_observation()

    def driver_order_match(self):
        random.shuffle(self.drivers)
        random.shuffle(self.orders)

        # 还没有司机或者order的情况下 直接返回
        if len(self.drivers) == 0 or len(self.orders) == 0:
            return

        for driver_index in range(len(self.drivers)):
            if not self.drivers[driver_index].is_serving and \
                    not self.drivers[driver_index].is_dispatched and \
                    not self.drivers[driver_index].is_collecting_data:
                driver_location = self.drivers[driver_index].now_location
                for order_index in range(len(self.orders)):
                    if not self.orders[order_index].is_matched and \
                            not self.orders[order_index].is_overdue and \
                            self.orders[order_index].init_location == driver_location:
                        self.orders[order_index].driver_match(self.drivers[driver_index])
                        self.drivers[driver_index].order_match(self.orders[order_index], self.time_slot)
                        self.accepted_order_number += 1
                        break

        self.orders.sort(key=lambda x: x.order_id)
        self.drivers.sort(key=lambda x: x.driver_id)

    def take_action(self, action):
        reward = np.zeros(self.n)  # 采取动作后产生的reward

        for i, driver_action in enumerate(action):

            if driver_action == 0:
                # 为0表示不调度,也不收集数据
                continue
            else:
                if driver_action == (self.action_space - 1):
                    # 表示收集数据
                    if not self.drivers[i].is_serving and not self.drivers[i].is_dispatched and \
                            not self.drivers[i].is_collecting_data:

                        self.drivers[i].is_collecting_data = True

                        if self.PoI_data[self.drivers[i].now_location] >= self.collect_unit:
                            self.PoI_data[self.drivers[i].now_location] \
                                = self.PoI_data[self.drivers[i].now_location] - self.collect_unit
                            self.drivers[i].data_vol += self.collect_unit

                            reward[i] += self.args.beta * (
                                    self.collect_unit / float(self.AoI_data[self.drivers[i].now_location]))  # 数据收集量 / 信息年龄

                            self.data_collected += self.collect_unit

                            self.ava_AoI += self.AoI_data[self.drivers[i].now_location]
                            self.data_q += self.collect_unit / self.AoI_data[self.drivers[i].now_location]
                        else:
                            self.drivers[i].data_vol += self.PoI_data[self.drivers[i].now_location]

                            reward[i] += self.args.beta * (self.PoI_data[self.drivers[i].now_location] /
                                                           float(self.AoI_data[self.drivers[i].now_location]))

                            self.data_collected += self.PoI_data[self.drivers[i].now_location]
                            self.ava_AoI += self.AoI_data[self.drivers[i].now_location]
                            self.data_q += self.PoI_data[self.drivers[i].now_location] \
                                           / self.AoI_data[self.drivers[i].now_location]
                            self.PoI_data[self.drivers[i].now_location] = 0

                else:
                    # 进行调度
                    if not self.drivers[i].is_serving and \
                            not self.drivers[i].is_dispatched and \
                            not self.drivers[i].is_collecting_data:
                        # 进入调度状态

                        self.drivers[i].dispatched(self.regions_neighbor[self.drivers[i].now_location][
                                                       int(driver_action - 1)])

                        driver_location = self.drivers[i].now_location  # 目前driver的所在地
                        idle_order = self.get_idle_order()
                        idle_driver = self.get_idle_driver()

                        reward[i] += self.args.alpha * (max((idle_order[self.drivers[i].dispatched_destination] -
                                                             idle_driver[self.drivers[i].dispatched_destination]) - \
                                                            (idle_order[driver_location] - idle_driver[
                                                                driver_location]), 0))

        return reward

    def update_drivers_status(self):
        for driver_index in range(len(self.drivers)):
            if not self.drivers[driver_index].is_serving and \
                    self.drivers[driver_index].is_dispatched and \
                    not self.drivers[driver_index].is_collecting_data:
                # 假设调度在下一个时隙都能到达
                self.drivers[driver_index].now_location = self.drivers[driver_index].dispatched_destination
                self.drivers[driver_index].dispatched_destination = None
                self.drivers[driver_index].is_dispatched = False
                self.drivers[driver_index].is_collecting_data = False
            elif self.drivers[driver_index].is_collecting_data:
                self.drivers[driver_index].is_collecting_data = False
            elif self.drivers[driver_index].is_serving:
                if self.drivers[driver_index].time_arriver_order_destination == self.time_slot:
                    self.drivers[driver_index].drop_off()

    def update_orders_status(self):
        for order_index in range(len(self.orders)):
            if self.orders[order_index].overdue_time == self.time_slot and \
                    not self.orders[order_index].is_matched and \
                    not self.orders[order_index].is_overdue:
                self.orders[order_index].is_overdue = True
                self.overdue_order_number += 1

    def update_poi_data(self):
        for i, _ in enumerate(self.AoI_data):
            if self.PoI_data[i] > 0:
                self.AoI_data[i] += 1
            else:
                self.AoI_data[i] = 1

    def step(self, action):

        reward = self.take_action(action)  # take action and get reward
        self.time_slot += 1

        self.update_drivers_status()
        self.update_orders_status()
        self.update_poi_data()

        self.driver_order_match()
        self.create_orders()

        obs_next = self.get_vehicle_observation()

        done_n = np.zeros(self.n)

        if self.time_slot <= self.max_time_slot - 1:
            for i in range(self.n):
                done_n[i] = False
        else:
            for i in range(self.n):
                done_n[i] = True

        information = None

        return obs_next, reward, done_n, information

    def close(self):
        pass


if __name__ == '__main__':
    pass
