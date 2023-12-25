import torch
import numpy as np
import random
from driver import Driver
from order import Order


class Env:
    def __init__(self, args):

        self.args = args
        self.time_slot = 0
        self.max_time_slot = 25
        self.city_road_graph = args.region_graph
        self.use_gnn = args.use_gnn

        self.n = args.agent_n  # 智能体的数量
        self.region_n = args.region_n  # 地图上划分的区域数量

        ###### 定义维度
        self.observation_space = 7  # [4 for _ in range(agent_n)]
        self.action_space = 5  # [8 for _ in range(agent_n)]
        ######

        self.drivers = []
        self.orders = []
        self.PoI_data = None
        self.AoI_data = None  # an array,used to record the AoI of PoI data
        self.regions_neighbor = []

        for i in range(self.region_n):
            neigh = []
            index = 0
            for node_index in self.city_road_graph.edges()[0]:
                if node_index.item() == i:
                    neigh.append(self.city_road_graph.edges()[1][index].item())
                index += 1
            self.regions_neighbor.append(neigh.copy())

        # 一些参数
        self.collect_unit = 100  # 每次可以收集100MB数据

    def init_poi_data(self):
        self.PoI_data = np.zeros(self.region_n)  # 一个numpy数组
        self.PoI_data[0] = 1000
        self.PoI_data[6] = 2000
        self.AoI_data = np.ones(self.region_n)

    def init_drivers(self):
        for i in range(self.n):
            init_location = np.random.randint(0, self.region_n)
            self.drivers.append(Driver(driver_id=i, init_location=init_location))

    def init_orders(self, numbers):
        number_of_orders = numbers
        region_item = range(self.region_n)

        prob = [0.1, 0.2, 0.05, 0.1, 0.05, 0.3, 0.05, 0.15, 0, 0]
        region = np.random.choice(region_item, size=number_of_orders, replace=True, p=prob)
        for i, data in enumerate(region):
            travel_time = np.random.randint(4, 9)
            destination = np.random.randint(0, self.region_n)
            self.orders.append(
                Order(order_id=i, init_location=data, generate_time=self.time_slot,
                      waiting_time=7, price=20, travel_time=travel_time, destination=destination)
            )

    def init_env(self):
        self.init_drivers()
        self.init_orders(10)
        self.init_poi_data()

    def get_idle_order(self):
        """
        返回每个区域内的待服务的订单数量
        :return:
        """
        result = np.zeros(self.region_n)
        for order in self.orders:
            if order.is_matched == False and order.is_overdue == False:
                result[order.init_location] += 1
        return result

    def get_idle_driver(self):
        """
        返回每个区域内空闲的车辆数量
        :return:
        """
        result = np.zeros(self.region_n)
        for driver in self.drivers:
            if driver.is_serving == False and driver.server_order == None and \
                    driver.is_dispatched == False and driver.is_collecting_data == False:
                result[driver.now_location] += 1
        return result

    def get_data_vol(self):
        """
        返回每个区域的目前未采集的数据量
        :return:
        """
        return self.PoI_data

    def get_vehicle_observation(self):
        obs = np.zeros((self.n, self.observation_space))
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
            obs[i, 6] = self.AoI_data[i]  # driver i 所在区域的数据信息年龄

        return obs

    def reset(self):
        self.time_slot = 0
        self.drivers.clear()
        self.orders.clear()

        self.init_env()
        return self.get_vehicle_observation()

    def driver_order_match(self):

        random.shuffle(self.drivers)
        random.shuffle(self.orders)
        for driver_index in range(len(self.drivers)):
            if self.drivers[driver_index].is_serving == False and \
                    self.drivers[driver_index].server_order == None and \
                    self.drivers[driver_index].is_dispatched == False and \
                    not self.drivers[driver_index].is_collecting_data:

                driver_location = self.drivers[driver_index].now_location

                for order_index in range(len(self.orders)):
                    if self.orders[order_index].is_matched == False and \
                            self.orders[order_index].is_overdue == False and \
                            self.orders[order_index].init_location == driver_location:
                        self.orders[order_index].driver_match(self.drivers[driver_index])
                        self.drivers[driver_index].order_match(self.orders[order_index], self.time_slot)
                        break

        self.orders.sort(key=lambda x: x.order_id)
        self.drivers.sort(key=lambda x: x.driver_id)

    def take_action(self, action):
        """
        :param action: action 是n维的numpy数组, 如[0, 2, 4, 1, 3, 0, 4] 每一个元素代表一个动作
        :return:
        """
        reward = np.zeros(self.n)
        idle_order = self.get_idle_order()
        idle_driver = self.get_idle_driver()
        data_vol = self.get_data_vol()
        for i, driver_action in enumerate(action):
            if driver_action == 0:
                # 为0表示不调度,也不收集数据
                pass
            elif driver_action == self.action_space - 1:
                # 为7表示收集数据
                if not self.drivers[i].is_serving and self.drivers[i].server_order == None and \
                        not self.drivers[i].is_dispatched and not self.drivers[i].is_collecting_data:
                    self.drivers[i].is_collecting_data = True
                    if self.PoI_data[self.drivers[i].now_location] >= self.collect_unit:
                        self.PoI_data[self.drivers[i].now_location] \
                            = self.PoI_data[self.drivers[i].now_location] - self.collect_unit
                        self.drivers[i].data_vol += self.collect_unit
                        reward[i] += self.args.beta * (
                                    self.collect_unit / self.AoI_data[self.drivers[i].now_location])  # 数据收集量 / 信息年龄
                    else:
                        self.drivers[i].data_vol += self.PoI_data[self.drivers[i].now_location]

                        reward[i] += self.args.beta * (self.PoI_data[self.drivers[i].now_location] / self.AoI_data[
                            self.drivers[i].now_location])
                        self.PoI_data[self.drivers[i].now_location] = 0
            else:
                if not self.drivers[i].is_serving and self.drivers[i].server_order == None and \
                        not self.drivers[i].is_dispatched and not self.drivers[i].is_collecting_data:
                    self.drivers[i].is_dispatched = True
                    self.drivers[i].dispatched_destination = self.regions_neighbor[self.drivers[i].now_location][
                        int(driver_action - 1)]

                    driver_location = self.drivers[i].now_location  # 目前driver的所在地
                    reward[i] += self.args.alpha * ((idle_order[self.drivers[i].dispatched_destination] -
                                                     idle_driver[self.drivers[i].dispatched_destination]) - \
                                                    (idle_order[driver_location] - idle_driver[driver_location]))

        return reward

    def step(self, action):
        """
        :param action:
        :return:
        """
        reward = self.take_action(action)
        self.time_slot += 1
        self.update_drivers_status()
        self.update_orders_status()
        self.update_poi_data()
        self.driver_order_match()

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

    def update_drivers_status(self):
        for driver_index in range(len(self.drivers)):
            if self.drivers[driver_index].is_serving == False and self.drivers[driver_index].server_order == None and \
                    self.drivers[driver_index].is_dispatched and \
                    not self.drivers[driver_index].is_collecting_data:
                # 假设调度在下一个时隙都能到达
                self.drivers[driver_index].now_location = self.drivers[driver_index].dispatched_destination
                self.drivers[driver_index].dispatched_destination = None
                self.drivers[driver_index].is_dispatched = False
                self.drivers[driver_index].is_collecting_data = False
            elif self.drivers[driver_index].is_collecting_data:
                self.drivers[driver_index].is_collecting_data = False
            elif self.drivers[driver_index].is_serving == True and self.drivers[driver_index].server_order != None:
                if self.drivers[driver_index].time_arriver_order_destination == self.time_slot:
                    self.drivers[driver_index].now_location = self.drivers[driver_index].destination
                    self.drivers[driver_index].is_serving = False
                    self.drivers[driver_index].server_order = None
                    self.drivers[driver_index].destination = None
                    self.drivers[driver_index].is_dispatched = False
                    self.drivers[driver_index].dispatched_destination = None
                    self.drivers[driver_index].is_collecting_data = False

    def update_orders_status(self):
        for order_index in range(len(self.orders)):
            if self.orders[order_index].overdue_time == self.time_slot and self.orders[order_index].is_matched == False \
                    and self.orders[order_index].is_overdue == False:
                self.orders[order_index].is_overdue = True

    def update_poi_data(self):
        """
        更新数据量和信息年龄
        :return:
        """
        for i, _ in enumerate(self.AoI_data):
            if self.PoI_data[i] != 0:
                self.AoI_data[i] += 1
            else:
                self.AoI_data[i] = 1

    def close(self):
        pass


if __name__ == "__main__":
    import dgl


    def print_aoi(env):
        print("AoI_Data:", env.AoI_data)


    def print_poi_data(env):
        print("PoI_data:", env.PoI_data)


    def print_order(env):
        for order in env.orders:
            print(order.init_location, order.is_matched)


    def print_driver(env):
        for driver in env.drivers:
            print(driver.now_location)


    u, v = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]), \
           torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 1, 3, 1, 3, 4, 4, 5, 6, 4, 6, 7, 5, 6, 7])
    region_graph = dgl.graph((u, v))
    args = None
    agent_n = 4
    use_gnn = True
    region_n = 10
    env = Env(args, region_graph, agent_n, use_gnn, region_n)
    env.init_env()
    print("region: ", env.region_n)
    print("neighbor:", env.regions_neighbor)
    print("order")
    print_order(env)
    print("driver")
    print_driver(env)
    print("vehicle observation:", env.get_vehicle_observation())
    action = torch.randint(0, 3, (4,)).numpy()
    print("action:", action)
    env.step(action)
    print("order:")
    print_order(env)
    print("driver")
    print_driver(env)
    print("vehicle observation:", env.get_vehicle_observation())
