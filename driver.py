import numpy as np


class Driver:
    def __init__(self, driver_id, init_location):
        self.driver_id = driver_id
        self.init_location = init_location
        self.now_location = self.init_location

        self.is_serving = False  # 是否正在提供服务
        self.server_order = None  # 所服务的订单
        self.is_dispatched = False  # 是否已被下达了调度指令，且还未到达目的地

        self.is_collecting_data = False  # 是否正在收集数据
        self.data_vol = 0  # 收集的数据量

        self.dispatched_destination = None  # 调度的目的地
        self.time_arriver_order_destination = None  # 到达调度目的地的时间
        self.destination = None  # 接受了订单，订单的目的地

    def order_match(self, order, match_time):
        self.is_serving = True
        self.server_order = order
        self.destination = order.destination
        self.time_arriver_order_destination = order.travel_time + match_time

    def drop_off(self):
        self.is_serving = False
        self.server_order = None
        self.destination = None
        self.time_arriver_order_destination = None

    def collect_data(self):
        self.is_collecting_data = True




