class Order:
    def __init__(self, order_id, init_location, generate_time, waiting_time=10,
                 price=100, travel_time=0, destination=0):
        self.order_id = order_id
        self.price = price
        self.waiting_time = waiting_time  # 最大等待时间
        self.init_location = init_location  # 订单的等待位置
        self.generate_time = generate_time  # 订单的产生时间
        self.overdue_time = self.generate_time + self.waiting_time  # 订单的过期时间
        self.destination = destination  # 该订单的目的地

        self.is_matched = False  # 是否被车辆接单
        self.is_overdue = False  # 订单是否已过期

        self.travel_time = travel_time  # 从出发地到达目的地需要的时间

    def driver_match(self, driver):
        self.is_matched = True
