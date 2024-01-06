import numpy as np


class PoI:
    def __init__(self, poi_id, init_location, expiration_time):
        self.poi_id = poi_id
        self.data_vol = 0  # 目前的数据量
        self.expiration_time = expiration_time  # 数据的有效时间，从产生经过这么多时间后的数据将会清空
        self.time_to_create_data = -1  # 再次产生数据的时间

