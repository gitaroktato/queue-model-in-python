import numpy as np

from src.queue import Queue


class TestQueue:

    def test_departure_times(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.ones_like(inter_arrival_time)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        assert queue.departure_times.tolist() == [2, 3, 4, 5]

    def test_departure_times_with_saturation(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.full(shape=4, dtype=int, fill_value=2)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        assert queue.departure_times.tolist() == [3, 5, 7, 9]

    def test_queue_size_with_saturation(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.full(shape=4, dtype=int, fill_value=2)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        assert queue.length.tolist() == [0, 1, 2, 2, 3, 2, 2, 1, 1, 0]

    def test_wait_times(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.ones_like(inter_arrival_time)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        assert queue.wait_times.tolist() == [1, 1, 1, 1]

    def test_wait_times_with_saturation(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.full(shape=4, dtype=int, fill_value=2)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        assert queue.wait_times.tolist() == [2, 3, 4, 5]

    def test_converting_from_inter_arrival_times_to_arrival_times(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        assert Queue._inter_arrival_time_to_arrival_time(inter_arrival_time) == [1, 2, 3, 4]
        inter_arrival_time_increasing = np.arange(start=1, stop=5, dtype=int)
        assert Queue._inter_arrival_time_to_arrival_time(inter_arrival_time_increasing) == [1, 3, 6, 10]

