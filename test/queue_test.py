import numpy
import numpy as np
import pytest
from numpy import ndarray

from src.queue import Queue


class TestQueue:

    def test_to_departure_times(self):
        inter_arrival_time = [1, 1, 1, 1]
        execution_time = [1, 1, 1, 1]
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        assert queue.departure_times == [2, 3, 4, 5]

    def test_to_departure_times_with_saturation(self):
        inter_arrival_time = np.array([1, 1, 1, 1])
        execution_time = np.array([2, 2, 2, 2])
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        assert queue.departure_times == [3, 5, 7, 9]
        assert queue.wait_times.tolist() == [0, 1, 2, 2, 3, 2, 2, 1, 1, 0]

    def test_inter_arrival_times(self):
        assert Queue._inter_arrival_time_to_arrival_time([1, 1, 1, 1]) == [1, 2, 3, 4]
        assert Queue._inter_arrival_time_to_arrival_time([1, 2, 3, 4]) == [1, 3, 6, 10]

