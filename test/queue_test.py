import numpy as np

from src.queue import Queue


class TestQueue:

    def test_arrival_times(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        queue = Queue(inter_arrival_time, np.zeros_like(inter_arrival_time))
        queue.process()
        np.testing.assert_equal(queue.arrival_times, [1, 2, 3, 4])
        inter_arrival_time_increasing = np.arange(start=1, stop=5, dtype=int)
        queue = Queue(inter_arrival_time_increasing, np.zeros_like(inter_arrival_time))
        queue.process()
        np.testing.assert_equal(queue.arrival_times, [1, 3, 6, 10])

    def test_departure_times(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.ones_like(inter_arrival_time)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.departure_times, [2, 3, 4, 5])

    def test_departure_times_with_saturation(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.full(shape=4, dtype=int, fill_value=2)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.departure_times, [3, 5, 7, 9])
        np.testing.assert_allclose(queue.utilization, [0.67, 1, 1, 1], rtol=1e-02)

    def test_departure_times_with_non_full_utilization(self):
        inter_arrival_time = np.full(shape=4, dtype=int, fill_value=2)
        execution_time = np.ones(shape=4, dtype=int)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.departure_times, [3, 5, 7, 9])
        np.testing.assert_allclose(queue.utilization, [0.333, 0.5, 0.5, 0.5], rtol=1e-02)

    def test_queue_size_with_saturation(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.full(shape=4, dtype=int, fill_value=2)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.length, [(0, 0), (1, 1), (2, 2), (3, 1), (3, 2), (4, 3), (5, 2), (7, 1), (9, 0)])

    def test_wait_times(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.ones_like(inter_arrival_time)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.wait_times, [1, 1, 1, 1])

    def test_wait_times_with_saturation(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.full(shape=4, dtype=int, fill_value=2)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.wait_times, [2, 3, 4, 5])


