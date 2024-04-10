import numpy as np

from src.queue import Queue, intervals_to_timestamps, timestamps_to_intervals


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
        np.testing.assert_allclose(queue.utilization(), [0.67, 1, 1, 1], rtol=1e-02)

    def test_departure_times_with_non_full_utilization(self):
        inter_arrival_time = np.full(shape=4, dtype=int, fill_value=2)
        execution_time = np.ones(shape=4, dtype=int)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.departure_times, [3, 5, 7, 9])
        np.testing.assert_allclose(queue.utilization(), [0.333, 0.5, 0.5, 0.5], rtol=1e-02)

    def test_queue_size_with_saturation(self):
        inter_arrival_time = np.ones(shape=4, dtype=int)
        execution_time = np.full(shape=4, dtype=int, fill_value=2)
        queue = Queue(inter_arrival_time, execution_time)
        queue.process()
        np.testing.assert_equal(queue.length_with_timestamps, [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 2), (7, 1), (9, 0)])

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

    def test_with_two_executors(self):
        inter_arrival_time = np.zeros(shape=8, dtype=int)
        execution_time = np.full(shape=8, dtype=int, fill_value=100)
        queue = Queue(inter_arrival_time, execution_time, executors=2)
        queue.process()
        np.testing.assert_equal(queue.departure_times, [100, 100, 200, 200, 300, 300, 400, 400])
        np.testing.assert_allclose(queue.utilization(0), [1, 1, 1, 1], rtol=1e-02)
        np.testing.assert_allclose(queue.utilization(1), [1, 1, 1, 1], rtol=1e-02)


class TestIntervalsAndTimestamps:

    def test_intervals_to_timestamps(self):
        result = intervals_to_timestamps(np.array([1, 1, 1, 1]))
        np.testing.assert_equal(result, [1, 2, 3, 4])

    def test_timestamps_to_intervals(self):
        result = timestamps_to_intervals(np.array([1, 2, 3, 4]))
        np.testing.assert_equal(result, [1, 1, 1, 1])

