import numpy as np


class Queue:

    def __init__(self, inter_arrival_times, execution_times, executors=1) -> None:
        # arrival times - sorted
        self._arrival_times = Queue._inter_arrival_time_to_arrival_time(inter_arrival_times)
        self._execution_times = execution_times
        self._departure_times = []
        self._executor_at = 0

    def process(self):
        self._process_departure_times()
        self._process_wait_times()

    def _process_departure_times(self):
        for index, arrive_at in enumerate(self._arrival_times):
            start_at = max(self._executor_at, arrive_at)
            processed_at = start_at + self._execution_times[index]
            self._executor_at = processed_at
            self._departure_times.append(processed_at)

    def _process_wait_times(self):
        last_departure_at = self._departure_times[-1] + 1
        self._wait_times = np.zeros(last_departure_at)
        for arrive_at in self._arrival_times:
            self._wait_times[arrive_at:] += 1
        for depart_at in self._departure_times:
            self._wait_times[depart_at:] -= 1

    @property
    def departure_times(self):
        return self._departure_times

    @property
    def wait_times(self):
        return self._wait_times

    @staticmethod
    def _inter_arrival_time_to_arrival_time(inter_arrival_time):
        result = [x for x in inter_arrival_time]
        for index, value in enumerate(result):
            if index is not 0:
                result[index] += result[index-1]
        return result

