import numpy as np
from numpy import append


class Queue:

    def __init__(self, inter_arrival_times, execution_times, executors=1) -> None:
        # arrival times - sorted
        self.__inter_arrival_times = inter_arrival_times
        self.__execution_times = execution_times
        self.__departure_times = np.empty_like(inter_arrival_times)
        self.__wait_times = np.empty_like(inter_arrival_times)
        self.__executor_at = 0

    def process(self):
        self.__process_arrival_times()
        self.__process_departure_times()
        self.__process_wait_times()
        self.__process_length()

    def __process_arrival_times(self):
        self.__arrival_times = np.copy(self.__inter_arrival_times)
        for index, value in enumerate(self.__arrival_times):
            if index is not 0:
                self.__arrival_times[index] += self.__arrival_times[index-1]

    def __process_departure_times(self):
        for index, arrive_at in enumerate(self.__arrival_times):
            # we can start only if previous execution is finished
            start_at = max(self.__executor_at, arrive_at)
            processed_at = start_at + self.__execution_times[index]
            self.__executor_at = processed_at
            self.__departure_times[index] = processed_at

    def __process_wait_times(self):
        self.__wait_times = self.__departure_times - self.__arrival_times

    def __process_length(self):
        queue_size = 0
        arrival_index_at = 0
        departure_index_at = 0
        self.__length = [(0, 0)]
        while arrival_index_at < len(self.__arrival_times) or departure_index_at < len(self.__departure_times):
            while (arrival_index_at < len(self.__arrival_times)
                    and (departure_index_at >= len(self.__departure_times)
                         or self.__arrival_times[arrival_index_at] < self.__departure_times[departure_index_at])):
                queue_size += 1
                self.__length += [(self.__arrival_times[arrival_index_at], queue_size)]
                arrival_index_at += 1
            while (departure_index_at < len(self.__departure_times)
                   and (arrival_index_at >= len(self.__arrival_times)
                        or self.__departure_times[departure_index_at] <= self.__arrival_times[arrival_index_at])):
                queue_size -= 1
                self.__length += [(self.__departure_times[departure_index_at], queue_size)]
                departure_index_at += 1

    @property
    def departure_times(self):
        return self.__departure_times

    @property
    def arrival_times(self):
        return self.__arrival_times

    @property
    def length(self):
        return self.__length

    @property
    def wait_times(self):
        return self.__wait_times


