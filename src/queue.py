import numpy as np


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
        last_departure_at = int(self.__departure_times[-1]) + 1
        self.__length = np.zeros(last_departure_at)
        for arrive_at in self.__arrival_times:
            self.__length[int(arrive_at):] += 1
        for depart_at in self.__departure_times:
            self.__length[int(depart_at):] -= 1

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


