import numpy as np

class Queue:

    def __init__(self, inter_arrival_times, execution_times, executors=1) -> None:
        self.__inter_arrival_times = inter_arrival_times
        self.__execution_times = execution_times
        self.__departure_times = np.empty_like(inter_arrival_times)
        self.__wait_times = np.empty_like(inter_arrival_times)
        self.__utilization_by_executor = {
            executor_id: []
            for executor_id in range(0, executors)
        }
        self.__executors_at = {executor_id: 0 for executor_id in range(0, executors)}

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
            earliest_executor_id = min(self.__executors_at, key=self.__executors_at.get)
            start_at = max(self.__executors_at[earliest_executor_id], arrive_at)
            processed_at = start_at + self.__execution_times[index]
            # processing utilization
            self.__utilization_by_executor[earliest_executor_id] += [self.__process_utilization(
                arrive_at,
                self.__executors_at[earliest_executor_id],
                self.__execution_times[index]
            )]
            self.__executors_at[earliest_executor_id] = processed_at
            self.__departure_times[index] = processed_at

    @staticmethod
    def __process_utilization(arrive_at, executor_at, execution_time) -> float:
        wait = max(arrive_at - executor_at, 0)
        duration = execution_time + wait
        utilization: float = 1 - wait / duration
        return utilization

    def __process_wait_times(self):
        self.__wait_times = self.__departure_times - self.__arrival_times

    def __process_length(self):
        queue_size = 0
        arrival_index_at = 0
        departure_index_at = 0
        # TODO dict?
        self.__length = [(0, 0)]

        def has_more_arrivals():
            return arrival_index_at < len(self.__arrival_times)

        def has_more_departures():
            return departure_index_at < len(self.__departure_times)

        def arrive_earlier_than_depart():
            return self.__arrival_times[arrival_index_at] < self.__departure_times[departure_index_at]

        def no_more_departures_or_arriving_first():
            return not has_more_departures() or arrive_earlier_than_depart()

        def no_more_arrivals_or_departing_first():
            return not has_more_arrivals() or not arrive_earlier_than_depart()

        while has_more_arrivals() or has_more_departures():
            while has_more_arrivals() and no_more_departures_or_arriving_first():
                queue_size += 1
                self.__length += [(self.__arrival_times[arrival_index_at], queue_size)]
                arrival_index_at += 1
            while has_more_departures() and no_more_arrivals_or_departing_first():
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

    def utilization(self, executor_id: int = 0):
        return self.__utilization_by_executor[executor_id]


