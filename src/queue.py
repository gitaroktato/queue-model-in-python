import numpy as np
import numpy.typing as npt


def intervals_to_timestamps(intervals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    timestamps = np.copy(intervals)
    for index, value in enumerate(timestamps):
        if index != 0:
            timestamps[index] += timestamps[index - 1]
    return timestamps


def timestamps_to_intervals(timestamps: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    intervals = np.copy(timestamps)
    for index, value in reversed(list(enumerate(intervals))):
        if index != 0:
            intervals[index] -= intervals[index - 1]
    return intervals


class Queue:

    def __init__(
            self,
            inter_arrival_times: npt.NDArray[np.float64],
            execution_times: npt.NDArray[np.float64],
            executors: int = 1
    ) -> None:
        self.__inter_arrival_times = inter_arrival_times
        self.__execution_times = execution_times
        self.__departure_times = np.empty_like(inter_arrival_times)
        self.__wait_times = np.empty_like(inter_arrival_times)
        self.__utilization_by_executor: dict[int, list[float]] = {
            executor_id: []
            for executor_id in range(0, executors)
        }
        self.__executors_at: dict[int, float] = {executor_id: 0 for executor_id in range(0, executors)}

    def process(self) -> None:
        self.__process_arrival_times()
        self.__process_departure_times()
        self.__process_wait_times()
        self.__process_length()

    def __process_arrival_times(self) -> None:
        self.__arrival_times = intervals_to_timestamps(self.__inter_arrival_times)

    def __process_departure_times(self) -> None:
        for index, arrive_at in enumerate(self.__arrival_times):
            # we can start only if previous execution is finished
            earliest_executor_id = min(self.__executors_at, key=self.__executors_at.__getitem__)
            start_at = max(self.__executors_at[earliest_executor_id], arrive_at)
            processed_at = start_at + self.__execution_times[index]
            # processing utilization
            utilization = self.__process_utilization(
                arrive_at,
                self.__executors_at[earliest_executor_id],
                self.__execution_times[index]
            )
            self.__utilization_by_executor[earliest_executor_id].append(utilization)
            self.__executors_at[earliest_executor_id] = processed_at
            self.__departure_times[index] = processed_at

    @staticmethod
    def __process_utilization(arrive_at: float, executor_at: float, execution_time: float) -> float:
        wait = max(arrive_at - executor_at, 0)
        duration = execution_time + wait
        utilization: float = 1 - wait / duration
        return utilization

    def __process_wait_times(self) -> None:
        self.__wait_times = self.__departure_times - self.__arrival_times

    def __process_length(self) -> None:
        queue_size = 0
        arrival_index_at = 0
        departure_index_at = 0
        self.__length: dict[float, int] = {0.0: 0}

        def has_more_arrivals() -> bool:
            return arrival_index_at < len(self.__arrival_times)

        def has_more_departures() -> bool:
            return departure_index_at < len(self.__departure_times)

        def arrive_earlier_than_depart() -> bool:
            return self.__arrival_times[arrival_index_at] < self.__departure_times[departure_index_at]

        def no_more_departures_or_arriving_first() -> bool:
            return not has_more_departures() or arrive_earlier_than_depart()

        def no_more_arrivals_or_departing_first() -> bool:
            return not has_more_arrivals() or not arrive_earlier_than_depart()

        while has_more_arrivals() or has_more_departures():
            while has_more_arrivals() and no_more_departures_or_arriving_first():
                queue_size += 1
                timestamp = self.__arrival_times[arrival_index_at]
                self.__length[timestamp] = max(queue_size, self.__length.get(timestamp, 0))
                arrival_index_at += 1
            while has_more_departures() and no_more_arrivals_or_departing_first():
                queue_size -= 1
                timestamp = self.__departure_times[departure_index_at]
                self.__length[timestamp] = max(queue_size, self.__length.get(timestamp, 0))
                departure_index_at += 1

    @property
    def departure_times(self) -> npt.NDArray[np.float64]:
        return self.__departure_times

    @property
    def arrival_times(self) -> npt.NDArray[np.float64]:
        return self.__arrival_times

    @property
    def length_with_timestamps(self) -> list[tuple[float, int]]:
        return list(self.__length.items())

    @property
    def length(self) -> npt.NDArray[np.int_]:
        return np.array(list(self.__length.values()))

    @property
    def wait_times(self) -> npt.NDArray[np.float64]:
        return self.__wait_times

    def utilization(self, executor_id: int = 0) -> npt.NDArray[np.float64]:
        return np.array(self.__utilization_by_executor[executor_id])


