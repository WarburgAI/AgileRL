import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


class MetricsTracker:
    """A generic metrics tracker for accumulating and averaging values."""

    def __init__(self):
        self.values: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, name: str, value: float) -> None:
        """Add a value to a named metric.

        :param name: Name of the metric
        :type name: str
        :param value: Value to add
        :type value: float
        """
        if name not in self.values:
            self.values[name] = 0.0
            self.counts[name] = 0

        self.values[name] += value
        self.counts[name] += 1

    def get_average(self, name: str) -> float:
        """Get average value for a named metric.

        :param name: Name of the metric
        :type name: str
        :return: Average value
        :rtype: float
        """
        if name not in self.values or self.counts[name] == 0:
            return 0.0
        return self.values[name] / self.counts[name]

    def get_total(self, name: str) -> float:
        """Get total value for a named metric.

        :param name: Name of the metric
        :type name: str
        :return: Total value
        :rtype: float
        """
        return self.values.get(name, 0.0)

    def get_count(self, name: str) -> int:
        """Get count for a named metric.

        :param name: Name of the metric
        :type name: str
        :return: Count
        :rtype: int
        """
        return self.counts.get(name, 0)

    def reset(self) -> None:
        """Reset all metrics."""
        self.values.clear()
        self.counts.clear()

    def get_all_averages(self, prefix: str = "") -> Dict[str, float]:
        """Get all metrics as averages with optional prefix.

        :param prefix: Prefix to add to metric names
        :type prefix: str
        :return: Dictionary of averaged metrics
        :rtype: Dict[str, float]
        """
        return {
            f"{prefix}{name}": self.get_average(name) for name in self.values.keys()
        }

    def get_all_totals(self, prefix: str = "") -> Dict[str, float]:
        """Get all metrics as totals with optional prefix.

        :param prefix: Prefix to add to metric names
        :type prefix: str
        :return: Dictionary of total metrics
        :rtype: Dict[str, float]
        """
        return {f"{prefix}{name}": self.get_total(name) for name in self.values.keys()}


class TimingTracker:
    """A utility class for tracking timing metrics."""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start timing a named operation.

        :param name: Name of the operation to time
        :type name: str
        """
        self._start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing a named operation and return the elapsed time.

        :param name: Name of the operation that was being timed
        :type name: str
        :return: Elapsed time in seconds
        :rtype: float
        """
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was never started")

        elapsed = time.time() - self._start_times[name]
        self.add_time(name, elapsed)
        del self._start_times[name]
        return elapsed

    def add_time(self, name: str, elapsed_time: float) -> None:
        """Add elapsed time to a named metric.

        :param name: Name of the timing metric
        :type name: str
        :param elapsed_time: Time to add in seconds
        :type elapsed_time: float
        """
        if name not in self.timings:
            self.timings[name] = 0.0
            self.counters[name] = 0

        self.timings[name] += elapsed_time
        self.counters[name] += 1

    def get_time(self, name: str) -> float:
        """Get total time for a named metric.

        :param name: Name of the timing metric
        :type name: str
        :return: Total time in seconds
        :rtype: float
        """
        return self.timings.get(name, 0.0)

    def get_average_time(self, name: str) -> float:
        """Get average time for a named metric.

        :param name: Name of the timing metric
        :type name: str
        :return: Average time in seconds
        :rtype: float
        """
        if name not in self.timings or self.counters[name] == 0:
            return 0.0
        return self.timings[name] / self.counters[name]

    def reset(self) -> None:
        """Reset all timing metrics."""
        self.timings.clear()
        self.counters.clear()
        self._start_times.clear()

    def get_metrics(self, prefix: str = "time/") -> Dict[str, float]:
        """Get all timing metrics as a dictionary with optional prefix.

        :param prefix: Prefix to add to metric names
        :type prefix: str
        :return: Dictionary of timing metrics
        :rtype: Dict[str, float]
        """
        return {f"{prefix}{name}": time_val for name, time_val in self.timings.items()}

    @contextmanager
    def time_context(self, name: str):
        """Context manager for timing operations.

        :param name: Name of the operation to time
        :type name: str
        """
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
