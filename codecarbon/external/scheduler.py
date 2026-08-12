import time
from threading import Event, Thread, current_thread

from codecarbon.external.logger import logger


class PeriodicScheduler:
    """
    Run ``function`` every ``interval`` seconds on a single daemon thread.

    The deadline is absolute, so the cadence does not drift with the time the
    function itself takes. A tick that overruns its slot is skipped rather than
    queued, so the function is never re-entered.
    """

    def __init__(self, interval, function, *args, **kwargs):
        """
        Init the scheduler. You have to call start() after initialization.
        ::interval:: interval in seconds to run the function.
        ::function:: function to run.
        ::args:: args to pass to the function.
        ::kwargs:: kwargs to pass to the function.
        """
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._stop_event = Event()
        self._thread = None

    @property
    def _stopped(self):
        return self._thread is None or not self._thread.is_alive()

    def start(self):
        """
        Start the scheduler. Calling it on a running scheduler is a no-op.
        """
        if not self._stopped:
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._loop,
            daemon=True,
            name=f"codecarbon-{getattr(self.function, '__name__', 'scheduler')}",
        )
        self._thread.start()

    def _loop(self):
        next_call = time.monotonic() + self.interval
        while not self._stop_event.wait(max(0.0, next_call - time.monotonic())):
            try:
                self.function(*self.args, **self.kwargs)
            except Exception:  # noqa: BLE001 - must not kill the only thread
                logger.error("Scheduled measurement failed", exc_info=True)
            # Absolute deadline so the cadence does not drift, but if we
            # overran (or the process was suspended) skip ahead instead of
            # firing a burst of catch-up ticks.
            next_call += self.interval
            now = time.monotonic()
            if next_call <= now:
                next_call = now + self.interval

    def stop(self, timeout=None):
        """
        Stop the scheduler and wait for the in-flight call to return.
        ::timeout:: seconds to wait for the running function, bounded by
        default so a wedged measurement cannot hang the caller for long.
        """
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not current_thread():
            # ponytail: 5s cap is a guess at "a measurement should never take
            # longer than this"; make it configurable if a slow hardware
            # backend ever needs more.
            thread.join(min(self.interval, 5.0) if timeout is None else timeout)
