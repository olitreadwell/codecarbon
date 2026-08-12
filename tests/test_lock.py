import signal
import threading
import unittest
from unittest.mock import mock_open, patch

from codecarbon.lock import LOCKFILE, Lock


class TestLock(unittest.TestCase):
    def setUp(self):
        self.lock = Lock()

    @patch("codecarbon.lock.os.remove")
    @patch("codecarbon.lock.open", new_callable=mock_open)
    def test_acquire_lock_creates_lock_file(self, mock_file, mock_remove):
        self.lock.acquire()
        mock_file.assert_called_once_with(LOCKFILE, "x")
        self.assertTrue(self.lock._has_created_lock)

    @patch("codecarbon.lock.os.remove")
    @patch("codecarbon.lock.open", new_callable=mock_open)
    def test_acquire_lock_exits_when_lock_file_exists(self, mock_file, mock_remove):
        mock_file.side_effect = FileExistsError
        with self.assertRaises(FileExistsError):
            self.lock.acquire()

    @patch("codecarbon.lock.os.remove")
    @patch("codecarbon.lock.open", new_callable=mock_open)
    def test_release_removes_lock_file(self, mock_file, mock_remove):
        self.lock.acquire()
        self.lock.release()
        mock_remove.assert_called_once_with(LOCKFILE)

    @patch("codecarbon.lock.os.remove")
    @patch("codecarbon.lock.open", new_callable=mock_open)
    def test_release_does_not_release_when_not_created_by_this_instance(
        self, mock_file, mock_remove
    ):
        self.lock.release()
        mock_remove.assert_not_called()
        self.assertFalse(self.lock._has_created_lock)

    @patch("codecarbon.lock.os.remove")
    @patch("codecarbon.lock.open", new_callable=mock_open)
    def test_acquire_release_with_multiple_threads(self, mock_file, mock_remove):
        """
        Test that the lock file is created and removed correctly when multiple threads are used.
        """
        # Simulate a complex threading scenario where the lock file is created and removed multiple times
        # First call succeeds, second fails, third succeeds, subsequent calls raise FileExistsError
        mock_file.side_effect = (
            [mock_open().return_value]
            + [FileExistsError]
            + [mock_open().return_value]
            + [FileExistsError] * 7
        )

        def thread_target():
            # Create a lock instance in each thread
            lock = Lock()
            try:
                lock.acquire()
            except Exception:
                pass
            finally:
                lock.release()

        threads = [threading.Thread(target=thread_target, args=()) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Ensure that the lock file was created and removed correctly
        self.assertTrue(mock_file.called)
        self.assertTrue(mock_remove.called)


class TestLockSignalHandlers(unittest.TestCase):
    """The lock must not permanently steal the host application's handlers."""

    def setUp(self):
        self.original_handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }

    def tearDown(self):
        for sig, handler in self.original_handlers.items():
            signal.signal(sig, handler)

    def test_release_restores_previous_handlers(self):
        def sentinel(signum, frame):
            pass

        signal.signal(signal.SIGTERM, sentinel)
        lock = Lock()
        self.assertEqual(signal.getsignal(signal.SIGTERM), lock._handle_exit)
        lock.release()
        self.assertIs(signal.getsignal(signal.SIGTERM), sentinel)
        self.assertIs(
            signal.getsignal(signal.SIGINT), self.original_handlers[signal.SIGINT]
        )

    @unittest.skipIf(
        not hasattr(signal, "raise_signal"), "requires signal.raise_signal (3.8+)"
    )
    @patch("codecarbon.lock.os.remove")
    def test_signal_is_forwarded_to_previous_handler(self, mock_remove):
        received = []

        signal.signal(signal.SIGTERM, lambda signum, frame: received.append(signum))
        lock = Lock()
        signal.raise_signal(signal.SIGTERM)
        self.assertEqual(received, [signal.SIGTERM])
        self.assertFalse(lock._previous_handlers)


if __name__ == "__main__":
    unittest.main()
