# SPDX-License-Identifier: Apache-2.0
"""No CUDA: exercise hard KILL and ensure prefix tokens are never matched."""
import datetime as dt
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch


@unittest.skipUnless(sys.platform == "linux", "Linux /proc + pidfd supervisor")
class DeadlineTests(unittest.TestCase):
    def test_stale_pidfd_does_not_abort_other_signals(self):
        from rdd.deadline import signal_owned
        import signal
        with patch("rdd.deadline.owned_pids", return_value=[101, 102]), \
                patch("os.pidfd_open", side_effect=[OSError(errno.EINVAL, "exited"), 19]), \
                patch("os.close"), patch("signal.pidfd_send_signal") as send:
            signal_owned("rdd-test-unique-token-012345", signal.SIGTERM)
            send.assert_called_once_with(19, signal.SIGTERM)

    def test_exact_token_and_hard_stop(self):
        token = "rdd-test-" + uuid.uuid4().hex
        code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        outsider = subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, RDD_JOB_TOKEN=token + "-other"))
        try:
            with tempfile.TemporaryDirectory() as root:
                deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=6)
                status = Path(root) / "status.json"
                subprocess.run([sys.executable, "-m", "rdd.deadline", "--deadline", deadline.isoformat(),
                                "--token", token, "--status", str(status), "--disk-root", root,
                                "--min-free-gib", "0", "--grace-seconds", "2", "--",
                                sys.executable, "-c", code], check=True, timeout=18)
                result = json.loads(status.read_text())
                self.assertEqual(result["remaining_pids"], [])
                self.assertEqual(result["reason"], "deadline")
                self.assertIsNone(outsider.poll(), "prefix-matching token must not be signaled")
                self.assertLess(time.time() - deadline.timestamp(), 4)
        finally:
            outsider.kill()
            outsider.wait()


if __name__ == "__main__":
    unittest.main()
