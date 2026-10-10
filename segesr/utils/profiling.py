import time
from collections import defaultdict

import torch

class PhaseTimer:
    """
    Times the consecutive phases of training iterations. The GPU is synchronized at every mark, so each phase
    gets its real duration (GPU work is otherwise asynchronous). When disabled, every call is a no-op.
    """

    def __init__(self):
        self.enabled = False
        self.totals = defaultdict(float)
        self.iterations = 0
        self._last = None

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        if self.enabled:
            self._sync()
            self._last = time.perf_counter()

    def mark(self, phase):
        """Ends the current phase, named `phase`, and starts the next one."""
        if self.enabled:
            self._sync()
            now = time.perf_counter()
            self.totals[phase] += now - self._last
            self._last = now

    def end_iteration(self):
        if self.enabled:
            self.iterations += 1

    def summary(self):
        """Mean seconds per iteration of every phase, in the order they were marked."""
        return {phase: total / max(self.iterations, 1) for phase, total in self.totals.items()}
