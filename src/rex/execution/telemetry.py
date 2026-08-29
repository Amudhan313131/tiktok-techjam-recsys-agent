"""Process-tree resource sampling used by the isolated runner."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass
class ResourceTotals:
    peak_rss_bytes: int = 0
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0

    def sample(self, pid: int) -> None:
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except (psutil.Error, ProcessLookupError):
            return
        rss = 0
        user = 0.0
        system = 0.0
        for process in processes:
            try:
                rss += process.memory_info().rss
                cpu = process.cpu_times()
                user += cpu.user
                system += cpu.system
            except (psutil.Error, ProcessLookupError):
                continue
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.cpu_user_seconds = max(self.cpu_user_seconds, user)
        self.cpu_system_seconds = max(self.cpu_system_seconds, system)
