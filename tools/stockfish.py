"""Test-only UCI adapter. Never include this file or Stockfish in a submission."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

from harness.sandbox import AgentFailure


class StockfishAgent:
    def __init__(self, executable: Path) -> None:
        self.executable = executable
        self.process: subprocess.Popen[str] | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.stderr_tail = ""

    def start(self, init_budget_s: float) -> None:
        self.process = subprocess.Popen(
            [str(self.executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.process.stdout is None:
            raise AgentFailure("init")
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        self._send("uci")
        self._wait_for("uciok", init_budget_s)
        # Keep the benchmark comparable with the one-core competition environment.
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 64")
        self._send("isready")
        self._wait_for("readyok", init_budget_s)

    def move(self, fen: str, time_left_ms: int) -> str:
        self._send(f"position fen {fen}")
        # Let Stockfish's own time manager choose the search duration. The adapter only
        # supplies the clock; there is no fixed per-move cap.
        self._send(
            f"go wtime {time_left_ms} btime {time_left_ms} winc 500 binc 500"
        )
        line = self._wait_for("bestmove", max(1.0, time_left_ms / 1000.0 + 1.0))
        parts = line.split()
        if len(parts) < 2:
            raise AgentFailure("illegal")
        return parts[1]

    def stop(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.process.wait(timeout=1)
            self.process = None

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise AgentFailure("crash")
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise AgentFailure("crash") from error

    def _wait_for(self, prefix: str, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line.startswith(prefix):
                return line
        raise AgentFailure("flag")

    def _read_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.lines.put(line.strip())
