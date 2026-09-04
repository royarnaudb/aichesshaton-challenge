# Running the Chess Lab

This guide is for contributors who want to run the local harness, compare search
changes, and benchmark against Stockfish. The local lab is for development only;
the competition submission is still just the root `agent.py` and any permitted
weight files.

## Requirements

### Development machine

- Python 3.12 or newer
- `uv` for dependency and command management
- Git
- Optional: GNU Make, or PowerShell for running the commands directly
- Optional: a local Stockfish executable for comparison games
- Internet access during setup only, so `uv sync` can download dependencies

The lab dependencies are pinned in `pyproject.toml` and `uv.lock`. They include
`python-chess`, NumPy, Numba, PyTorch CPU, and ONNX Runtime. Stockfish is not a
Python dependency and is installed separately.

### Competition runtime

The starter repository is designed around the competition environment:

- One CPU core
- 2 GB RAM
- No GPU
- No network access while the agent is playing
- A 50 MB maximum for the uncompressed submission archive
- A root-level `agent.py` exposing `get_move(fen, time_left_ms)`

The platform rules and current limits can change, so read the official
[AI Chessathon documentation](https://aichessathon.com/docs) before uploading.

Weight files must comply with the competition rules. In particular, do not add
third-party engine binaries or an external engine wrapper to the submission.
Keep all benchmark-only resources outside the submission archive.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run ruff check .
uv run mypy
uv run python -m unittest discover -s tests -v
```

The equivalent shortcut is:

```bash
make setup
make gate
```

On Windows, use the `uv run ...` commands directly if `make` is unavailable.

## Play locally

Play one full-clock game against a baseline:

```bash
uv run python -m harness.play --white . --black baselines/greedy
```

The default lab clock is 120 seconds plus a 500 ms increment. To run a faster
development check, use the arena command:

```bash
uv run python -m harness.arena \
  --opponent baselines/minimax \
  --games 20 \
  --base-ms 10000 \
  --increment-ms 100
```

Keep the time control, colour alternation, number of games, and starting
positions fixed when comparing two versions of an agent. Short games are useful
for iteration, but their results should not be treated as full-clock strength.

## Metrics and Elo

Measure the current agent against an opponent and save a JSON report:

```bash
uv run python -m tools.metrics \
  --agent . \
  --opponent baselines/minimax \
  --games 20 \
  --base-ms 120000 \
  --increment-ms 500 \
  --json-out metrics.json
```

Run the local round-robin ladder:

```bash
uv run python -m tools.tournament \
  . baselines/random baselines/greedy baselines/minimax baselines/numba \
  --games-per-pair 20 \
  --base-ms 120000 \
  --increment-ms 500 \
  --json-out tournament.json
```

The reported Elo is provisional and relative to the selected agent pool. It is
not an official rating. Use multiple games and inspect failed games, latency,
nodes, and completed depth alongside wins and draws.

## Stockfish comparison

Stockfish is an external local benchmark, not part of the submission. Download
an official build from the [Stockfish download page](https://stockfishchess.org/download/)
and store the executable outside the repository, or under the ignored
`tools/stockfish-bin/` directory.

Examples:

```bash
# Linux/macOS
uv run python -m tools.tournament \
  . baselines/minimax \
  --stockfish /path/to/stockfish \
  --games-per-pair 20 \
  --base-ms 120000 \
  --increment-ms 500 \
  --json-out stockfish.json

# Windows PowerShell
uv run python -m tools.tournament `
  . baselines/minimax `
  --stockfish "C:\\tools\\stockfish\\stockfish-windows-x86-64-avx2.exe" `
  --games-per-pair 20 `
  --base-ms 120000 `
  --increment-ms 500 `
  --json-out stockfish.json
```

The adapter configures Stockfish to one thread and 64 MB hash so the comparison
resembles the competition's one-core environment. It gives Stockfish the same
clock and increment rather than imposing a fixed move-time cap. Do not copy the
executable into `submission.zip`, import it from `agent.py`, or use it to choose
moves in a competition submission.

## Testing search changes

Run the fast correctness suite after every change:

```bash
uv run python -m unittest discover -s tests -v
```

For an A/B time-management comparison:

```bash
uv run python -m tools.profile_match \
  --first balanced \
  --second conservative \
  --games 100 \
  --base-ms 120000 \
  --increment-ms 500 \
  --json-out profile-match.json
```

Search metrics are written by the agent to stderr as JSON records. The harness
keeps protocol output separate, so diagnostic metrics do not corrupt move
responses.

## Submission check

Build and inspect the archive before uploading:

```bash
uv run python -m harness.package
```

The generated `submission.zip` must remain within the competition's unzipped
size limit. Only include files that the competition rules allow; local tools,
Stockfish, benchmark reports, and caches are not submission assets.
