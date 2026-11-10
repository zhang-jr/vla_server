"""Model-level inference latency / throughput benchmark (one robot, no sim, no arm).

Answers "does model X meet an end-to-end latency / FPS requirement on this GPU":
each timed call goes from raw camera frames (uint8 HxWx3) + instruction + state
to actions on the host (numpy), i.e. preprocessing, every forward pass and the
device→host copy, with the model's own native implementation.

Entry point::

    python -m benchmarks.latency.run --help

Modules:
    harness — timing loop, stage timer, GPU memory monitor, statistics, verdict
    models  — model adapters (mock, OpenVLA, OpenVLA-OFT, LeRobot π0 / π0.5 / SmolVLA, openpi)
    run     — CLI; runs each model in its own process / Python env and aggregates
"""
