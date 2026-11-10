"""Real-robot closed-loop latency / throughput benchmark (ARX X5, one arm).

Script 2 of the requirement check (script 1 is ``benchmarks/latency``, model
only). Measures the whole path a deployment takes: camera frame (ROS2
``realsense2_camera``) → observation → TCP → policy server (GPU container) →
actions → ``/arm_cmd`` on the arm's ROS2 controller.

Two processes, two Python environments:

* ``policy_server`` runs in the benchmark container with the model's own venv
  and reuses the script-1 adapters (``benchmarks.latency.models``).
* ``arx_x5`` runs on the robot host under ROS2 Foxy (system Python 3.8): it
  owns the camera / arm topics and drives the loop. It and everything it
  imports must stay Python 3.8 / numpy 1.17 compatible and stdlib + numpy only.

Safety modes (``--mode``): ``dry-run`` never publishes a command; ``hold``
publishes the arm's own starting pose every step (``mode=5`` joint position
control), so the command path is timed while the arm stays put. Model actions
are never sent to the arm.

Modules:
    protocol      — length-prefixed JSON header + raw numpy blobs over TCP (stdlib)
    loop          — ROS-free loop core: client, timings, hold guard, statistics
    policy_server — model process (container side)
    arx_x5        — ROS2 node + CLI (robot host side)
"""
