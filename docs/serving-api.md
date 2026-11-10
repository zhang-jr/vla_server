# Policy Server

`PolicyServer` puts a `FleetScheduler` behind a WebSocket endpoint. Each connection is one robot, and requests from every connection share the fleet, so heads batch across robots (ADR-022). The wire protocol is openpi's (ADR-016): an unmodified openpi client (`openpi_client.websocket_client_policy.WebsocketClientPolicy`, tested with openpi-client 0.1.2) can drive it.

Source: `vla_runtime/serving/api_server.py`, `vla_runtime/serving/wire.py`

## Running it

```bash
# CPU-only plumbing check: mock backbone + reference head
vla-serve --policy mock --port 8000

# π0.5 on a GPU host, with the acceleration stack the fleet measurements used
vla-serve --policy pi05 \
    --policy-opt repo=/models/pi05_libero_finetuned_v044 \
    --policy-opt tokenizer=/models/paligemma-3b-pt-224 \
    --policy-opt cuda_graph=1 --policy-opt vision_bf16=1 --policy-opt expert_sdpa=2 \
    --image-key agentview=observation/image --image-key wrist=observation/wrist_image \
    --max-robots 4 --port 8000
```

`vla-serve` is `python -m vla_runtime.serving.api_server`. `--policy-opt` takes the same options as the sim benchmark (`vla_runtime/models/registry.py`), plus `head_hz` and, for the LeRobot models, `vlm_budget_ms`. `--image-key CAMERA=KEY` maps each camera the model reads (its camera map's robot-side names) to a request key; the defaults are openpi's LIBERO client keys. The server refuses to start if the model reads a camera no key maps.

## Protocol

1. **Connect** to `ws://host:port/`, optionally `?robot_id=<id>`; without one the server names the robot `robot-N`. The server registers the robot and sends a metadata map first: `server`, `policy`, `robot_id`, `chunk_size`, `action_dim`, `supports_rtc`, `replan_threshold`, the wire keys, and any `--metadata` entries (e.g. `action_mode=qpos`).
2. **Request**: one msgpack map per inference.

   | Key | Value |
   |---|---|
   | each `--image-key` | `[H, W, 3]` uint8 (or `[3, H, W]`; float in [0, 1] also accepted) |
   | `--state-key` (default `observation/state`) | state vector, in the convention the checkpoint was trained on |
   | `--prompt-key` (default `prompt`) | instruction string |
   | `prev_actions` (optional) | `[n, action_dim]` actions still queued on the robot, executed space; RTC policies only |
   | `inference_delay_steps` (optional) | control steps expected before the new chunk lands; RTC policies only |

3. **Response**:

   ```
   {"actions": float32 [chunk_size, action_dim],
    "server_timing": {"infer_ms": ..., "prev_total_ms": ...},
    "runtime": {"robot_id", "request_id", "degraded", "rtc", "head_batch_size", ...}}
   ```

   `runtime` carries the chunk's scalar metadata: `head_batch_size`; `replan_threshold` under admission control (the client should use it to decide when to request next, ADR-023); `fallback` and `reason` if the head failed and the fleet substituted a zero chunk. A coupled policy whose backbone timed out returns zero actions with `degraded` set, and the robot keeps executing what it has queued.

4. **Errors**: if a request fails, the server sends the traceback as a text frame and closes with 1011, as openpi's server does; openpi's client raises `RuntimeError` with it. A robot the fleet cannot take (`--max-robots`, admission control) is closed with 1013 (try again later); a duplicate `robot_id` with 1008.

`GET /healthz` answers `200 OK`.

Arrays use openpi's `msgpack_numpy` encoding: `{b"__ndarray__": True, b"data", b"dtype", b"shape"}` per array (`vla_runtime/serving/wire.py`).

## Using it from a robot

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="gpu-server", port=8000)
chunk = policy.infer({
    "observation/image": base_rgb,          # HWC uint8
    "observation/wrist_image": wrist_rgb,
    "observation/state": state,
    "prompt": "put the bowl on the plate",
})["actions"]                               # [chunk_size, action_dim]
```

The client owns the chunk buffer: which actions to execute, when to request the next chunk, and how to splice it in. The sim harness's realtime mode (`benchmarks/sim/runners.py`) is the reference for that loop. For RTC policies, send the queued tail as `prev_actions` together with `inference_delay_steps`, captured when the observation is taken (ADR-015). For trim blending, drop the leading steps that elapsed while the request was in flight (about latency × control_hz).

Two deployment notes:

- **Proxies.** websockets clients (openpi's included) honor `HTTP_PROXY` / `HTTPS_PROXY`. On a robot with a system proxy, set `NO_PROXY` for the server's address; otherwise every request makes a detour through the proxy, and if the proxy cannot reach the server the handshake fails with "did not receive a valid HTTP response".
- **Budgets.** The fleet measurements are server-side latency. On the ARX X5, about 65 ms of the end-to-end time is outside inference (camera frame age, transport, command). Subtract it from the policy's latency tolerance before deciding how many robots one GPU can serve.

## Embedding the server

```python
from vla_runtime.serving.api_server import PolicyServer, WireFormat
from vla_runtime.serving.fleet import FleetConfig

server = PolicyServer(policy, WireFormat(), fleet_config=FleetConfig(max_robots=4), port=8000)
await server.serve_forever()          # or start() / stop()
```

`policy` is any `vla_runtime.policy.Policy`. `WireFormat.to_request()` maps the state vector into `RobotState.extra["policy_state"]`, which prefix-state models (π0.5, SmolVLA) read. Heads that read the flat state get the vector as `joint_positions` with zero velocities and gripper.

## Limits

- One request in flight per connection, as in openpi's protocol. A robot that wants overlapping requests opens a second connection, which counts as a second robot.
- The state arrives as one vector. Policies that need structured state (separate joint velocities, end-effector pose) need a richer wire format.
- The ARX X5 closed-loop benchmark still uses its own TCP protocol and a single-model server (`benchmarks/closed_loop`), not this endpoint.

## Related documents

- [Fleet serving](./fleet-serving.md): what happens to a request inside the fleet
- [Scheduler](./scheduler.md): chunk stitching, which the client side has to do
