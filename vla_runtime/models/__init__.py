"""Model adapters: real VLA checkpoints behind VLMBackend / ActionHeadRuntime.

- ``prefix_kv``: the model-agnostic prefix-KV backbone / expert head / policy
- ``smolvla``, ``pi05``: the LeRobot SmolVLA and π0.5 adapters
- ``registry``: load an adapter by name from ``key=value`` options

LeRobot is imported only when a model is constructed, so importing this
package does not require it.
"""
