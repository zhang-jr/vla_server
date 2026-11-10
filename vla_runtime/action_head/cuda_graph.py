"""CUDA Graphs for launch-bound adapter loops (V4).

A flow-matching expert runs its small network once per denoise step, so at
batch 1 the GPU mostly waits for Python to launch kernels: the π0.5 expert's
10 steps take 89 ms eager and 33 ms replayed from a graph (BENCHMARKS.md, V4).
:class:`GraphCache` captures a function once per input shape and replays it
with the inputs copied into the captured buffers.

The function must be capturable: no host-device copies, no CPU reads of GPU
values, no RNG, and the same kernels for every call with the same shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

# Eager runs before capture: lazy init (cuBLAS handles, allocator growth)
# must not happen inside the graph.
_WARMUP_RUNS = 3


@dataclass
class _Captured:
    graph: torch.cuda.CUDAGraph
    inputs: list[torch.Tensor]
    output: torch.Tensor


class GraphCache:
    """One CUDA Graph of ``fn`` per distinct input signature.

    Args:
        fn: ``(*tensors) → tensor``; capturable (see module docstring).

    Graphs share one memory pool, so calls must not overlap: the returned
    tensor is the graph's output buffer and the next call overwrites it.
    """

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        self._fn = fn
        self._graphs: dict[tuple, _Captured] = {}
        self._pool = torch.cuda.graph_pool_handle()

    @property
    def num_graphs(self) -> int:
        return len(self._graphs)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        key = tuple((t.shape, t.dtype, t.device) for t in inputs)
        captured = self._graphs.get(key)
        if captured is None:
            captured = self._graphs[key] = self._capture(inputs)
        for static, value in zip(captured.inputs, inputs, strict=True):
            static.copy_(value)
        captured.graph.replay()
        return captured.output

    def _capture(self, inputs: Sequence[torch.Tensor]) -> _Captured:
        static = [t.clone() for t in inputs]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_WARMUP_RUNS):
                self._fn(*static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # thread_local: other threads may keep launching work during capture.
        with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
            output = self._fn(*static)
        return _Captured(graph=graph, inputs=static, output=output)
