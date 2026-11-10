"""Test-wide configuration.

Tests marked ``@pytest.mark.gpu`` need a CUDA device (kernels are never
mocked). They are skipped where none is visible, so the CPU suite stays
runnable on any machine; run them on a CUDA host with ``pytest -m gpu``.
"""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device; run on a CUDA host with -m gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
