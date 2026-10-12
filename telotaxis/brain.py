"""The MaleCNS visual pathway, one shared graph, many membrane states.

Synapses are the paper's counts, signed by the presynaptic consensus
transmitter, then scaled so each neuron's incoming mass is SYN_GAIN.
Light enters at lamina L1–L5. The decoder may see only DNp left versus DNp right.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from telotaxis.constants import (
    ADAPT_JUMP,
    INPUT_GAIN,
    NEURAL_DT,
    RATE_TRACE_S,
    REFRACTORY_STEPS,
    SPIKE_VIS_S,
    SYN_GAIN,
    TAU_ADAPT,
    TAU_M,
    TAU_SYN,
    V_THRESHOLD,
)

PATHWAY_PATH = Path(__file__).resolve().parents[1] / "data" / "pathway.npz"


class Pathway:
    def __init__(self, n_agents: int, seed: int, path: Path | None = None) -> None:
        blob = np.load(path or PATHWAY_PATH, allow_pickle=False)
        self.meta = json.loads(str(blob["meta"]))
        self.ids = blob["ids"]
        self.types = blob["types"]
        self.role = blob["role"]
        self.side = blob["side"]
        self.dnp = blob["dnp"].astype(bool)
        self.column = blob["column"]
        self.column_azimuth = blob["column_azimuth"].astype(np.float64)
        self.position = blob["position"].astype(np.float32)
        self.n_neurons = int(self.ids.shape[0])
        self.n_columns = int(self.column_azimuth.shape[0])
        self.n = n_agents
        pre = blob["pre"].astype(np.int32)
        post = blob["post"].astype(np.int32)
        raw = blob["sign"][pre].astype(np.float32) * np.log1p(blob["count"].astype(np.float32))
        # A strong synapse (90th percentile of log count) is SYN_GAIN. Weaker
        # counts stay weaker. Normalising by in-degree silenced the pathway,
        # because a visual-projection neuron's anatomical degree is huge.
        raw *= np.float32(SYN_GAIN / np.percentile(np.abs(raw), 90))
        self.weights = sparse.csr_matrix(
            (raw, (pre, post)), shape=(self.n_neurons, self.n_neurons), dtype=np.float32
        )
        self.lamina = np.flatnonzero(self.role == 1)
        self.lamina_column = self.column[self.lamina]
        self.driven = self.lamina[self.lamina_column >= 0]
        self.driven_column = self.lamina_column[self.lamina_column >= 0]
        self.left_dnp = np.flatnonzero(self.dnp & (self.side < 0))
        self.right_dnp = np.flatnonzero(self.dnp & (self.side > 0))
        if len(self.left_dnp) == 0 or len(self.right_dnp) == 0:
            raise RuntimeError("DNp pools are empty on one side")
        self._syn_decay = np.float32(np.exp(-NEURAL_DT / TAU_SYN))
        self._adapt_decay = np.float32(np.exp(-NEURAL_DT / TAU_ADAPT))
        self._trace_decay = np.float32(np.exp(-NEURAL_DT / RATE_TRACE_S))
        self._vis_decay = np.float32(np.exp(-NEURAL_DT / SPIKE_VIS_S))
        self._step_scale = np.float32(NEURAL_DT / TAU_M)
        self.reset(seed)

    def reset(self, seed: int) -> None:
        shape = (self.n, self.n_neurons)
        self.v = np.zeros(shape, np.float32)
        self.current = np.zeros(shape, np.float32)
        self.adapt = np.zeros(shape, np.float32)
        self.refr = np.zeros(shape, np.int16)
        self.trace = np.zeros(shape, np.float32)
        self.vis = np.zeros(shape, np.float32)
        self.spiked = np.zeros(shape, np.float32)
        self.rate_hz = np.zeros(shape, np.float32)
        self._rng = np.random.default_rng(seed)

    def advance(self, drive: np.ndarray, steps: int) -> None:
        """drive is (agents, columns) in [0, 1]."""

        visual = np.zeros((self.n, self.n_neurons), np.float32)
        visual[:, self.driven] = np.asarray(drive, dtype=np.float32)[:, self.driven_column] * INPUT_GAIN
        for _ in range(steps):
            self.current *= self._syn_decay
            # Same sum as spiked @ W. Only the neurons that fired are touched,
            # which is the whole nonzero pattern of that product.
            for agent in range(self.n):
                firing = np.flatnonzero(self.spiked[agent])
                if firing.size:
                    pulse = np.asarray(self.weights[firing].sum(axis=0), dtype=np.float32).ravel()
                    self.current[agent] += pulse
            noise = self._rng.normal(0.0, 0.015, size=(self.n, self.driven.shape[0])).astype(np.float32)
            self.current[:, self.driven] += visual[:, self.driven] + noise
            self.v += (self.current - self.v - self.adapt) * self._step_scale
            ready = self.refr == 0
            fired = (self.v >= V_THRESHOLD) & ready
            self.v = np.where(fired, np.float32(0.0), self.v)
            self.refr = np.where(fired, np.int16(REFRACTORY_STEPS), np.maximum(self.refr - 1, 0)).astype(np.int16)
            self.adapt *= self._adapt_decay
            self.adapt += fired.astype(np.float32) * np.float32(ADAPT_JUMP)
            self.trace *= self._trace_decay
            self.trace += fired.astype(np.float32)
            self.vis *= self._vis_decay
            self.vis += fired.astype(np.float32)
            self.spiked = fired.astype(np.float32)
        self.rate_hz = self.trace / np.float32(RATE_TRACE_S)

    def descending(self) -> tuple[np.ndarray, np.ndarray]:
        left = self.rate_hz[:, self.left_dnp].mean(axis=1)
        right = self.rate_hz[:, self.right_dnp].mean(axis=1)
        return left.astype(np.float64), right.astype(np.float64)

    def brightest_type(self, agent: int) -> str:
        vis = self.vis[agent]
        if float(vis.max()) <= 0.0:
            return "quiet"
        return str(self.types[int(np.argmax(vis))])
