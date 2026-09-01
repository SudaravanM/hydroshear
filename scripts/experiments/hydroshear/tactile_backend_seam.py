"""The seam a direct RealSim backend must satisfy, plus a latency-parameterised stand-in.

Contract, read from drawer_task_pulling.py:468-487 rather than assumed:

    get_force_fields_dict(handle_quat, handle_pos)
        -> {'elastomer_left': (N, 9, 7, 3), 'elastomer_right': (N, 9, 7, 3)}
           torch tensors, on the env's device, matching its dtype

    the env then does  [..., :2].flip(1, -1)  and  [..., 0] *= -1
    to produce the policy's (N, 9, 7, 2). The NORMAL CHANNEL NEVER REACHES THE POLICY.

Why a stand-in is worth building before the real backend exists: it makes the F4 question
answerable NOW. Sweeping `latency_ms` and measuring end-to-end agent-steps/s yields the LATENCY
BUDGET -- how fast a tactile backend must be for a given throughput target -- without needing a
working RealSim. That budget is what F3's per-substep numbers have to be judged against, and it is
useful whichever way the direct-vs-surrogate decision goes.

Deliberately NOT a physics model. It returns structurally valid fields with the right shapes,
device and dtype, and burns a configurable amount of wall-clock. Any use of it to make a
PHYSICS claim would be misuse; it exists to measure plumbing and throughput.
"""
import time

import torch

GRID = (9, 7, 3)
KEYS = ("elastomer_left", "elastomer_right")


class TactileBackendSeam:
    """Base: what any direct backend must provide. Subclass and override compute_fields."""

    def __init__(self, num_envs, device, dtype=torch.float32):
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self.calls = 0
        self.total_s = 0.0

    def compute_fields(self, handle_quat, handle_pos):
        raise NotImplementedError

    def get_force_fields_dict(self, handle_quat, handle_pos):
        t0 = time.perf_counter()
        out = self.compute_fields(handle_quat, handle_pos)
        for k in KEYS:
            assert k in out, f"backend must return '{k}'"
            t = out[k]
            assert tuple(t.shape) == (self.num_envs, *GRID), \
                f"{k}: expected {(self.num_envs, *GRID)}, got {tuple(t.shape)}"
            assert t.device.type == torch.device(self.device).type, \
                f"{k}: on {t.device}, env expects {self.device}"
        self.calls += 1
        self.total_s += time.perf_counter() - t0
        return out

    def reset_idx(self, env_ids):
        """Per-environment reset. A stateful backend MUST clear only these environments;
        GT-DESIGN-009 requires per-env reset correctness, and Isaac auto-resets finished envs."""
        pass

    def stats(self):
        return {"calls": self.calls, "total_s": round(self.total_s, 4),
                "ms_per_call": round(1000.0 * self.total_s / max(self.calls, 1), 4)}


class LatencyDummyBackend(TactileBackendSeam):
    """Structurally valid fields, configurable wall-clock cost. NOT physics.

    `latency_ms` is spent on the GPU when the env is on CUDA, so it is exposed to the same
    synchronisation the real backend would be, rather than a host sleep that a real GPU backend
    would not experience.
    """

    def __init__(self, num_envs, device, dtype=torch.float32, latency_ms=0.0, amplitude=9e-4):
        super().__init__(num_envs, device, dtype)
        self.latency_ms = float(latency_ms)
        self.amplitude = float(amplitude)
        self._buf = {k: torch.zeros((self.num_envs, *GRID), device=device, dtype=dtype)
                     for k in KEYS}
        self._burn = None
        self._resets = 0

    def _spend(self):
        if self.latency_ms <= 0:
            return
        if torch.device(self.device).type == "cuda":
            if self._burn is None:
                self._burn = torch.randn(512, 512, device=self.device, dtype=torch.float32)
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) * 1000.0 < self.latency_ms:
                self._burn = torch.mm(self._burn, self._burn).clamp_(-1e3, 1e3)
            torch.cuda.synchronize()
        else:
            time.sleep(self.latency_ms / 1000.0)

    def compute_fields(self, handle_quat, handle_pos):
        self._spend()
        # a deterministic, handle-pose-dependent pattern: exercises the same tensor path as a real
        # backend without pretending to be physics
        p = handle_pos.reshape(self.num_envs, -1)[:, :1].to(self.dtype)
        for k in KEYS:
            self._buf[k].copy_(
                (p.view(-1, 1, 1, 1) * torch.ones((1, *GRID), device=self.device, dtype=self.dtype))
                * self.amplitude)
        return self._buf

    def reset_idx(self, env_ids):
        for k in KEYS:
            self._buf[k][env_ids] = 0.0
        self._resets += len(env_ids)

    def stats(self):
        d = super().stats()
        d.update(latency_ms=self.latency_ms, resets=self._resets)
        return d
