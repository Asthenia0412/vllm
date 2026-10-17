# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OffloadPolicy implementations and factory.

Mirrors tests/v1/kv_offload/cpu/policies/test_factory.py which covers
the sibling CachePolicy factory.
"""
from collections.abc import Iterable

import pytest

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, ScheduleEndContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.offload_factory import OffloadPolicyFactory
from vllm.v1.kv_offload.cpu.policies.store_policy import (
    OffloadPolicy,
    PreemptionOnlyPolicy,
    StoreOnComputePolicy,
)


class _DummyOffloadPolicy(OffloadPolicy):
    """Minimal OffloadPolicy for factory registration tests."""

    def __init__(self, always_store: bool = True) -> None:
        self.always_store = always_store

    def should_store(
        self,
        req_id: str,
        preempted_req_ids: set[str],
        is_finished: bool,
    ) -> bool:
        return self.always_store


def _make_key(n: int = 1) -> OffloadKey:
    """Synthesize an OffloadKey without requiring a hash function."""
    return OffloadKey(f"hash{n:032d}".encode() + (0).to_bytes(4, "big"))


@pytest.fixture(autouse=True)
def restore_offload_policy_registry():
    """Save and restore OffloadPolicyFactory._registry between tests."""
    original = dict(OffloadPolicyFactory._registry)
    yield
    OffloadPolicyFactory._registry = original


# ---------------------------------------------------------------------------
# OffloadPolicy unit tests
# ---------------------------------------------------------------------------


class TestStoreOnComputePolicy:
    """StoreOnComputePolicy stores unconditionally every step."""

    def test_always_returns_true(self):
        policy = StoreOnComputePolicy()
        # active non-preempted request -> True
        assert policy.should_store("r1", set(), False) is True
        # preempted request -> True
        assert policy.should_store("r1", {"r1"}, False) is True
        # finished request -> True
        assert policy.should_store("r1", set(), True) is True
        # preempted + finished -> True
        assert policy.should_store("r1", {"r1"}, True) is True
        # other request preempted, this one not -> True
        assert policy.should_store("r1", {"r2"}, False) is True


class TestPreemptionOnlyPolicy:
    """PreemptionOnlyPolicy skips active non-preempted requests."""

    def test_skips_active_non_preempted(self):
        policy = PreemptionOnlyPolicy()
        assert policy.should_store("r1", set(), False) is False
        assert policy.should_store("r1", {"r2", "r3"}, False) is False

    def test_stores_preempted(self):
        policy = PreemptionOnlyPolicy()
        assert policy.should_store("r1", {"r1"}, False) is True
        assert policy.should_store("r1", {"r1", "r2"}, False) is True

    def test_stores_finished(self):
        policy = PreemptionOnlyPolicy()
        assert policy.should_store("r1", set(), True) is True
        # preempted by someone else, but this request is finished
        assert policy.should_store("r1", {"r2"}, True) is True

    def test_stores_preempted_and_finished(self):
        policy = PreemptionOnlyPolicy()
        assert policy.should_store("r1", {"r1"}, True) is True


# ---------------------------------------------------------------------------
# OffloadPolicyFactory tests
# ---------------------------------------------------------------------------


class TestOffloadPolicyFactory:
    """Unit tests for OffloadPolicyFactory (registration/resolution by name)."""

    def test_pre_registered_policies_can_be_imported(self):
        for name in OffloadPolicyFactory._registry:
            cls = OffloadPolicyFactory._registry[name]()
            assert issubclass(cls, OffloadPolicy)

    def test_builtin_policies_registered(self):
        assert OffloadPolicyFactory.get_offload_policy_cls(
            "store_on_compute"
        ) is StoreOnComputePolicy
        assert OffloadPolicyFactory.get_offload_policy_cls(
            "preemption_only"
        ) is PreemptionOnlyPolicy

    def test_register_and_resolve_custom_policy(self):
        OffloadPolicyFactory.register_offload_policy(
            "dummy",
            "tests.v1.kv_offload.cpu.policies.test_store_policy",
            "_DummyOffloadPolicy",
        )
        policy_cls = OffloadPolicyFactory.get_offload_policy_cls("dummy")
        assert policy_cls is _DummyOffloadPolicy

    def test_unregistered_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown offload policy"):
            OffloadPolicyFactory.get_offload_policy_cls("nonexistent")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="is already registered"):
            OffloadPolicyFactory.register_offload_policy(
                "store_on_compute", "some.module", "SomeClass"
            )

    def test_dynamic_load_via_module_path(self):
        policy_cls = OffloadPolicyFactory.get_offload_policy_cls(
            "_DummyOffloadPolicy",
            "tests.v1.kv_offload.cpu.policies.test_store_policy",
        )
        assert policy_cls is _DummyOffloadPolicy


# ---------------------------------------------------------------------------
# Integration: CPUOffloadingManager wiring
# ---------------------------------------------------------------------------


class TestManagerOffloadPolicyWiring:
    """End-to-end: CPUOffloadingManager respects OffloadPolicy in prepare_store."""

    def _new_req_context(self, req_id: str) -> ReqContext:
        return ReqContext(req_id=req_id)

    def _schedule_end(
        self,
        manager: CPUOffloadingManager,
        preempted: Iterable[str] = (),
        finished: Iterable[str] = (),
    ) -> None:
        manager.on_schedule_end(
            ScheduleEndContext(
                new_req_ids=(),
                preempted_req_ids=tuple(preempted),
                finished_req_ids=tuple(finished),
            )
        )

    def test_store_on_compute_always_stores(self):
        manager = CPUOffloadingManager(num_chunks=16, offload_policy="store_on_compute")
        self._schedule_end(manager)

        ctx = self._new_req_context("r1")
        out = manager.prepare_store([_make_key(1)], ctx)
        assert out is not None
        assert len(out.keys_to_store) == 1

    def test_preemption_only_skips_active_non_preempted(self):
        manager = CPUOffloadingManager(num_chunks=16, offload_policy="preemption_only")
        self._schedule_end(manager)

        ctx = self._new_req_context("r1")
        out = manager.prepare_store([_make_key(1)], ctx)
        assert out is not None
        assert out.keys_to_store == []

    def test_preemption_only_stores_preempted(self):
        manager = CPUOffloadingManager(num_chunks=16, offload_policy="preemption_only")
        self._schedule_end(manager, preempted=["r1"])

        ctx = self._new_req_context("r1")
        out = manager.prepare_store([_make_key(1)], ctx)
        assert out is not None
        assert len(out.keys_to_store) == 1

    def test_preemption_only_stores_finished(self):
        manager = CPUOffloadingManager(num_chunks=16, offload_policy="preemption_only")
        self._schedule_end(manager, finished=["r1"])

        ctx = self._new_req_context("r1")
        out = manager.prepare_store([_make_key(1)], ctx)
        assert out is not None
        assert len(out.keys_to_store) == 1

    def test_default_policy_is_store_on_compute(self):
        """Backward-compat: default is unchanged behavior."""
        manager = CPUOffloadingManager(num_chunks=16)
        assert isinstance(manager._offload_policy, StoreOnComputePolicy)
