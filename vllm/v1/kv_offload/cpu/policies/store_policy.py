# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy abstraction for deciding *when* to store computed KV blocks to the
CPU offload cache. Orthogonal to :class:`CachePolicy` which decides *which*
blocks to evict when the cache is full.

``OffloadPolicy`` controls whether ``prepare_store`` is allowed to proceed at
all for a given request. Two built-in policies ship here:

* :class:`StoreOnComputePolicy` -- the default. Every compute step offers the
  newly-computed chunks for offloading, matching pre-abstraction behavior.
* :class:`PreemptionOnlyPolicy` -- conserves PCIe bandwidth by only storing
  when a request is preempted or has finished generating. Useful for
  memory-constrained deployments where preemption is rare.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable

from typing_extensions import override

class OffloadPolicy(ABC):
    """Decides *when* to store computed KV blocks to the CPU offload cache.

    The policy is consulted in ``prepare_store``. Returning ``False`` skips
    the store entirely -- no eviction, no allocation, no transfer job is
    created for this request this step.
    """

    @abstractmethod
    def should_store(
        self,
        req_id: str,
        preempted_req_ids: set[str],
        is_finished: bool,
    ) -> bool:
        """Whether to proceed with storing blocks for this request this step.

        Args:
            req_id: The request ID being considered for store.
            preempted_req_ids: Set of request IDs preempted this scheduler step.
            is_finished: Whether the request has finished generating tokens.

        Returns:
            ``True`` if ``prepare_store`` should proceed, ``False`` to skip.
        """

    def update_preempted(self, preempted_req_ids: Iterable[str]) -> None:
        """Called at the end of each scheduler step with the full set of
        preempted request IDs. Default is a no-op; policies that need to
        track state across steps may override this.
        """
        return


class StoreOnComputePolicy(OffloadPolicy):
    """Store chunks as soon as they are computed.

    This is the default policy and reproduces the pre-abstraction behavior:
    every compute step offers newly-computed chunks for offloading.
    """

    @override
    def should_store(
        self,
        req_id: str,
        preempted_req_ids: set[str],
        is_finished: bool,
    ) -> bool:
        return True


class PreemptionOnlyPolicy(OffloadPolicy):
    """Only store blocks when a request is preempted or has finished.

    Skips store for active, non-preempted requests to conserve PCIe
    bandwidth. When a request is preempted, its partially-computed KV is
    preserved so a future request sharing its prefix can reuse it. Finished
    requests are always stored because their GPU blocks will be freed.
    """

    @override
    def should_store(
        self,
        req_id: str,
        preempted_req_ids: set[str],
        is_finished: bool,
    ) -> bool:
        return req_id in preempted_req_ids or is_finished
