# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for :class:`OffloadPolicy` implementations, resolved by name."""
import importlib
from collections.abc import Callable

from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.policies.store_policy import OffloadPolicy

logger = init_logger(__name__)


class OffloadPolicyFactory:
    """Registry for OffloadPolicy implementations, resolved by name.

    Mirrors :class:`CachePolicyFactory`: built-in policies are pre-registered
    below. External policies can either register a friendly short name up
    front, or skip registration entirely and pass a module path at lookup
    time (out-of-tree, no vLLM fork/patch required).
    """

    _registry: dict[str, Callable[[], type[OffloadPolicy]]] = {}

    @classmethod
    def register_offload_policy(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register an offload policy with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Offload policy '{name}' is already registered.")

        def loader() -> type[OffloadPolicy]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_offload_policy_cls(
        cls, name: str, module_path: str | None = None
    ) -> type[OffloadPolicy]:
        """Get an offload policy class by name.

        Args:
            name: Name of the offload policy. Checked against the registry
                first; if it is not registered and ``module_path`` is given,
                ``name`` is imported from there instead.
            module_path: Python import path to load ``name`` from when it is
                not a registered policy.

        Returns:
            The offload policy class.

        Raises ValueError if the policy is neither registered nor resolvable
        via ``module_path``.
        """
        if name in cls._registry:
            return cls._registry[name]()
        if module_path is None:
            raise ValueError(
                f"Unknown offload policy: {name!r}. Supported: {list(cls._registry)}. "
                "For an out-of-tree policy, also set offload_policy_module_path."
            )
        logger.warning_once(
            "Loading out-of-tree offload policy. This API is "
            "experimental and subject to change in the future "
            "as we iterate the design."
        )
        logger.info(
            "Loading out-of-tree offload policy '%s' from '%s'.",
            name,
            module_path,
        )
        module = importlib.import_module(module_path)
        policy_cls = getattr(module, name)
        assert issubclass(policy_cls, OffloadPolicy)
        return policy_cls


# Register built-in policies here.
OffloadPolicyFactory.register_offload_policy(
    "store_on_compute",
    "vllm.v1.kv_offload.cpu.policies.store_policy",
    "StoreOnComputePolicy",
)
OffloadPolicyFactory.register_offload_policy(
    "preemption_only",
    "vllm.v1.kv_offload.cpu.policies.store_policy",
    "PreemptionOnlyPolicy",
)
