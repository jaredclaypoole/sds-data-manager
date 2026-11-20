"""Construct product-specific dependency handlers without the Dagster runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, TypeVar

from sds_data_manager.orchestration.job_dependencies import JobDependencyHandler

if TYPE_CHECKING:
    from sds_data_manager.orchestration.dependency_metadata import DependencySource
    from sds_data_manager.orchestration.types import ProcessingJobNode

HandlerType = TypeVar("HandlerType", bound=type[JobDependencyHandler])
JobKey = tuple[str, str, str]


class DependencyHandlerRegistry:
    """Register selection rules separately from production scheduling behavior."""

    _registry: ClassVar[dict[JobKey, type[JobDependencyHandler]]] = {}

    @classmethod
    def register(
        cls, source: str, data_type: str, descriptor: str
    ) -> Callable[[HandlerType], HandlerType]:
        """Register a dependency handler for one configured job key."""

        def decorate(handler: HandlerType) -> HandlerType:
            key = (source, data_type, descriptor)
            if key in cls._registry:
                raise ValueError(f"Duplicate dependency handler for {key}")
            cls._registry[key] = handler
            return handler

        return decorate

    @classmethod
    def keys_for(cls, handler: type[JobDependencyHandler]) -> list[JobKey]:
        """Return the keys for a shared dependency handler."""
        return [
            key for key, registered in cls._registry.items() if registered is handler
        ]

    @classmethod
    def get_handler(
        cls,
        job_config: ProcessingJobNode,
        source: DependencySource,
        logger: logging.Logger | None = None,
    ) -> JobDependencyHandler:
        """Construct a registered handler without building assets, sensors, or jobs."""
        from sds_data_manager.orchestration.custom_behavior import (  # noqa: PLC0415
            dependencies,  # noqa: F401
        )

        key = (job_config.source, job_config.data_type, job_config.descriptor)
        handler = cls._registry.get(key, JobDependencyHandler)
        return handler(job_config, source, logger)
