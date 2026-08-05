"""Utility to create a lazy-loaded AWS client."""

from collections.abc import Callable


class LazyClient:
    """Class to create a lazy-loaded object."""

    def __init__(self, factory: Callable, *args, **kwargs):
        """Create the lazy-loaded object.

        Does not create the underlying object until an attribute is requested of it.

        Parameters
        ----------
        factory : Callable
            The function to call to construct the underlying lazy-loaded object
        args
            Optional positional arguments to the factory function
        kwargs
            Optional keyword arguments to the factory function
        """
        self._factory = factory
        self._args = args
        self._kwargs = kwargs
        self._instance = None

    def __getattr__(self, name: str):
        """Get attributes from the underlying object.

        Create and cache the underlying object if it has not yet been created.
        """
        if self._instance is None:
            instance = self._factory(*self._args, **self._kwargs)
            self._instance = instance
        return getattr(self._instance, name)
