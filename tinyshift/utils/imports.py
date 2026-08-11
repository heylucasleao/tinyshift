# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""
Dependency checking utilities for optional package management.
"""

import importlib.util
from functools import wraps
from typing import Callable, Any, Optional


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is required but missing."""

    pass


def check_dependency(
    module_name: str, extra_name: str, pip_name: Optional[str] = None
) -> bool:
    """
    Check whether a Python package is installed without importing it.

    Parameters
    ----------
    module_name : str
        Python import name of the package (e.g., "plotly").
    extra_name : str
        Name of the optional dependency group defined in pyproject.toml (e.g., "plot").
    pip_name : str, optional
        Name of the package on PyPI if different from `module_name`.

    Returns
    -------
    bool
        True if the package is found in the environment.

    Raises
    ------
    MissingDependencyError
        If the package is not installed.
    """
    pkg_name = pip_name or module_name
    if importlib.util.find_spec(module_name) is None:
        raise MissingDependencyError(
            f"The requested feature requires '{pkg_name}'. "
            f"Install it using: pip install 'tinyshift[{extra_name}]' "
            f"or uv add 'tinyshift[{extra_name}]'"
        )
    return True


def requires_extra(extra_name: str, *modules: str):
    """
    Decorator to guard functions or methods that depend on optional packages.

    Parameters
    ----------
    extra_name : str
        Name of the optional dependency group.
    *modules : tuple of str
        Module names to check before executing the decorated function.

    Returns
    -------
    Callable
        Wrapped function that verifies dependencies prior to execution.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for module in modules:
                check_dependency(module, extra_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator
