"""This module contains utility functions for the metric toolbox."""

from enum import Enum
from typing import TypeVar

E = TypeVar("E", bound=Enum)


def value_to_enum(value: str | Enum, enum_class: type[E]) -> E:
    """Convert a string values to Enum.

    If the value is already an enum, it is returned as is.
    String values are automacitaly uppercased and stripped to match enum names.

    Parameters
    ----------
    value : str
        The string representation of the enum value.
    enum_class : type[Enum]
        The enum class to convert to.

    Raises
    ------
    ValueError
        If the value cannot be converted to the specified enum class,
        supported values are listed in the error message.

    Returns
    -------
    Enum
        The corresponding enum value.
    """
    if isinstance(value, enum_class):
        return value

    value = str(value).upper().strip()
    try:
        return enum_class[value]
    except Exception:
        raise ValueError(
            f'Cannot convert "{value}" to {enum_class}.\nSupported values: {[e.name for e in enum_class]}'
        )
