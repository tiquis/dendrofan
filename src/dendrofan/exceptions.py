"""Exceptions raised by dendrofan."""


class DendrofanError(Exception):
    """Base class for all errors raised by dendrofan."""


class InvalidLinkageError(DendrofanError):
    """Raised when a linkage matrix fails SciPy's own consistency checks."""


class LabelMismatchError(DendrofanError):
    """Raised when the number of labels does not match the number of leaves."""


class DegenerateTreeError(DendrofanError):
    """Raised when a tree has too few leaves to be drawn (n < 2)."""
