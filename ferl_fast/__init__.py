"""ferl_fast: Cython-accelerated FERL (Fast Evidential Rule Learning).

Separate, self-contained implementation that mirrors the reference
``FuzzyCART`` in ``tree_learning.py``. The Python layer owns the tree
structure and growth; the compiled ``_kernels`` module owns the hot
per-candidate split scoring and vote accumulation.

Build the compiled kernels with::

    <env-python> ferl_fast/setup.py build_ext --inplace
"""

from .tree_fast import FuzzyCARTFast
from .fuzzification_fast import learn_partitions_mdlp_fast

__all__ = ["FuzzyCARTFast", "learn_partitions_mdlp_fast"]
