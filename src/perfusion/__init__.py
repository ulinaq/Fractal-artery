"""perfusion -- multiscale blood circulation and perfusion models.

Python port of the models from U. N. A. Qohar's PhD thesis, *Modeling and
simulation of blood circulation and perfusion* (University of Bergen, 2023).

Layout
------
- :mod:`perfusion.core` -- pure compute engine (imported here for convenience as
  ``perfusion.parameters``, ``perfusion.darcy``, ``perfusion.geometry``,
  ``perfusion.network``, ``perfusion.direct``, ``perfusion.transport``,
  ``perfusion.tkm``, ``perfusion.timeofflight``, ``perfusion.viscosity``,
  ``perfusion.pipeline``).
- :mod:`perfusion.io`  -- data loaders / exporters.
- :mod:`perfusion.viz` -- matplotlib figures.
- :mod:`perfusion.segmentation` -- image -> vascular network.
- :mod:`perfusion.growth` -- vessel growth by space colonisation.
- :mod:`perfusion.optimize` -- vascular-structure optimisation (Paper D).

See ``docs/ROADMAP.md`` for the ``api`` (FastAPI back-end) and the planned web
front-end layers.
"""

from . import growth
from .core import (
    darcy,
    direct,
    geometry,
    network,
    parameters,
    pipeline,
    timeofflight,
    tkm,
    transport,
    viscosity,
)
from .core.darcy import Grid
from .core.geometry import Network, arterial_tree, venous_tree
from .core.parameters import Parameters
from .core.pipeline import HybridResult, run_hybrid

__version__ = "0.1.0"

__all__ = [
    "Grid",
    "HybridResult",
    "Network",
    "Parameters",
    "arterial_tree",
    "darcy",
    "direct",
    "geometry",
    "growth",
    "network",
    "parameters",
    "pipeline",
    "run_hybrid",
    "timeofflight",
    "tkm",
    "transport",
    "venous_tree",
    "viscosity",
]
