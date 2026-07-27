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
    darcy3d,
    direct,
    geometry,
    network,
    parameters,
    perfusion_metrics,
    pipeline,
    pipeline3d,
    timeofflight,
    tkm,
    transport,
    viscosity,
)
from .core.darcy import Grid
from .core.geometry import Network, arterial_tree, grow_tree_3d, venous_tree
from .core.parameters import Parameters
from .core.perfusion_metrics import transit_metrics
from .core.pipeline import HybridResult, run_hybrid
from .core.pipeline3d import HybridResult3D, run_hybrid_3d

__version__ = "0.1.0"

__all__ = [
    "Grid",
    "HybridResult",
    "HybridResult3D",
    "Network",
    "Parameters",
    "arterial_tree",
    "darcy",
    "darcy3d",
    "direct",
    "geometry",
    "grow_tree_3d",
    "growth",
    "network",
    "parameters",
    "perfusion_metrics",
    "pipeline",
    "pipeline3d",
    "run_hybrid",
    "run_hybrid_3d",
    "timeofflight",
    "tkm",
    "transit_metrics",
    "transport",
    "venous_tree",
    "viscosity",
]
