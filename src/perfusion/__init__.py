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
    adaptation,
    darcy,
    darcy3d,
    direct,
    geometry,
    multicompartment,
    network,
    parameters,
    perfusion_metrics,
    pipeline,
    pipeline3d,
    timeofflight,
    tkm,
    transient,
    transport,
    viscosity,
)
from .core.darcy import Grid
from .core.geometry import Network, arterial_tree, grow_tree_3d, venous_tree
from .core.parameters import Parameters
from .core.perfusion_metrics import transit_metrics
from .core.pipeline import (
    AdaptationResult,
    HybridResult,
    MultiCompartmentResult,
    run_adaptation,
    run_hybrid,
    run_multicompartment,
)
from .core.pipeline3d import HybridResult3D, run_hybrid_3d

__version__ = "0.1.0"

__all__ = [
    "AdaptationResult",
    "Grid",
    "HybridResult",
    "HybridResult3D",
    "MultiCompartmentResult",
    "Network",
    "Parameters",
    "adaptation",
    "arterial_tree",
    "darcy",
    "darcy3d",
    "direct",
    "geometry",
    "grow_tree_3d",
    "growth",
    "multicompartment",
    "network",
    "parameters",
    "perfusion_metrics",
    "pipeline",
    "pipeline3d",
    "run_adaptation",
    "run_hybrid",
    "run_hybrid_3d",
    "run_multicompartment",
    "timeofflight",
    "tkm",
    "transient",
    "transit_metrics",
    "transport",
    "venous_tree",
    "viscosity",
]
