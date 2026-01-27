from enum import Enum, unique
try:
    from enum import StrEnum  # Python 3.11+
except ImportError:
    class StrEnum(str, Enum):
        def __str__(self):
            return str(self.value)

from typing import Any, TypeAlias

from graph_uq.experiment import experiment


@unique
class PlotType(StrEnum):
    UNCERTAINTY_DISTRIBUTION = "uncertainty_distribution"


PlottingConfig: TypeAlias = dict[str, Any]


@experiment.config
def default_plotting_config():
    plot = dict()
