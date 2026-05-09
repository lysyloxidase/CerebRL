"""Static quality contracts for the web visualizers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewerQuality:
    component: str
    target_fps: int
    cell_types: tuple[str, ...]
    animated_latencies: bool

    @property
    def renders_all_microzone_cell_types(self) -> bool:
        required = {"GrC", "GoC", "MLI", "PC", "DCN", "IO"}
        return required.issubset(set(self.cell_types))


def cerebellar_viewer_quality() -> ViewerQuality:
    return ViewerQuality(
        component="CerebellarCircuit",
        target_fps=30,
        cell_types=("GrC", "GoC", "MLI", "PC", "DCN", "IO"),
        animated_latencies=False,
    )


def loop_diagram_quality() -> ViewerQuality:
    return ViewerQuality(
        component="LoopDiagram",
        target_fps=30,
        cell_types=("M1", "pontine", "GrC", "PC", "DCN", "VL", "IO"),
        animated_latencies=True,
    )
