"""anywidget wrapper (optional dependency: ``pip install anywidget``)."""

import pathlib as _pathlib

import anywidget as _anywidget
import traitlets as _traitlets

from splinepy.viewer import scene as _scene

_ESM = (_pathlib.Path(__file__).parent / "static" / "viewer.bundle.js").read_text(
    encoding="utf-8"
)


class ViewerWidget(_anywidget.AnyWidget):
    """Interactive widget for Jupyter, JupyterLab, VS Code, Colab, marimo."""

    _esm = _ESM
    scene_json = _traitlets.Unicode("{}").tag(sync=True)
    height = _traitlets.Int(480).tag(sync=True)

    def __init__(self, scene, height=480, **kwargs):
        super().__init__(scene_json=_scene.to_json(scene), height=height, **kwargs)

    def set_scene(self, scene):
        """Replaces the displayed scene without recreating the widget."""
        self.scene_json = _scene.to_json(scene)
