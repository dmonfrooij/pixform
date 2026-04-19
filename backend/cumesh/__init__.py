"""
cumesh - stub package for PIXFORM compatibility.

The real CuMesh library provides CUDA-accelerated mesh processing for TRELLIS.2.
This stub allows TRELLIS.2 to load without the compiled CuMesh extension.
Operations that truly require CuMesh will raise NotImplementedError at call time,
but the pipeline can still run via fallback paths.
"""

__version__ = "0.0.0+stub"

class _StubModule:
    """Lazy stub: raises NotImplementedError on actual use."""
    def __init__(self, name):
        self._name = name

    def __getattr__(self, item):
        def _not_implemented(*args, **kwargs):
            raise NotImplementedError(
                f"cumesh.{self._name}.{item} is not available (native CuMesh not installed). "
                "Install the CUDA Toolkit + VS Build Tools and re-run install.ps1 to enable full CuMesh support."
            )
        return _not_implemented

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"cumesh.{self._name} is not available (native CuMesh not installed)."
        )


# Expose common cumesh sub-modules as stubs so `from cumesh import X` works.
mesh = _StubModule("mesh")
ops  = _StubModule("ops")

