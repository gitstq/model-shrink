"""ModelShrink — a tiny, dependency-light neural-network compression toolbox.

ModelShrink bundles the three most widely used post-training / training-time
compression techniques — **quantization**, **pruning** and **knowledge
distillation** — into a single, pure-NumPy toolkit that runs on the CPU in
seconds. It is designed for learning, rapid prototyping and teaching, with no
heavy deep-learning framework or GPU required.

Public API
----------
- :class:`~modelshrink.nn.MLP`            tiny trainable NumPy MLP
- :func:`~modelshrink.data.load_synthetic` built-in separable dataset
- :func:`~modelshrink.quantize.quantize_tensor`
- :func:`~modelshrink.prune.magnitude_prune`
- :func:`~modelshrink.distill.distill`
- :mod:`~modelshrink.metrics`             compression / error metrics
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
