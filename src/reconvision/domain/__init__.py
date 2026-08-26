"""Pure domain layer.

Nothing in this package may import OpenCV, ONNX Runtime, SQL drivers or any web
framework. Keeping the recognition rules free of I/O is what lets the whole test
suite run in under a second without a model or a camera.
"""
