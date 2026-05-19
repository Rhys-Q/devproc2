"""PyTorch-like nn frontend for devproc2."""
from devproc2.nn.builder import GraphBuilder, TraceValue
from devproc2.nn.layers import Embedding, GELU, LayerNorm, Linear, RMSNorm, SiLU
from devproc2.nn.module import Module, ModuleList, Sequential
from devproc2.nn.specs import ObjectSpec, Parameter, ScalarSpec, TensorSpec

__all__ = [
    "Embedding",
    "GELU",
    "GraphBuilder",
    "LayerNorm",
    "Linear",
    "Module",
    "ModuleList",
    "ObjectSpec",
    "Parameter",
    "RMSNorm",
    "ScalarSpec",
    "Sequential",
    "SiLU",
    "TensorSpec",
    "TraceValue",
]
