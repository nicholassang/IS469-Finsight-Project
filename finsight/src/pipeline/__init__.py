from .baseline import BaselinePipeline
from .advanced_a import AdvancedAPipeline
from .advanced_b import AdvancedBPipeline

ALL_PIPELINES = {
    "v1_baseline": BaselinePipeline,
    "v2_advanced_a": AdvancedAPipeline,
    "v3_advanced_b": AdvancedBPipeline,
}

__all__ = ["BaselinePipeline", "AdvancedAPipeline", "AdvancedBPipeline", "ALL_PIPELINES"]
