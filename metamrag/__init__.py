"""MetaMed-RAG package."""

from .calibration import TemperatureScaler
from .confidence import build_feature_vector
from .data import DatasetFormatError, TrainingSample, load_training_samples
from .evaluation import ClassificationMetrics, build_metrics
from .experts import Expert, ExpertOutput, MockExpert
from .meta_model import LinearMetaModel
from .pipeline import MetaRAGPipeline, MetaPrediction, TrainingReport

__all__ = [
    "TemperatureScaler",
    "DatasetFormatError",
    "TrainingSample",
    "load_training_samples",
    "ClassificationMetrics",
    "build_metrics",
    "Expert",
    "ExpertOutput",
    "MockExpert",
    "build_feature_vector",
    "LinearMetaModel",
    "MetaRAGPipeline",
    "MetaPrediction",
    "TrainingReport",
]
