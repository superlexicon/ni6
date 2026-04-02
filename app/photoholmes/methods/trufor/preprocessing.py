from ...preprocessing.image import ZeroOneRange
from ...preprocessing.pipeline import PreProcessingPipeline

trufor_preprocessing = PreProcessingPipeline(
    inputs=["image"], outputs_keys=["image"], transforms=[ZeroOneRange()]
)
