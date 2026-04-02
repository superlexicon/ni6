from ...preprocessing.image import ToNumpy
from ...preprocessing.pipeline import PreProcessingPipeline

noisesniffer_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image"],
    transforms=[ToNumpy()],
)
