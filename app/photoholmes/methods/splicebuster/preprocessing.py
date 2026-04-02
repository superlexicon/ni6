from ...preprocessing.image import RGBtoGray, ToNumpy, ZeroOneRange
from ...preprocessing.pipeline import PreProcessingPipeline

splicebuster_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image"],
    transforms=[ZeroOneRange(), RGBtoGray(), ToNumpy()],
)
