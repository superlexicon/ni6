from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.models import PhotoHolmesResults


class DatasetInfo(BaseModel):
    """Information about the benchmark dataset"""
    name: str = Field(description="Dataset name")
    path: str = Field(description="Dataset file path")
    total_images: int = Field(description="Total number of images in dataset")
    has_ground_truth: bool = Field(description="Whether ground truth annotations are available")
    ground_truth_file: Optional[str] = Field(default=None, description="Path to ground truth file")


class BenchmarkMetrics(BaseModel):
    """Comprehensive metrics for benchmark evaluation"""
    accuracy: float = Field(description="Classification accuracy (0-1)")
    precision: float = Field(description="Precision score (0-1)")
    recall: float = Field(description="Recall score (0-1)")
    f1_score: float = Field(description="F1 score (0-1)")
    auc: float = Field(description="Area under ROC curve (0-1)")
    processing_time: float = Field(description="Processing time in seconds")
    method_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-method detailed metrics"
    )


class BenchmarkResult(BaseModel):
    """Result of benchmarking a single image"""
    image_id: str = Field(description="Unique identifier for the image")
    method_scores: Dict[str, float] = Field(description="Scores for each method")
    processing_time: float = Field(description="Total processing time in seconds")
    metrics: BenchmarkMetrics = Field(description="Computed metrics")
    ground_truth: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Ground truth annotations if available"
    )
    photoholmes_results: Optional[PhotoHolmesResults] = Field(
        default=None,
        description="Original PhotoHolmes analysis results"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Benchmark timestamp")


class ComparisonReport(BaseModel):
    """Comprehensive comparison report across multiple images"""
    dataset_info: DatasetInfo = Field(description="Dataset information")
    method_ranking: Dict[str, float] = Field(
        description="Methods ranked by average performance score"
    )
    comparison_summary: Dict[str, Any] = Field(
        description="Summary statistics and comparisons"
    )
    detailed_results: List[BenchmarkResult] = Field(
        description="Detailed results for each processed image"
    )
    overall_metrics: BenchmarkMetrics = Field(
        description="Overall aggregated metrics across all images"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Report creation timestamp")

    def get_best_method(self) -> str:
        """Get the best performing method"""
        if not self.method_ranking:
            return "No methods evaluated"
        return max(self.method_ranking, key=self.method_ranking.get)

    def get_method_summary(self, method_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific method"""
        if method_name not in self.comparison_summary.get('methods_evaluated', []):
            return {"error": f"Method {method_name} not found in evaluation"}

        method_results = [
            r for r in self.detailed_results
            if method_name in r.method_scores
        ]

        if not method_results:
            return {"error": f"No results found for method {method_name}"}

        scores = [r.method_scores[method_name] for r in method_results]

        return {
            'method_name': method_name,
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'samples_processed': len(method_results),
            'average_processing_time': sum(r.processing_time for r in method_results) / len(method_results)
        }


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution"""
    max_concurrent_images: int = Field(default=4, description="Maximum concurrent image processing")
    score_threshold: float = Field(default=0.5, description="Threshold for binary classification")
    include_method_details: bool = Field(default=True, description="Include detailed method results")
    export_format: str = Field(default="dict", description="Export format (dict, json, csv)")


class BenchmarkSession(BaseModel):
    """Track a complete benchmark session"""
    session_id: str = Field(description="Unique session identifier")
    config: BenchmarkConfig = Field(description="Benchmark configuration")
    start_time: datetime = Field(default_factory=datetime.now, description="Session start time")
    end_time: Optional[datetime] = Field(default=None, description="Session end time")
    status: str = Field(default="initialized", description="Session status")
    progress: Dict[str, Any] = Field(default_factory=dict, description="Progress tracking")

    def mark_completed(self):
        """Mark session as completed"""
        self.end_time = datetime.now()
        self.status = "completed"

    def get_duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None