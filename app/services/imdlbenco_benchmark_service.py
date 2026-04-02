import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from app.dto import (
    PhotoHolmesResults,
    DQMethodData,
    AdaptiveMethodData,
    NoiseSnifferData,
    PsccnetMethodData,
    CatnetMethodData
)
from app.dto.benchmark import (
    BenchmarkMetrics,
    BenchmarkResult,
    ComparisonReport,
    DatasetInfo
)
from app.core.logger import get_logger
from app.services.comprehensive_photoholmes_service import ComprehensivePhotoHolmesService


class IMDLBenCoBenchmarkService:
    """
    IMDLBenCo benchmark service for comprehensive evaluation of forgery detection methods.

    This service provides standardized benchmarking capabilities for all PhotoHolmes methods
    following IMDLBenCo evaluation patterns and metrics.
    """

    def __init__(self):
        self.logger = get_logger()
        # Use the shared comprehensive PhotoHolmes service (singleton)
        from app.services import comprehensive_photoholmes_service
        self.photoholmes_service = comprehensive_photoholmes_service
        self.logger.info("IMDLBenCo Benchmark Service initialized")

    async def benchmark_single_image(
        self,
        image_bytes: bytes,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark all methods on a single image with optional ground truth.

        Args:
            image_bytes: Input image as bytes
            ground_truth: Optional ground truth data for evaluation

        Returns:
            BenchmarkResult: Individual benchmark results for all methods
        """
        self.logger.info("Starting single image benchmark")

        start_time = time.time()

        # Run all PhotoHolmes methods
        try:
            photoholmes_results = await self.photoholmes_service.run_all_methods(image_bytes)
        except Exception as e:
            self.logger.error(f"PhotoHolmes analysis failed: {e}")
            photoholmes_results = PhotoHolmesResults()

        processing_time = time.time() - start_time

        # Extract method scores
        method_scores = {}

        if photoholmes_results.dq:
            method_scores['DQ'] = photoholmes_results.dq.max_probability
        if photoholmes_results.adaptive:
            method_scores['Adaptive'] = photoholmes_results.adaptive.tampered_ratio
        if photoholmes_results.noisesniffer:
            method_scores['NoiseSniffer'] = photoholmes_results.noisesniffer.noise_confidence_score
        if photoholmes_results.psccnet:
            method_scores['PSCCNet'] = photoholmes_results.psccnet.psccnet_confidence_score
        if photoholmes_results.catnet:
            method_scores['IMDLBenCo_CATNet'] = photoholmes_results.catnet.catnet_confidence_score

        # Calculate metrics (with ground truth if available)
        if ground_truth:
            metrics = self._calculate_metrics_with_ground_truth(method_scores, ground_truth)
        else:
            metrics = self._calculate_metrics_without_ground_truth(method_scores)

        # Create benchmark result
        result = BenchmarkResult(
            image_id=ground_truth.get('image_id') if ground_truth else f"img_{int(time.time())}",
            method_scores=method_scores,
            processing_time=processing_time,
            metrics=metrics,
            ground_truth=ground_truth,
            photoholmes_results=photoholmes_results
        )

        self.logger.info(f"Single image benchmark completed in {processing_time:.2f}s")
        return result

    async def benchmark_dataset(
        self,
        dataset_path: Path,
        ground_truth_file: Optional[Path] = None,
        max_images: Optional[int] = None
    ) -> ComparisonReport:
        """
        Benchmark all methods on a dataset of images.

        Args:
            dataset_path: Path to dataset directory
            ground_truth_file: Optional path to ground truth annotations
            max_images: Maximum number of images to process

        Returns:
            ComparisonReport: Comprehensive comparison report
        """
        self.logger.info(f"Starting dataset benchmark: {dataset_path}")

        # Load dataset info
        dataset_info = await self._load_dataset_info(dataset_path, ground_truth_file)

        # Load ground truth if available
        ground_truth_data = {}
        if ground_truth_file and ground_truth_file.exists():
            ground_truth_data = await self._load_ground_truth(ground_truth_file)

        # Process images
        all_results = []
        processed_count = 0

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))

        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]

        self.logger.info(f"Found {len(image_files)} images to process")

        # Process images with concurrency control
        semaphore = asyncio.Semaphore(4)  # Limit concurrent processing

        async def process_single_image(image_path):
            async with semaphore:
                try:
                    # Read image
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()

                    # Get ground truth for this image
                    image_name = image_path.name
                    ground_truth = ground_truth_data.get(image_name, {})
                    ground_truth['image_path'] = str(image_path)

                    # Benchmark single image
                    return await self.benchmark_single_image(image_bytes, ground_truth)

                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
                    return None

        # Process all images
        tasks = [process_single_image(img_path) for img_path in image_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        all_results = [r for r in results if r is not None and not isinstance(r, Exception)]

        self.logger.info(f"Successfully processed {len(all_results)}/{len(image_files)} images")

        # Generate comparison report
        report = await self._generate_comparison_report(
            all_results,
            dataset_info,
            ground_truth_file is not None
        )

        return report

    def _calculate_metrics_with_ground_truth(
        self,
        method_scores: Dict[str, float],
        ground_truth: Dict[str, Any]
    ) -> BenchmarkMetrics:
        """Calculate metrics when ground truth is available"""

        # For simplicity, we'll treat all methods as binary classifiers
        # In a real implementation, you'd use more sophisticated evaluation

        # Get ground truth label
        is_forged = ground_truth.get('is_forged', ground_truth.get('forgery_label', 0))

        # Convert method scores to binary predictions (using 0.5 threshold)
        predictions = {method: score > 0.5 for method, score in method_scores.items()}

        # Calculate basic metrics for each method
        metrics_data = {}

        for method, predicted in predictions.items():
            score = method_scores[method]

            # Simple metrics (would be more sophisticated with multiple samples)
            metrics_data[method] = {
                'score': score,
                'predicted_forgery': predicted,
                'ground_truth_forgery': bool(is_forged),
                'correct': predicted == bool(is_forged)
            }

        # Calculate aggregated metrics
        if metrics_data:
            accuracy = np.mean([m['correct'] for m in metrics_data.values()])

            # For single sample, other metrics are approximations
            precision = accuracy  # Approximation
            recall = accuracy     # Approximation
            f1 = accuracy         # Approximation
            auc = np.mean(list(method_scores.values()))  # Approximation
        else:
            accuracy = precision = recall = f1 = auc = 0.0

        return BenchmarkMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc=float(auc),
            processing_time=0.0,  # Will be set by caller
            method_metrics=metrics_data
        )

    def _calculate_metrics_without_ground_truth(
        self,
        method_scores: Dict[str, float]
    ) -> BenchmarkMetrics:
        """Calculate basic metrics when no ground truth is available"""

        if not method_scores:
            return BenchmarkMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0, auc=0.0,
                processing_time=0.0, method_metrics={}
            )

        # Basic statistics
        scores = list(method_scores.values())

        return BenchmarkMetrics(
            accuracy=0.0,  # Cannot compute without ground truth
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc=np.mean(scores),  # Use average score as proxy for AUC
            processing_time=0.0,  # Will be set by caller
            method_metrics={method: {'score': score} for method, score in method_scores.items()}
        )

    async def _load_dataset_info(self, dataset_path: Path, ground_truth_file: Optional[Path]) -> DatasetInfo:
        """Load dataset information"""

        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        total_images = 0
        for ext in image_extensions:
            total_images += len(list(dataset_path.glob(f"**/*{ext}")))
            total_images += len(list(dataset_path.glob(f"**/*{ext.upper()}")))

        return DatasetInfo(
            name=dataset_path.name,
            path=str(dataset_path),
            total_images=total_images,
            has_ground_truth=ground_truth_file is not None and ground_truth_file.exists(),
            ground_truth_file=str(ground_truth_file) if ground_truth_file else None
        )

    async def _load_ground_truth(self, ground_truth_file: Path) -> Dict[str, Dict[str, Any]]:
        """Load ground truth annotations"""
        # This is a placeholder implementation
        # In a real system, you'd parse the actual ground truth format

        ground_truth = {}

        try:
            # Simple JSON format example
            if ground_truth_file.suffix.lower() == '.json':
                import json
                with open(ground_truth_file, 'r') as f:
                    data = json.load(f)

                # Expected format: {"image_name.jpg": {"is_forged": True, "forgery_type": "splicing"}}
                for image_name, annotation in data.items():
                    ground_truth[image_name] = annotation

            # Add other format parsers as needed

        except Exception as e:
            self.logger.error(f"Failed to load ground truth from {ground_truth_file}: {e}")

        return ground_truth

    async def _generate_comparison_report(
        self,
        results: List[BenchmarkResult],
        dataset_info: DatasetInfo,
        has_ground_truth: bool
    ) -> ComparisonReport:
        """Generate comprehensive comparison report"""

        if not results:
            return ComparisonReport(
                dataset_info=dataset_info,
                method_ranking={},
                comparison_summary={},
                detailed_results=[],
                overall_metrics=BenchmarkMetrics(
                    accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0, auc=0.0,
                    processing_time=0.0, method_metrics={}
                )
            )

        # Aggregate method scores across all results
        method_scores = {}
        method_processing_times = {}

        for result in results:
            for method, score in result.method_scores.items():
                if method not in method_scores:
                    method_scores[method] = []
                    method_processing_times[method] = []
                method_scores[method].append(score)
                method_processing_times[method].append(result.processing_time)

        # Calculate method statistics
        method_stats = {}
        for method, scores in method_scores.items():
            method_stats[method] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'mean_processing_time': np.mean(method_processing_times[method]),
                'total_processed': len(scores)
            }

        # Rank methods by mean score
        method_ranking = {
            method: stats['mean_score']
            for method, stats in sorted(
                method_stats.items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
        }

        # Calculate overall metrics
        if has_ground_truth:
            # Aggregate metrics across all results
            all_accuracies = [r.metrics.accuracy for r in results if r.metrics.accuracy > 0]
            all_precisions = [r.metrics.precision for r in results if r.metrics.precision > 0]
            all_recalls = [r.metrics.recall for r in results if r.metrics.recall > 0]
            all_f1s = [r.metrics.f1_score for r in results if r.metrics.f1_score > 0]
            all_aucs = [r.metrics.auc for r in results if r.metrics.auc > 0]

            overall_metrics = BenchmarkMetrics(
                accuracy=float(np.mean(all_accuracies)) if all_accuracies else 0.0,
                precision=float(np.mean(all_precisions)) if all_precisions else 0.0,
                recall=float(np.mean(all_recalls)) if all_recalls else 0.0,
                f1_score=float(np.mean(all_f1s)) if all_f1s else 0.0,
                auc=float(np.mean(all_aucs)) if all_aucs else 0.0,
                processing_time=float(np.mean([r.processing_time for r in results])),
                method_metrics=method_stats
            )
        else:
            # Use score statistics as proxy metrics
            overall_metrics = BenchmarkMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc=float(np.mean([r.metrics.auc for r in results])),
                processing_time=float(np.mean([r.processing_time for r in results])),
                method_metrics=method_stats
            )

        # Create comparison summary
        comparison_summary = {
            'total_images_processed': len(results),
            'methods_evaluated': list(method_scores.keys()),
            'best_performing_method': max(method_ranking, key=method_ranking.get),
            'method_consistency': {
                method: 1.0 - stats['std_score'] if stats['std_score'] <= 1.0 else 0.0
                for method, stats in method_stats.items()
            },
            'average_processing_time_per_image': float(np.mean([r.processing_time for r in results]))
        }

        return ComparisonReport(
            dataset_info=dataset_info,
            method_ranking=method_ranking,
            comparison_summary=comparison_summary,
            detailed_results=results,
            overall_metrics=overall_metrics
        )

    async def export_report_to_dict(self, report: ComparisonReport) -> Dict[str, Any]:
        """Export comparison report to dictionary format"""

        return {
            'dataset_info': {
                'name': report.dataset_info.name,
                'path': report.dataset_info.path,
                'total_images': report.dataset_info.total_images,
                'has_ground_truth': report.dataset_info.has_ground_truth
            },
            'method_ranking': report.method_ranking,
            'comparison_summary': report.comparison_summary,
            'overall_metrics': {
                'accuracy': report.overall_metrics.accuracy,
                'precision': report.overall_metrics.precision,
                'recall': report.overall_metrics.recall,
                'f1_score': report.overall_metrics.f1_score,
                'auc': report.overall_metrics.auc,
                'processing_time': report.overall_metrics.processing_time
            },
            'detailed_results_count': len(report.detailed_results)
        }