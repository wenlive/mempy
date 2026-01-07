"""Evaluation metrics for LOCOMO benchmark.

This module provides metrics for evaluating memory systems on the LOCOMO dataset,
including retrieval metrics and QA accuracy.
"""

from typing import Any, Dict, List, Optional, Tuple
import time


def recall_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 5,
) -> float:
    """Calculate Recall@K metric.

    Recall@K measures whether at least one relevant item is in the top K retrieved items.

    Args:
        retrieved_ids: List of retrieved item IDs (ordered by relevance)
        ground_truth_ids: List of ground truth relevant item IDs
        k: Number of top items to consider

    Returns:
        Recall@K score (0.0 to 1.0)

    Example:
        >>> recall_at_k(["mem1", "mem2", "mem3"], ["mem2"], k=3)
        1.0
        >>> recall_at_k(["mem1", "mem2", "mem3"], ["mem4"], k=3)
        0.0
    """
    if not ground_truth_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    return 1.0 if any(id in ground_truth_ids for id in top_k) else 0.0


def mean_recall_at_k(
    all_retrieved: List[List[str]],
    all_ground_truth: List[List[str]],
    k: int = 5,
) -> float:
    """Calculate mean Recall@K across multiple queries.

    Args:
        all_retrieved: List of retrieved ID lists for each query
        all_ground_truth: List of ground truth ID lists for each query
        k: Number of top items to consider

    Returns:
        Mean Recall@K score (0.0 to 1.0)
    """
    if len(all_retrieved) != len(all_ground_truth):
        raise ValueError("Retrieved and ground truth lists must have same length")

    if not all_retrieved:
        return 0.0

    scores = [
        recall_at_k(retrieved, ground_truth, k)
        for retrieved, ground_truth in zip(all_retrieved, all_ground_truth)
    ]
    return sum(scores) / len(scores)


def precision_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 5,
) -> float:
    """Calculate Precision@K metric.

    Precision@K measures the proportion of relevant items in the top K retrieved items.

    Args:
        retrieved_ids: List of retrieved item IDs (ordered by relevance)
        ground_truth_ids: List of ground truth relevant item IDs
        k: Number of top items to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for id in top_k if id in ground_truth_ids)
    return relevant_count / k


def mean_precision_at_k(
    all_retrieved: List[List[str]],
    all_ground_truth: List[List[str]],
    k: int = 5,
) -> float:
    """Calculate mean Precision@K across multiple queries.

    Args:
        all_retrieved: List of retrieved ID lists for each query
        all_ground_truth: List of ground truth ID lists for each query
        k: Number of top items to consider

    Returns:
        Mean Precision@K score (0.0 to 1.0)
    """
    if len(all_retrieved) != len(all_ground_truth):
        raise ValueError("Retrieved and ground truth lists must have same length")

    if not all_retrieved:
        return 0.0

    scores = [
        precision_at_k(retrieved, ground_truth, k)
        for retrieved, ground_truth in zip(all_retrieved, all_ground_truth)
    ]
    return sum(scores) / len(scores)


def qa_accuracy(
    predicted_answers: List[str],
    ground_truth_answers: List[str],
    strict: bool = False,
) -> float:
    """Calculate QA accuracy.

    By default, uses a lenient matching that checks if the predicted answer
    contains key information from the ground truth.

    Args:
        predicted_answers: List of predicted answer strings
        ground_truth_answers: List of ground truth answer strings
        strict: If True, requires exact match; otherwise uses substring matching

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predicted_answers) != len(ground_truth_answers):
        raise ValueError("Predicted and ground truth lists must have same length")

    if not predicted_answers:
        return 0.0

    correct = 0
    for pred, truth in zip(predicted_answers, ground_truth_answers):
        # Convert to string if needed (handle numeric answers in dataset)
        pred_str = str(pred) if not isinstance(pred, str) else pred
        truth_str = str(truth) if not isinstance(truth, str) else truth

        pred_lower = pred_str.lower().strip()
        truth_lower = truth_str.lower().strip()

        if strict:
            if pred_lower == truth_lower:
                correct += 1
        else:
            # Lenient matching: check if key words from truth appear in prediction
            # Split into words and check for overlap
            truth_words = set(truth_lower.split())
            pred_words = set(pred_lower.split())

            # If prediction contains at least 30% of truth words, consider it correct
            overlap = truth_words & pred_words
            if len(overlap) >= max(1, len(truth_words) * 0.3):
                correct += 1

    return correct / len(predicted_answers)


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Args:
        answer: Raw answer string (or numeric value)

    Returns:
        Normalized answer string
    """
    # Convert to string if needed (handle numeric answers in dataset)
    answer = str(answer) if not isinstance(answer, str) else answer

    # Convert to lowercase
    answer = answer.lower()
    # Remove extra whitespace
    answer = " ".join(answer.split())
    # Remove common punctuation at the end
    for punct in [".", "!", "?", ",", ";", ":"]:
        answer = answer.rstrip(punct)
    return answer


def exact_match_score(
    predicted: str,
    ground_truth: str,
) -> bool:
    """Check if predicted answer exactly matches ground truth (after normalization).

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if they match after normalization
    """
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def f1_score(
    predicted: str,
    ground_truth: str,
) -> float:
    """Calculate F1 score between predicted and ground truth answers.

    F1 score is based on token-level precision and recall.

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = set(normalize_answer(predicted).split())
    truth_tokens = set(normalize_answer(ground_truth).split())

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


class MetricsTracker:
    """Track and compute evaluation metrics over multiple runs.

    Attributes:
        track_time: Whether to track timing information
    """

    def __init__(self, track_time: bool = True):
        """Initialize the metrics tracker.

        Args:
            track_time: Whether to track timing information
        """
        self.track_time = track_time
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._qa_predictions: List[Tuple[str, str]] = []
        self._retrieval_predictions: List[Tuple[List[str], List[str]]] = []
        self._start_times: Dict[str, float] = {}
        self._durations: Dict[str, List[float]] = {}

    def start_timer(self, name: str) -> None:
        """Start a named timer.

        Args:
            name: Timer name
        """
        if self.track_time:
            self._start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End a named timer and record the duration.

        Args:
            name: Timer name

        Returns:
            Duration in milliseconds
        """
        if self.track_time and name in self._start_times:
            duration = (time.time() - self._start_times[name]) * 1000  # Convert to ms
            if name not in self._durations:
                self._durations[name] = []
            self._durations[name].append(duration)
            del self._start_times[name]
            return duration
        return 0.0

    def add_qa_prediction(self, predicted: str, ground_truth: str) -> None:
        """Add a QA prediction.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
        """
        self._qa_predictions.append((predicted, ground_truth))

    def add_retrieval_prediction(self, retrieved: List[str], ground_truth: List[str]) -> None:
        """Add a retrieval prediction.

        Args:
            retrieved: List of retrieved IDs
            ground_truth: List of ground truth IDs
        """
        self._retrieval_predictions.append((retrieved, ground_truth))

    def get_qa_accuracy(self, strict: bool = False) -> float:
        """Get QA accuracy.

        Args:
            strict: Whether to use strict matching

        Returns:
            QA accuracy
        """
        if not self._qa_predictions:
            return 0.0

        predictions = [p[0] for p in self._qa_predictions]
        ground_truths = [p[1] for p in self._qa_predictions]
        return qa_accuracy(predictions, ground_truths, strict=strict)

    def get_qa_f1(self) -> float:
        """Get mean F1 score for QA predictions.

        Returns:
            Mean F1 score
        """
        if not self._qa_predictions:
            return 0.0

        f1_scores = [
            f1_score(p[0], p[1])
            for p in self._qa_predictions
        ]
        return sum(f1_scores) / len(f1_scores)

    def get_recall_at_k(self, k: int = 5) -> float:
        """Get mean Recall@K for retrieval predictions.

        Args:
            k: Number of top items to consider

        Returns:
            Mean Recall@K
        """
        if not self._retrieval_predictions:
            return 0.0

        all_retrieved = [p[0] for p in self._retrieval_predictions]
        all_ground_truth = [p[1] for p in self._retrieval_predictions]
        return mean_recall_at_k(all_retrieved, all_ground_truth, k)

    def get_precision_at_k(self, k: int = 5) -> float:
        """Get mean Precision@K for retrieval predictions.

        Args:
            k: Number of top items to consider

        Returns:
            Mean Precision@K
        """
        if not self._retrieval_predictions:
            return 0.0

        all_retrieved = [p[0] for p in self._retrieval_predictions]
        all_ground_truth = [p[1] for p in self._retrieval_predictions]
        return mean_precision_at_k(all_retrieved, all_ground_truth, k)

    def get_avg_duration(self, name: str) -> float:
        """Get average duration for a named timer.

        Args:
            name: Timer name

        Returns:
            Average duration in milliseconds
        """
        if name not in self._durations or not self._durations[name]:
            return 0.0
        return sum(self._durations[name]) / len(self._durations[name])

    def get_total_duration(self, name: str) -> float:
        """Get total duration for a named timer.

        Args:
            name: Timer name

        Returns:
            Total duration in milliseconds
        """
        if name not in self._durations or not self._durations[name]:
            return 0.0
        return sum(self._durations[name])

    def compute_metrics(self, ks: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """Compute all metrics as a dictionary.

        Args:
            ks: List of k values for recall/precision calculation

        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            "num_qa_predictions": len(self._qa_predictions),
            "num_retrieval_predictions": len(self._retrieval_predictions),
            "qa_accuracy": self.get_qa_accuracy(),
            "qa_accuracy_strict": self.get_qa_accuracy(strict=True),
            "qa_f1": self.get_qa_f1(),
        }

        for k in ks:
            metrics[f"recall_at_{k}"] = self.get_recall_at_k(k)
            metrics[f"precision_at_{k}"] = self.get_precision_at_k(k)

        # Add timing metrics
        for name, durations in self._durations.items():
            if durations:
                metrics[f"{name}_avg_ms"] = sum(durations) / len(durations)
                metrics[f"{name}_total_ms"] = sum(durations)

        return metrics

    def __repr__(self) -> str:
        """Return string representation."""
        metrics = self.compute_metrics()
        parts = [f"qa_acc={metrics['qa_accuracy']:.3f}"]
        if metrics['num_retrieval_predictions'] > 0:
            parts.append(f"r@5={metrics['recall_at_5']:.3f}")
        return f"MetricsTracker({', '.join(parts)})"
