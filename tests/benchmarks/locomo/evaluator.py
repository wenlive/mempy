"""LOCOMO evaluator for testing memory systems.

This module provides the LOCOMOEvaluator class for evaluating memory systems
on the LOCOMO benchmark dataset.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type

from tests.benchmarks.locomo.dataset import (
    Conversation,
    LOCOMODataset,
    QA,
    Session,
)
from tests.benchmarks.locomo.metrics import MetricsTracker, normalize_answer
from mempy.core.interfaces import Embedder
from mempy.memory import Memory as MemoryClient


class LOCOMOEvaluator:
    """Evaluator for memory systems on the LOCOMO benchmark.

    This evaluator can test different memory systems on their ability to:
    1. Store and retrieve conversational history
    2. Answer questions based on stored memories
    3. Handle long-term multi-session conversations

    Usage:
        >>> dataset = LOCOMODataset("path/to/locomo10.json")
        >>> embedder = MyEmbedder()
        >>> evaluator = LOCOMOEvaluator(dataset, embedder)
        >>> results = await evaluator.evaluate()
        >>> print(results)
    """

    def __init__(
        self,
        dataset: LOCOMODataset,
        embedder: Embedder,
        qa_function: Optional[Callable[[str, List[str]], str]] = None,
        verbose: bool = False,
    ):
        """Initialize the LOCOMO evaluator.

        Args:
            dataset: Loaded LOCOMO dataset
            embedder: Embedder instance for generating embeddings
            qa_function: Optional async function for answering questions.
                If None, uses a simple baseline that returns retrieved context.
                Signature: async (question: str, context: List[str]) -> str
            verbose: Whether to print progress information
        """
        self.dataset = dataset
        self.embedder = embedder
        self.qa_function = qa_function
        self.verbose = verbose

        self.metrics = MetricsTracker()

    async def evaluate_retrieval(
        self,
        memory_client: MemoryClient,
        limit: int = 10,
        max_qa_pairs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate retrieval performance.

        This tests the memory system's ability to retrieve relevant context
        for each question.

        Args:
            memory_client: The memory client to evaluate
            limit: Maximum number of memories to retrieve per query
            max_qa_pairs: Maximum number of QA pairs to evaluate (for quick testing)

        Returns:
            Dictionary with retrieval metrics
        """
        retrieval_metrics = MetricsTracker()
        qa_count = 0

        for conv in self.dataset.get_conversations():
            # Index the conversation into memory
            await self._index_conversation(memory_client, conv)

            # Evaluate retrieval for each QA pair
            for qa in conv.qa_pairs:
                # Check if we've reached the QA pair limit
                if max_qa_pairs and qa_count >= max_qa_pairs:
                    break

                self.metrics.start_timer("retrieval")

                if self.verbose:
                    print(f"[DEBUG] Query: {qa.question}")

                results = await memory_client.search(
                    qa.question,
                    user_id=conv.sample_id,
                    limit=limit,
                )
                self.metrics.end_timer("retrieval")

                if self.verbose:
                    print(f"[DEBUG] Retrieved {len(results)} memories")
                    for i, r in enumerate(results[:3]):
                        content_preview = r.content[:50] + "..." if len(r.content) > 50 else r.content
                        print(f"[DEBUG]   [{i}] {r.memory_id}: {content_preview}")

                # Extract retrieved IDs (use memory_id or content-based ID)
                retrieved_ids = [m.memory_id for m in results]

                # For ground truth, we use the dialogue IDs from evidence
                ground_truth_ids = qa.evidence if qa.evidence else []

                retrieval_metrics.add_retrieval_prediction(retrieved_ids, ground_truth_ids)

                if self.verbose:
                    recall = retrieval_metrics.get_recall_at_k(k=limit)
                    print(f"Retrieval: {conv.sample_id} | R@{limit}={recall:.3f}")

                qa_count += 1

            # Reset memory for next conversation
            await memory_client.reset()

            # Check if we've reached the QA pair limit
            if max_qa_pairs and qa_count >= max_qa_pairs:
                break

        return retrieval_metrics.compute_metrics()

    async def evaluate_qa(
        self,
        memory_client: MemoryClient,
        limit: int = 10,
        max_qa_pairs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate question-answering performance.

        This tests the full pipeline: store memories, retrieve relevant context,
        and answer questions.

        Args:
            memory_client: The memory client to evaluate
            limit: Maximum number of memories to retrieve per query
            max_qa_pairs: Maximum number of QA pairs to evaluate (for quick testing)

        Returns:
            Dictionary with QA metrics
        """
        qa_metrics = MetricsTracker()
        qa_count = 0

        for conv in self.dataset.get_conversations():
            # Index the conversation into memory
            await self._index_conversation(memory_client, conv)

            # Evaluate QA for each question
            for qa in conv.qa_pairs:
                # Check if we've reached the QA pair limit
                if max_qa_pairs and qa_count >= max_qa_pairs:
                    break

                self.metrics.start_timer("qa")

                # Retrieve relevant context
                if self.verbose:
                    print(f"[DEBUG] QA Query: {qa.question}")

                results = await memory_client.search(
                    qa.question,
                    user_id=conv.sample_id,
                    limit=limit,
                )

                # Build context from retrieved memories
                context = [m.content for m in results]

                if self.verbose:
                    print(f"[DEBUG] Retrieved {len(results)} memories for QA")
                    for i, r in enumerate(results[:3]):
                        content_preview = r.content[:50] + "..." if len(r.content) > 50 else r.content
                        print(f"[DEBUG]   [{i}] {r.memory_id}: {content_preview}")

                # Generate answer
                if self.qa_function:
                    predicted = await self.qa_function(qa.question, context)
                else:
                    # Default: concatenate retrieved context
                    predicted = " ".join(context[:3]) if context else "I don't have enough information."

                self.metrics.end_timer("qa")

                # Compare with ground truth
                qa_metrics.add_qa_prediction(predicted, qa.answer)

                if self.verbose:
                    acc = qa_metrics.get_qa_accuracy()
                    print(f"QA: {conv.sample_id} | Acc={acc:.3f} | Q: {qa.question[:50]}...")

                qa_count += 1

            # Reset memory for next conversation
            await memory_client.reset()

            # Check if we've reached the QA pair limit
            if max_qa_pairs and qa_count >= max_qa_pairs:
                break

        return qa_metrics.compute_metrics()

    async def evaluate(
        self,
        limit: int = 10,
        max_qa_pairs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation (retrieval + QA).

        Args:
            limit: Maximum number of memories to retrieve per query
            max_qa_pairs: Maximum number of QA pairs to evaluate (for quick testing)

        Returns:
            Dictionary with all evaluation metrics
        """
        # Create a new memory client for this evaluation
        memory_client = MemoryClient(
            embedder=self.embedder,
            verbose=self.verbose,
        )

        # Run retrieval evaluation
        if self.verbose:
            print("\n=== Evaluating Retrieval ===")
        retrieval_results = await self.evaluate_retrieval(memory_client, limit, max_qa_pairs)

        # Run QA evaluation
        if self.verbose:
            print("\n=== Evaluating Question Answering ===")
        qa_results = await self.evaluate_qa(memory_client, limit, max_qa_pairs)

        # Calculate actual number of QA pairs evaluated
        actual_qa_pairs = min(max_qa_pairs, self.dataset.num_qa_pairs) if max_qa_pairs else self.dataset.num_qa_pairs

        # Combine results
        combined_results = {
            "dataset": "LOCOMO",
            "num_conversations": self.dataset.num_conversations,
            "num_qa_pairs": self.dataset.num_qa_pairs,
            "num_qa_pairs_evaluated": actual_qa_pairs,
            "retrieval": retrieval_results,
            "qa": qa_results,
        }

        return combined_results

    async def _index_conversation(
        self,
        memory_client: MemoryClient,
        conversation: Conversation,
    ) -> None:
        """Index a conversation into memory.

        Args:
            memory_client: The memory client to use
            conversation: The conversation to index
        """
        if self.verbose:
            print(f"[DEBUG] Indexing conversation {conversation.sample_id}...")

        memory_count = 0
        for session_idx, session in enumerate(conversation.sessions):
            for turn in session.turns:
                # Create a memory entry for each turn
                content = f"{turn.speaker}: {turn.text}"
                memory_id = await memory_client.add(
                    content,
                    user_id=conversation.sample_id,
                    metadata={
                        "speaker": turn.speaker,
                        "session_id": session.session_id,
                        "dia_id": turn.dia_id,
                        "sample_id": conversation.sample_id,
                    },
                )
                memory_count += 1

                if self.verbose:
                    content_preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"[DEBUG] Stored memory {memory_id}: {content_preview}")

        if self.verbose:
            print(f"[DEBUG] Total memories stored for {conversation.sample_id}: {memory_count}")

    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty-print evaluation results.

        Args:
            results: Results dictionary from evaluate()
        """
        print("\n" + "=" * 60)
        print("LOCOMO Benchmark Results")
        print("=" * 60)
        print(f"Dataset: {results['dataset']}")
        print(f"Conversations: {results['num_conversations']}")
        print(f"QA Pairs: {results['num_qa_pairs']}")
        print("\n--- Retrieval Metrics ---")

        for key, value in results.get("retrieval", {}).items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif not key.startswith("num_"):
                print(f"  {key}: {value}")

        print("\n--- QA Metrics ---")
        for key, value in results.get("qa", {}).items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif not key.startswith("num_"):
                print(f"  {key}: {value}")

        print("=" * 60)

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: Results dictionary from evaluate()
            output_path: Path to save the results
        """
        import json
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


async def evaluate_locomo(
    data_path: str,
    embedder: Embedder,
    qa_function: Optional[Callable] = None,
    verbose: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to run LOCOMO evaluation.

    Args:
        data_path: Path to LOCOMO JSON file
        embedder: Embedder instance
        qa_function: Optional QA function
        verbose: Whether to print progress
        output_path: Optional path to save results

    Returns:
        Evaluation results dictionary

    Example:
        >>> from mempy import Embedder
        >>> class MyEmbedder(Embedder):
        ...     # implementation
        ...     pass
        >>> results = await evaluate_locomo(
        ...     "tests/benchmarks/data/locomo10.json",
        ...     MyEmbedder()
        ... )
    """
    # Load dataset
    dataset = LOCOMODataset(data_path)

    # Create evaluator
    evaluator = LOCOMOEvaluator(dataset, embedder, qa_function, verbose)

    # Run evaluation
    results = await evaluator.evaluate()

    # Print results
    if verbose:
        evaluator.print_results(results)

    # Save results if path provided
    if output_path:
        evaluator.save_results(results, output_path)

    return results
