# Benchmark Evaluation Guide

Guide for running and interpreting mempy benchmark evaluations.

## Overview

mempy includes evaluation frameworks for testing memory systems on standard datasets:
- **LOCOMO**: Long-term conversational memory benchmark (ACL 2024)

## LOCOMO Benchmark

### What is LOCOMO?

LOCOMO (Long Conversational Memory) is a benchmark from SNAP Research that evaluates very long-term conversational memory in dialogue systems.

- **Paper**: [arxiv.org/abs/2402.17753](https://arxiv.org/abs/2402.17753)
- **GitHub**: [snap-research/locomo](https://github.com/snap-research/locomo)
- **Website**: [snap-research.github.io/locomo/](https://snap-research.github.io/locomo/)

### Dataset

- 10 very long conversations (~300 turns each)
- Question-answering evaluation
- Event summarization tasks
- Included in repo at `tests/benchmarks/data/locomo10.json`

---

## Running Evaluation

### Prerequisites

The dataset is included in the repository. No additional download needed.

### Quick Test (Mock Embedder)

```bash
python tests/benchmarks/adapters/examples/run_with_mock.py
```

This uses a mock embedder for testing the framework without a real model.

### With Qwen Models

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Run evaluation
python tests/benchmarks/adapters/examples/run_with_qwen.py \
    --base-url http://localhost:8000
```

### With OpenAI

```bash
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --embedder openai
```

---

## Understanding Results

### Output Format

```
============================================================
LOCOMO Benchmark Results
============================================================
Dataset: LOCOMO
Conversations: 10
QA Pairs: 125

--- Retrieval Metrics ---
  recall_at_1: 0.4500
  recall_at_5: 0.7200
  precision_at_5: 0.1440

--- QA Metrics ---
  qa_accuracy: 0.5200
  qa_accuracy_strict: 0.2800
  qa_f1: 0.4100
============================================================
```

### Metrics Explained

| Metric | Description | Good Range |
|--------|-------------|------------|
| `recall_at_k` | % of queries where relevant item is in top-k results | Higher is better |
| `precision_at_k` | % of retrieved items that are relevant | Higher is better |
| `qa_accuracy` | QA accuracy with lenient matching | Higher is better |
| `qa_f1` | Token-level F1 score | Higher is better |

---

## Comparison with mem0

Our goal is to provide fair, reproducible benchmarking.

| System | QA Accuracy | vs OpenAI Memory | Token Savings |
|--------|-------------|------------------|---------------|
| mem0 | 66.9% | +26% | 90% |
| mempy | TBD | TBD | TBD |

*Results will be added as we run evaluations*

---

## Running Your Own Evaluation

### Using Custom Embedder

```python
import asyncio
from mempy import Memory, Embedder
from typing import List
from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator

class MyEmbedder(Embedder):
    def __init__(self):
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Your embedding logic
        return await my_llm.embed(text)

async def main():
    # Load dataset
    dataset = LOCOMODataset("tests/benchmarks/data/locomo10.json")

    # Create evaluator
    evaluator = LOCOMOEvaluator(
        dataset=dataset,
        embedder=MyEmbedder(),
        verbose=True
    )

    # Run evaluation
    results = await evaluator.evaluate(limit=10)

    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results, "my_results.json")

asyncio.run(main())
```

---

## Advanced: Retrieval Only

To evaluate just the retrieval component (without QA):

```python
from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator
from mempy import Memory

async def main():
    dataset = LOCOMODataset("tests/benchmarks/data/locomo10.json")
    evaluator = LOCOMOEvaluator(dataset, embedder=YourEmbedder())

    # Create memory client
    memory = Memory(embedder=YourEmbedder())

    # Evaluate retrieval only
    retrieval_results = await evaluator.evaluate_retrieval(
        memory_client=memory,
        limit=10
    )

    print(f"Recall@1: {retrieval_results['recall_at_1']:.3f}")
    print(f"Recall@5: {retrieval_results['recall_at_5']:.3f}")

asyncio.run(main())
```

---

## Tips for Better Results

1. **Use a Quality Embedding Model**: The quality of semantic search heavily depends on your embeddings.

2. **Tune Retrieval Parameters**: Experiment with `limit` to find the optimal number of retrieved memories.

3. **Consider Context Length**: For very long conversations, ensure your embedder handles long inputs well.

4. **Monitor Resource Usage**: Large evaluations can be memory-intensive.

---

## Citation

If you use LOCOMO in your research, please cite:

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
