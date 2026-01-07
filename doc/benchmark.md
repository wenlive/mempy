# Benchmark Evaluation Guide

Guide for running and interpreting mempy benchmark evaluations using the LOCOMO dataset.

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

## Quick Start (E2E Testing)

This section provides three ways to run LOCOMO benchmark tests, from fastest validation to full evaluation.

### Option 1: Quick Validation (Mock Embedder, Zero Cost)

**Purpose:** Validate the evaluation framework works correctly without API costs.

**Script:** `test_e2e_quick.py`

```bash
# Activate environment
conda activate mempy

# Run quick test (5 QA pairs, < 1 minute)
python tests/benchmarks/locomo/test_e2e_quick.py
```

**Expected Results:**
- Execution time: < 1 minute
- Token cost: 0 (no API calls)
- QA accuracy: Low (~0-10%, expected for mock embeddings)

### Option 2: Small-Scale Real Test (10 QA Pairs)

**Purpose:** Test with real embeddings at low cost before running full evaluation.

**Script:** `run_evaluation.py`

```bash
# With OpenAI embeddings
export OPENAI_API_KEY=your-key
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --embedder openai \
    --max-qa-pairs 10

# With mock embedder (free)
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --max-qa-pairs 10
```

**Estimated cost:** ~$0.01-0.05 for OpenAI embeddings

### Option 3: Full Evaluation

**Purpose:** Complete evaluation on all 125 QA pairs.

**Warning:** This evaluates all QA pairs and may take significant time and API costs (~$0.10-0.50 for OpenAI).

```bash
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --embedder openai
```

---

## Running Evaluation

### Prerequisites

The dataset is included in the repository. No additional download needed.

### Environment Setup

**1. Activate Conda Environment**

```bash
# Activate mempy environment
conda activate mempy

# Verify installation
python -c "import mempy; print('mempy installed successfully')"
```

**2. Install Dependencies**

**Required (already installed):**
```bash
pip install chromadb networkx aiohttp
```

**Optional (for OpenAI embeddings):**
```bash
pip install openai
export OPENAI_API_KEY=your-api-key
```

**Optional (for local embeddings):**
```bash
pip install sentence-transformers
```

### Command-Line Options

**Script:** `run_evaluation.py`

| Option | Description | Default |
|--------|-------------|---------|
| `--data-path` | Path to LOCOMO dataset JSON file | Required |
| `--max-qa-pairs N` | Limit to N QA pairs (for quick testing) | All (125) |
| `--limit N` | Retrieve top N memories per query | 10 |
| `--embedder {mock,openai,zhipu}` | Embedder to use | mock |
| `--output FILE` | Save results to JSON file | None |
| `--quiet` | Suppress progress output | False |

**Examples:**

```bash
# Test with 10 QA pairs using mock embedder
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --max-qa-pairs 10

# Test with OpenAI embeddings and save results
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --embedder openai \
    --max-qa-pairs 10 \
    --output my_results.json

# Test with Zhipu embeddings (智谱AI)
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --embedder zhipu \
    --max-qa-pairs 10
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

#### Retrieval Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| `recall_at_k` | % of queries where relevant item is in top-k results | Higher is better |
| `precision_at_k` | % of retrieved items that are relevant | Higher is better |

#### QA Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| `qa_accuracy` | QA accuracy with lenient matching | Higher is better |
| `qa_f1` | Token-level F1 score | Higher is better |

### Expected Performance

| Embedder | QA Accuracy | Recall@5 | Notes |
|----------|-------------|----------|-------|
| Mock | ~0-10% | ~5-10% | Validates framework, not model quality |
| OpenAI (text-embedding-3-small) | ~50-70% | ~70-80% | Good baseline, reasonable cost |
| Zhipu (embedding-3) | ~55-75% | ~75-85% | Strong multilingual performance |
| BGE-M3 (local) | ~60-75% | ~75-85% | Best performance, requires setup |

**Note:** Mock embeddings will have very low accuracy. This is expected and normal! The purpose is to validate the evaluation framework works correctly.

---

## Cost Estimation

### OpenAI Embeddings (text-embedding-3-small)

| Test Size | QA Pairs | Estimated Tokens | Estimated Cost |
|-----------|----------|------------------|----------------|
| Quick test | 5 | ~1,000 | <$0.01 |
| Small test | 10 | ~2,000 | ~$0.01 |
| Medium test | 50 | ~10,000 | ~$0.05 |
| Full test | 125 | ~25,000 | ~$0.10 |

**Note:** Costs are estimates only. Actual costs depend on text length and retrieval limits.

### Free Alternatives

1. **Mock Embedder:** Zero cost, validates framework
2. **Local Models:** One-time setup cost, then free
   - BAAI/bge-small-en-v1.5 (384 dims, fast)
   - BAAI/bge-base-en-v1.5 (768 dims, better)
   - BAAI/bge-m3 (1024 dims, multilingual, best)

---

## Troubleshooting

### Issue: Dataset not found

**Error:** `Error: Data file not found: tests/benchmarks/data/locomo10.json`

**Solution:**
- Make sure you're running the command from the project root directory
- Check that the dataset file exists: `ls tests/benchmarks/data/locomo10.json`

### Issue: Import errors

**Error:** `ModuleNotFoundError: No module named 'mempy'`

**Solution:**
- Activate the conda environment: `conda activate mempy`
- Verify mempy is installed: `pip list | grep mempy`

### Issue: Low accuracy with mock embedder

**This is expected!** Mock embeddings use hash-based vectors that have no semantic understanding. The test validates the framework works, not the model quality.

### Issue: API key errors (OpenAI)

**Error:** `openai.AuthenticationError: No API key provided`

**Solution:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

Or pass it as a command-line argument:
```bash
python tests/benchmarks/locomo/run_evaluation.py \
    --embedder openai \
    --openai-key your-api-key-here
```

---

## Advanced Usage

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

### Custom QA Function

To test with a custom QA function (e.g., using DeepSeek for generation):

```python
import asyncio
from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator
from mempy import Embedder

# Your embedder
embedder = MyEmbedder()

# Your QA function
async def my_qa_function(question: str, context: list) -> str:
    # Use your LLM here (e.g., DeepSeek, Zhipu)
    return await my_llm.answer(question, context)

# Create evaluator with custom QA function
dataset = LOCOMODataset("tests/benchmarks/data/locomo10.json")
dataset.load()

evaluator = LOCOMOEvaluator(
    dataset=dataset,
    embedder=embedder,
    qa_function=my_qa_function,
    verbose=True
)

# Run evaluation
results = await evaluator.evaluate(limit=10, max_qa_pairs=5)
evaluator.print_results(results)
```

### Retrieval Only Evaluation

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

## Comparison with mem0

Our goal is to provide fair, reproducible benchmarking.

| System | QA Accuracy | vs OpenAI Memory | Token Savings |
|--------|-------------|------------------|---------------|
| mem0 | 66.9% | +26% | 90% |
| mempy | TBD | TBD | TBD |

*Results will be added as we run evaluations*

---

## Tips for Better Results

1. **Use a Quality Embedding Model**: The quality of semantic search heavily depends on your embeddings.

2. **Tune Retrieval Parameters**: Experiment with `limit` to find the optimal number of retrieved memories.

3. **Consider Context Length**: For very long conversations, ensure your embedder handles long inputs well.

4. **Monitor Resource Usage**: Large evaluations can be memory-intensive.

5. **Start Small**: Always run a small test first to validate your setup before running the full evaluation.

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

---

**Last Updated**: 2026-01-06
**Related Documentation**:
- [Architecture](architecture.md) - mempy architecture design
- [Adapter Guide](adapter-guide.md) - Creating custom embedders
- [Strategy System](strategy_system.md) - Pluggable strategy architecture
