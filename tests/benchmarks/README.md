# mempy Benchmarks

Evaluation framework for testing memory systems on standard datasets.

## LOCOMO Benchmark

[LOCOMO](https://arxiv.org/abs/2402.17753) (Long Conversational Memory) is a benchmark from SNAP Research for evaluating very long-term conversational memory in dialogue systems.

### Dataset

- **Source**: ACL 2024, SNAP Research
- **Size**: 10 very long conversations (~300 turns each, ~2.7MB)
- **Tasks**: Question Answering + Event Summarization
- **Website**: https://snap-research.github.io/locomo/
- **Status**: Included in repo at `tests/benchmarks/data/locomo10.json`

---

## Quick Start

### 1. Using Mock Embedder (Testing)

Test the framework without a real model:

```bash
python tests/benchmarks/adapters/examples/run_with_mock.py
```

### 2. Using Qwen Models

For local Qwen models (Qwen3-235B-A22B, Qwen3-32B, etc.):

```bash
# Start vLLM server with Qwen model
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Run evaluation
python tests/benchmarks/adapters/examples/run_with_qwen.py \
    --base-url http://localhost:8000 \
    --model-name Qwen/Qwen2.5-7B-Instruct
```

### 3. Using OpenAI Embeddings

```bash
pip install openai
export OPENAI_API_KEY=your-key

python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --embedder openai \
    --output results.json
```

---

## Model Adapters

The benchmark framework includes pre-built adapters for common models:

| Adapter | Models | Usage |
|---------|--------|-------|
| `MockEmbedder` | None (testing) | Free, deterministic vectors |
| `QwenEmbedder` | Qwen3-235B, Qwen3-32B, Qwen2.5-7B | Local, via vLLM |
| `OpenAIEmbedder` | text-embedding-3-small/large | Paid API |

### Using Adapters

```python
from tests.benchmarks.adapters import QwenEmbedder, MockEmbedder, OpenAIEmbedder

# Qwen (local model)
embedder = QwenEmbedder(
    base_url="http://localhost:8000",
    model_name="Qwen/Qwen2.5-7B-Instruct"
)

# Mock (testing only)
embedder = MockEmbedder(dimension=768)

# OpenAI (API key required)
embedder = OpenAIEmbedder(api_key="your-key")
```

---

## Embedder Interface

All adapters implement the `mempy.Embedder` interface:

```python
from mempy import Embedder
from typing import List

class MyEmbedder(Embedder):
    def __init__(self):
        self._dimension = 768  # Must declare dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Your embedding logic here
        return await my_llm.embed(text)
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `recall_at_k` | Whether a relevant item appears in top-k results |
| `precision_at_k` | Proportion of relevant items in top-k results |
| `qa_accuracy` | Question answering accuracy (lenient matching) |
| `qa_f1` | Token-level F1 score for QA |

---

## Comparison with mem0

Our goal is to provide fair, reproducible benchmarking similar to [mem0](https://mem0.ai):

| System | QA Accuracy | vs OpenAI Memory | Token Savings |
|--------|-------------|------------------|---------------|
| mem0 | 66.9% | +26% | 90% |
| mempy | TBD | TBD | TBD |

*Results coming soon*

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
