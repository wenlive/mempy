# mempy Benchmarks

Evaluation framework for testing memory systems on standard datasets.

## LOCOMO Benchmark

[LOCOMO](https://arxiv.org/abs/2402.17753) (Long Conversational Memory) is a benchmark from SNAP Research for evaluating very long-term conversational memory in dialogue systems.

### Dataset

- **Source**: ACL 2024, SNAP Research
- **Size**: 10 very long conversations (~300 turns each)
- **Tasks**: Question Answering + Event Summarization
- **Website**: https://snap-research.github.io/locomo/

### Quick Start

```bash
# Download the dataset
wget https://github.com/snap-research/locomo/raw/main/data/locomo10.json \
    -O tests/benchmarks/data/locomo10.json

# Run evaluation with mock embedder (for testing)
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json

# Run with OpenAI embeddings
pip install openai
python tests/benchmarks/locomo/run_evaluation.py \
    --data-path tests/benchmarks/data/locomo10.json \
    --embedder openai \
    --output results.json
```

### Using Your Own Embedder

```python
import asyncio
from mempy.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator
from mempy import Embedder

class MyEmbedder(Embedder):
    def __init__(self):
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Your embedding logic here
        return await my_llm.embed(text)

async def main():
    # Load dataset
    dataset = LOCOMODataset("tests/benchmarks/data/locomo10.json")

    # Create evaluator
    evaluator = LOCOMOEvaluator(
        dataset=dataset,
        embedder=MyEmbedder(),
        verbose=True,
    )

    # Run evaluation
    results = await evaluator.evaluate()
    evaluator.print_results(results)

asyncio.run(main())
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `recall_at_k` | Whether a relevant item appears in top-k results |
| `precision_at_k` | Proportion of relevant items in top-k results |
| `qa_accuracy` | Question answering accuracy (lenient matching) |
| `qa_f1` | Token-level F1 score for QA |

### Comparison with mem0

Our goal is to provide fair, reproducible benchmarking similar to [mem0](https://mem0.ai):

| System | QA Accuracy | vs OpenAI Memory | Token Savings |
|--------|-------------|------------------|---------------|
| mem0 | 66.9% | +26% | 90% |
| mempy | TBD | TBD | TBD |

*Results coming soon*

### Citation

If you use LOCOMO in your research, please cite:

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
