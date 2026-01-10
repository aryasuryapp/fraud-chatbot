# RAGAS Evaluation System

This directory contains the RAGAS-based evaluation system for the fraud detection chatbot.

## Overview

The evaluation system uses the [RAGAS library](https://github.com/explodinggradients/ragas) to assess the quality of the RAG (Retrieval-Augmented Generation) system with the following metrics:

### Core Metrics (Reference-Free)

- **context_precision**: Measures whether retrieved contexts are relevant to the question
- **answer_relevancy**: Evaluates if the answer addresses the question appropriately
- **faithfulness**: Assesses whether the answer is grounded in the retrieved context

### Ground Truth Metrics (Optional)

- **context_recall**: Checks if all necessary contexts were retrieved (requires ground truth contexts)
- **answer_correctness**: Measures semantic similarity to ground truth answer

## Files

- **`scorer.py`**: RAGAS-based evaluation scorer (replaces heuristic evaluation)
- **`test_dataset.py`**: Test dataset with 10 fraud detection questions and expected contexts
- **`run_evaluation.py`**: Main evaluation script with CLI interface
- **`results/`**: Output directory for evaluation results (JSON and CSV)

## Installation

Install RAGAS and dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `ragas>=0.1.0`
- `datasets>=2.14.0`
- `langchain-openai>=0.0.5`

## Usage

### Quick Start

Run evaluation with default settings (reference-free metrics only):

```bash
python evaluation/run_evaluation.py
```

### With Ground Truth Metrics

Include context_recall and answer_correctness:

```bash
python evaluation/run_evaluation.py --with-ground-truth
```

### Custom Options

```bash
# Use GPT-4 for evaluation
python evaluation/run_evaluation.py --model gpt-4

# Evaluate only first 3 test cases
python evaluation/run_evaluation.py --num-cases 3

# Custom output directory
python evaluation/run_evaluation.py --output-dir my_results
```

### Full Command

```bash
python evaluation/run_evaluation.py \
  --with-ground-truth \
  --model gpt-3.5-turbo \
  --output-dir evaluation/results \
  --num-cases 10
```

## Output

The evaluation produces two files:

1. **JSON file** (`ragas_evaluation_YYYYMMDD_HHMMSS.json`):
   - Individual results for each test case
   - Average metrics across all cases
   - Metadata (timestamp, model, settings)

2. **CSV file** (`ragas_summary_YYYYMMDD_HHMMSS.csv`):
   - Question and metric scores in tabular format
   - Easy to import into Excel/Pandas for analysis

## Test Dataset

The test dataset contains 10 fraud detection questions covering:

- Transaction statistics and fraud rates
- Fraud pattern identification
- Machine learning for fraud detection
- Feature engineering
- Geographic and temporal patterns
- Best practices and prevention
- False positives

Each test case includes:
- `question`: The user query
- `ground_truth`: Expected answer (for answer_correctness)
- `contexts`: List of relevant context passages (for context_recall)

## Customization

### Adding Test Cases

Edit `test_dataset.py` and add new entries to `TEST_DATASET`:

```python
{
    "question": "Your question here?",
    "ground_truth": "Expected answer...",
    "contexts": [
        "Relevant context 1...",
        "Relevant context 2..."
    ]
}
```

### Changing Evaluation Metrics

Edit `scorer.py` and modify the `metrics` list in `QAScorer.__init__()`:

```python
from ragas.metrics import context_precision, answer_relevancy, faithfulness

self.metrics = [
    context_precision,
    answer_relevancy,
    faithfulness
    # Add custom metrics here
]
```

### Using in Code

```python
from evaluation.scorer import QAScorer
from llm.qa_chain import QAChain

# Initialize
qa_chain = QAChain()
scorer = QAScorer(use_ground_truth_metrics=False)

# Generate answer
result = qa_chain.ask_for_evaluation("What are fraud patterns?")

# Evaluate
metrics = scorer.evaluate_single(
    question=result["question"],
    answer=result["answer"],
    contexts=result["contexts"]
)

print(f"Faithfulness: {metrics['faithfulness']:.3f}")
print(f"Answer Relevancy: {metrics['answer_relevancy']:.3f}")
```

## Cost Considerations

RAGAS uses LLM calls for faithfulness evaluation:
- **gpt-3.5-turbo**: ~$0.001 per test case (recommended)
- **gpt-4**: ~$0.03 per test case (higher quality)

For 10 test cases:
- gpt-3.5-turbo: ~$0.01 total
- gpt-4: ~$0.30 total

## Interpreting Results

### Good Scores

- **context_precision > 0.8**: Retrieval is working well
- **answer_relevancy > 0.7**: Answers are on-topic
- **faithfulness > 0.8**: Answers are grounded in context

### Warning Signs

- **Low context_precision**: Retriever is returning irrelevant documents
- **Low answer_relevancy**: LLM is not addressing the question
- **Low faithfulness**: LLM is hallucinating or adding unsupported information

## Troubleshooting

### "OPENAI_API_KEY not set"

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### "No document embeddings found"

Build the vector store first:
```bash
python ingestion/load_docs.py
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Further Reading

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS Metrics Explained](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
- [RAG Evaluation Best Practices](https://www.anthropic.com/research/evaluating-rag)
