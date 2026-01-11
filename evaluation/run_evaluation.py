"""
Run RAGAS evaluation on the fraud detection chatbot.

This script evaluates the QA system using the RAGAS library with metrics:
- context_precision: Are retrieved contexts relevant to the question?
- answer_relevancy: Does the answer address the question?
- faithfulness: Is the answer grounded in the retrieved context?

Optionally (with ground truth):
- context_recall: Did we retrieve all necessary contexts?
- answer_correctness: Semantic similarity to ground truth answer
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from llm.qa_chain import QAChain
from evaluation.scorer import QAScorer
from evaluation.test_dataset import get_test_dataset, get_dataset_size

# Load environment variables
load_dotenv()


def run_evaluation(
    use_ground_truth_metrics: bool = False,
    llm_model: str = "gpt-3.5-turbo",
    output_dir: str = "evaluation/results",
    num_test_cases: int = None,
    json_output_path: str = None,
    csv_output_path: str = None
):
    """
    Run RAGAS evaluation on test dataset.
    
    Args:
        use_ground_truth_metrics: Whether to include metrics requiring ground truth
        llm_model: OpenAI model to use for evaluation
        output_dir: Directory to save evaluation results
        num_test_cases: Number of test cases to evaluate (None = all)
        json_output_path: Custom path for JSON output (overrides default naming)
        csv_output_path: Custom path for CSV output (overrides default naming)
        
    Returns:
        dict: Aggregate results with average scores for each metric
    """
    print("\n" + "="*70)
    print("FRAUD DETECTION CHATBOT - RAGAS EVALUATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize QA chain
    print("\n1. Initializing QA Chain...")
    qa_chain = QAChain(
        llm_provider="openai",
        model_name=llm_model
    )
    print(f"   ✓ QA Chain initialized with {llm_model}")
    
    # Initialize scorer
    print("\n2. Initializing RAGAS Scorer...")
    scorer = QAScorer(
        llm_model=llm_model,
        use_ground_truth_metrics=use_ground_truth_metrics
    )
    print(f"   ✓ Scorer initialized")
    print(f"   ✓ Ground truth metrics: {'Enabled' if use_ground_truth_metrics else 'Disabled'}")
    
    # Load test dataset
    print("\n3. Loading Test Dataset...")
    test_dataset = get_test_dataset()
    total_cases = get_dataset_size()
    
    # Limit test cases if specified
    if num_test_cases:
        test_dataset = test_dataset[:num_test_cases]
        print(f"   ✓ Loaded {num_test_cases} test cases (out of {total_cases} total)")
    else:
        print(f"   ✓ Loaded {total_cases} test cases")
    
    # Generate answers for each test question
    print("\n4. Generating Answers...")
    test_cases_with_answers = []
    
    for i, test_case in enumerate(test_dataset, 1):
        print(f"   [{i}/{len(test_dataset)}] Processing: {test_case['question'][:60]}...")
        
        # Get answer from QA chain (in RAGAS format)
        result = qa_chain.ask_for_evaluation(test_case["question"])
        
        # Prepare test case for evaluation
        eval_case = {
            "question": result["question"],
            "answer": result["answer"],
            "contexts": result["contexts"]  # List[str] format for RAGAS
        }
        
        # Add ground truth if available and needed
        if use_ground_truth_metrics and "ground_truth" in test_case:
            eval_case["ground_truth"] = test_case["ground_truth"]
        
        test_cases_with_answers.append(eval_case)
    
    print(f"   ✓ Generated {len(test_cases_with_answers)} answers")
    
    # Run RAGAS evaluation
    print("\n5. Running RAGAS Evaluation...")
    print("   This may take a few minutes (LLM calls for faithfulness evaluation)...")
    
    try:
        results = scorer.batch_evaluate(test_cases_with_answers)
        print("   ✓ Evaluation complete")
    except Exception as e:
        print(f"   ✗ Evaluation failed: {e}")
        raise
    
    # Display results
    print("\n6. Evaluation Results")
    scorer.print_summary(results)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use custom paths if provided, otherwise use default naming
    if json_output_path is None:
        json_output_path = os.path.join(output_dir, f"ragas_evaluation_{timestamp}.json")
    if csv_output_path is None:
        csv_output_path = os.path.join(output_dir, f"ragas_summary_{timestamp}.csv")
    
    # Add metadata
    results["metadata"] = {
        "timestamp": timestamp,
        "model": llm_model,
        "use_ground_truth_metrics": use_ground_truth_metrics,
        "num_test_cases": len(test_cases_with_answers)
    }
    
    scorer.save_evaluation(results, json_output_path)
    
    # Also save a CSV summary for easy analysis
    save_csv_summary(results, csv_output_path)
    print(f"CSV summary saved to {csv_output_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70 + "\n")
    
    # Return aggregate results for comparison
    return results.get("aggregate_results", {})


def save_csv_summary(results: dict, output_path: str):
    """Save evaluation results as CSV for easy analysis."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        # Get all metric names
        if results["individual_results"]:
            metric_names = list(results["individual_results"][0]["metrics"].keys())
            fieldnames = ["question"] + metric_names
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results["individual_results"]:
                row = {"question": result["question"][:100]}  # Truncate for CSV
                row.update(result["metrics"])
                writer.writerow(row)


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on fraud detection chatbot"
    )
    
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="Include metrics that require ground truth (context_recall, answer_correctness)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use for evaluation (default: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Directory to save results (default: evaluation/results)"
    )
    
    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help="Number of test cases to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment")
        print("Please set it in your .env file or environment variables")
        return
    
    # Run evaluation
    run_evaluation(
        use_ground_truth_metrics=args.with_ground_truth,
        llm_model=args.model,
        output_dir=args.output_dir,
        num_test_cases=args.num_cases
    )


if __name__ == "__main__":
    main()
