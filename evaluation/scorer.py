"""
Evaluation metrics for QA system quality scoring using RAGAS.

This module provides RAGAS-based evaluation metrics for the fraud detection chatbot:
- context_precision: Are retrieved contexts relevant to the question?
- answer_relevancy: Does the answer address the question?
- faithfulness: Is the answer grounded in the retrieved context?
- context_recall: Did we retrieve all necessary contexts? (requires ground_truth)
- answer_correctness: Semantic similarity to ground truth (requires ground_truth)
"""

from typing import List, Dict, Optional
import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
    answer_correctness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class QAScorer:
    """Score answer quality for QA system evaluation using RAGAS metrics."""
    
    def __init__(self, 
                 llm_model: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-ada-002",
                 use_ground_truth_metrics: bool = False):
        """
        Initialize RAGAS-based scorer.
        
        Args:
            llm_model: OpenAI model for LLM-based metrics (faithfulness)
            embedding_model: OpenAI embedding model for semantic similarity
            use_ground_truth_metrics: Whether to include metrics requiring ground truth
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.use_ground_truth_metrics = use_ground_truth_metrics
        
        # Initialize OpenAI clients for RAGAS
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Select metrics based on whether ground truth is available
        # Note: context_precision requires ground_truth in newer RAGAS versions
        if use_ground_truth_metrics:
            self.metrics = [
                context_precision,
                answer_relevancy,
                faithfulness,
                context_recall,
                answer_correctness
            ]
        else:
            # Reference-free metrics only
            self.metrics = [
                answer_relevancy,
                faithfulness
            ]
    
    def evaluate_single(self,
                       question: str,
                       answer: str,
                       contexts: List[str],
                       ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a single QA result using RAGAS metrics.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: List of retrieved context strings
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with RAGAS evaluation metrics
        """
        # Prepare data in RAGAS format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        
        # Add ground truth if provided and metrics require it
        if ground_truth and self.use_ground_truth_metrics:
            data["ground_truth"] = [ground_truth]
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Run RAGAS evaluation
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        # Convert EvaluationResult to pandas DataFrame for easier access
        result_df = result.to_pandas()
        
        # Extract metric scores from the first row
        scores = {}
        for col in result_df.columns:
            if col not in ['question', 'answer', 'contexts', 'ground_truth', 'user_input', 'retrieved_contexts', 'response']:
                value = result_df[col].iloc[0]
                scores[col] = float(value) if pd.notna(value) else 0.0
        
        return scores
    
    def batch_evaluate(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate multiple QA test cases using RAGAS.
        
        Args:
            test_cases: List of dicts with 'question', 'answer', 'contexts', 'ground_truth' (optional)
            
        Returns:
            Aggregated evaluation results with individual and average metrics
        """
        # Prepare data for batch evaluation
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        
        has_ground_truth = all("ground_truth" in tc for tc in test_cases)
        
        for test_case in test_cases:
            questions.append(test_case["question"])
            answers.append(test_case["answer"])
            contexts_list.append(test_case["contexts"])
            if has_ground_truth:
                ground_truths.append(test_case["ground_truth"])
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list
        }
        
        if has_ground_truth and self.use_ground_truth_metrics:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Run RAGAS evaluation
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        # Convert EvaluationResult to pandas DataFrame for easier access
        result_df = result.to_pandas()
        
        # Extract individual results
        individual_results = []
        for i in range(len(test_cases)):
            test_result = {
                "question": questions[i],
                "metrics": {}
            }
            # Get metric columns (exclude input columns)
            for col in result_df.columns:
                if col not in ["question", "answer", "contexts", "ground_truth", "user_input", "retrieved_contexts", "response"]:
                    value = result_df[col].iloc[i]
                    # Only convert numeric values to float
                    if pd.notna(value):
                        try:
                            test_result["metrics"][col] = float(value)
                        except (ValueError, TypeError):
                            # Skip non-numeric columns
                            continue
                    else:
                        test_result["metrics"][col] = 0.0
            individual_results.append(test_result)
        
        # Calculate average metrics
        avg_metrics = {}
        metric_columns = []
        
        # Only include numeric columns for averaging
        for col in result_df.columns:
            if col not in ["question", "answer", "contexts", "ground_truth", "user_input", "retrieved_contexts", "response"]:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    metric_columns.append(col)
                    values = result_df[col].dropna()
                    avg_metrics[col] = float(values.mean()) if len(values) > 0 else 0.0
        
        return {
            "individual_results": individual_results,
            "average_metrics": avg_metrics,
            "num_test_cases": len(test_cases),
            "metrics_used": metric_columns
        }
    
    def save_evaluation(self, results: Dict, output_path: str = "evaluation_results.json"):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """
        Print a formatted summary of evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("RAGAS Evaluation Summary")
        print("="*60)
        print(f"Number of test cases: {results['num_test_cases']}")
        print(f"Metrics used: {', '.join(results.get('metrics_used', []))}")
        print("\nAverage Metrics:")
        print("-"*60)
        
        for metric, score in results["average_metrics"].items():
            print(f"  {metric:.<30} {score:.4f}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage with RAGAS
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize scorer
    scorer = QAScorer(
        llm_model="gpt-3.5-turbo",
        use_ground_truth_metrics=False  # Set to True if you have ground truth
    )
    
    # Single evaluation example
    question = "What are common fraud indicators?"
    answer = "Common fraud indicators include unusual transaction amounts, rapid succession of transactions, and transactions from high-risk locations."
    contexts = [
        "Fraud detection systems monitor for suspicious patterns such as unusually large amounts.",
        "Multiple transactions in short time periods can indicate fraudulent activity.",
        "Geographical anomalies are strong indicators of potential fraud."
    ]
    
    print("Evaluating single QA result...")
    metrics = scorer.evaluate_single(question, answer, contexts)
    
    print("\nEvaluation Metrics:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")
