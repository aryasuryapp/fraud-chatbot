"""
Evaluation metrics for QA system quality scoring.
"""

from typing import List, Dict
import json


class QAScorer:
    """Score answer quality for QA system evaluation."""
    
    def __init__(self):
        """Initialize scorer."""
        self.metrics = []
    
    def calculate_retrieval_metrics(self, 
                                   retrieved_docs: List[str], 
                                   relevant_docs: List[str]) -> Dict[str, float]:
        """
        Calculate retrieval quality metrics.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of ground truth relevant document IDs
            
        Returns:
            Dictionary with precision, recall, F1 scores
        """
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        # Calculate metrics
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives
        }
    
    def calculate_answer_relevance(self, question: str, answer: str, context: str) -> float:
        """
        Calculate answer relevance score using simple heuristics.
        
        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Relevance score (0-1)
        """
        # Check if answer contains key terms from question
        question_terms = set(question.lower().split())
        answer_terms = set(answer.lower().split())
        
        # Term overlap
        overlap = len(question_terms & answer_terms) / len(question_terms) if question_terms else 0
        
        # Check if answer uses context
        context_usage = 0.5 if any(word in answer.lower() for word in context.lower().split()) else 0
        
        # Length penalty (too short = low quality)
        length_score = min(len(answer.split()) / 50, 1.0)
        
        relevance = (overlap * 0.4) + (context_usage * 0.4) + (length_score * 0.2)
        
        return min(relevance, 1.0)
    
    def calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Calculate faithfulness score (answer grounded in context).
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Faithfulness score (0-1)
        """
        # Simple heuristic: check if answer statements appear in context
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        grounded_count = 0
        for sentence in answer_sentences:
            # Check if key words from sentence appear in context
            sentence_words = set(sentence.lower().split())
            context_words = set(context.lower().split())
            
            overlap = len(sentence_words & context_words) / len(sentence_words) if sentence_words else 0
            
            if overlap > 0.5:  # At least 50% overlap
                grounded_count += 1
        
        return grounded_count / len(answer_sentences)
    
    def evaluate_qa_result(self, 
                          question: str, 
                          answer: str, 
                          context: str,
                          ground_truth: str = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of QA result.
        
        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "relevance": self.calculate_answer_relevance(question, answer, context),
            "faithfulness": self.calculate_faithfulness(answer, context),
            "answer_length": len(answer.split())
        }
        
        # If ground truth is provided, calculate similarity
        if ground_truth:
            metrics["ground_truth_overlap"] = self._calculate_text_similarity(answer, ground_truth)
        
        # Overall score (weighted average)
        metrics["overall_score"] = (
            metrics["relevance"] * 0.4 +
            metrics["faithfulness"] * 0.4 +
            min(metrics["answer_length"] / 100, 1.0) * 0.2
        )
        
        return metrics
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-level similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def batch_evaluate(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate multiple QA test cases.
        
        Args:
            test_cases: List of dicts with 'question', 'answer', 'context', 'ground_truth'
            
        Returns:
            Aggregated evaluation results
        """
        results = []
        
        for test_case in test_cases:
            metrics = self.evaluate_qa_result(
                question=test_case["question"],
                answer=test_case["answer"],
                context=test_case["context"],
                ground_truth=test_case.get("ground_truth")
            )
            
            results.append({
                "question": test_case["question"],
                "metrics": metrics
            })
        
        # Calculate average metrics
        avg_metrics = {
            "relevance": sum(r["metrics"]["relevance"] for r in results) / len(results),
            "faithfulness": sum(r["metrics"]["faithfulness"] for r in results) / len(results),
            "overall_score": sum(r["metrics"]["overall_score"] for r in results) / len(results)
        }
        
        return {
            "individual_results": results,
            "average_metrics": avg_metrics,
            "num_test_cases": len(test_cases)
        }
    
    def save_evaluation(self, results: Dict, output_path: str = "evaluation_results.json"):
        """Save evaluation results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    scorer = QAScorer()
    
    # Single evaluation
    question = "What are common fraud indicators?"
    answer = "Common fraud indicators include unusual transaction amounts, rapid succession of transactions, and transactions from high-risk locations."
    context = "Fraud detection systems monitor for suspicious patterns such as unusually large amounts, multiple transactions in short time periods, and geographical anomalies."
    
    metrics = scorer.evaluate_qa_result(question, answer, context)
    
    print("Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.3f}")
