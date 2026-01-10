#!/usr/bin/env python3
"""
Quick example of using RAGAS evaluation for a single question.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from llm.qa_chain import QAChain
from evaluation.scorer import QAScorer

load_dotenv()

def main():
    print("\n" + "="*70)
    print("RAGAS Evaluation - Single Question Example")
    print("="*70 + "\n")
    
    # Initialize QA chain
    print("1. Initializing QA Chain...")
    qa_chain = QAChain(llm_provider="openai", model_name="gpt-3.5-turbo")
    
    # Initialize RAGAS scorer
    print("2. Initializing RAGAS Scorer...")
    scorer = QAScorer(
        llm_model="gpt-3.5-turbo",
        use_ground_truth_metrics=False
    )
    
    # Ask a question
    question = "What percentage of transactions are fraudulent?"
    print(f"\n3. Asking question: '{question}'")
    
    # Get answer in RAGAS format
    result = qa_chain.ask_for_evaluation(question)
    
    print(f"\n4. Answer Generated:")
    print(f"   {result['answer']}\n")
    
    print(f"5. Contexts Retrieved: {len(result['contexts'])} passages")
    for i, ctx in enumerate(result['contexts'][:2], 1):  # Show first 2
        print(f"   [{i}] {ctx[:100]}...")
    
    # Evaluate with RAGAS
    print(f"\n6. Running RAGAS Evaluation...")
    metrics = scorer.evaluate_single(
        question=result["question"],
        answer=result["answer"],
        contexts=result["contexts"]
    )
    
    print(f"\n7. RAGAS Metrics:")
    print("   " + "-"*50)
    for metric, score in metrics.items():
        print(f"   {metric:.<40} {score:.4f}")
    print("   " + "-"*50)
    
    # Interpret results
    print(f"\n8. Interpretation:")
    if metrics.get("faithfulness", 0) > 0.8:
        print("   ✓ High faithfulness - answer is well grounded in context")
    else:
        print("   ⚠ Low faithfulness - answer may contain hallucinations")
    
    if metrics.get("answer_relevancy", 0) > 0.7:
        print("   ✓ Good relevancy - answer addresses the question")
    else:
        print("   ⚠ Low relevancy - answer may be off-topic")
    
    if metrics.get("context_precision", 0) > 0.7:
        print("   ✓ Good precision - retrieved contexts are relevant")
    else:
        print("   ⚠ Low precision - some contexts may be irrelevant")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
