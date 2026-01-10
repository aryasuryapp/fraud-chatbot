"""
Test dataset for RAGAS evaluation of the fraud detection chatbot.

This module provides sample question-answer-context triples for evaluating
the RAG system using RAGAS metrics.
"""

# Sample test dataset for fraud detection queries
# Format: Each test case contains:
# - question: The user query
# - ground_truth: Expected/reference answer (optional for some metrics)
# - contexts: List of relevant context strings that should be retrieved
TEST_DATASET = [
    {
        "question": "What percentage of transactions are fraudulent?",
        "ground_truth": "Approximately 0.17% of transactions in the dataset are fraudulent.",
        "contexts": [
            "The fraud rate in credit card transactions is typically very low, around 0.1-0.2% of all transactions.",
            "Out of the total transactions analyzed, fraudulent transactions represent a small fraction, approximately 0.17%."
        ]
    },
    {
        "question": "What are common fraud patterns in online transactions?",
        "ground_truth": "Common fraud patterns include unusual transaction amounts, multiple transactions in quick succession, transactions from unusual geographic locations, and purchases from high-risk merchant categories.",
        "contexts": [
            "Fraud patterns often include rapid-fire transactions within short time windows, which may indicate stolen card use.",
            "Unusual transaction amounts, particularly those just below reporting thresholds, are common indicators of fraud.",
            "Geographic anomalies, such as transactions from locations far from the cardholder's typical locations, suggest potential fraud.",
            "High-risk merchant categories like online gambling, electronics, and jewelry are frequently targeted by fraudsters."
        ]
    },
    {
        "question": "How can machine learning help detect fraud?",
        "ground_truth": "Machine learning models can analyze patterns in transaction data to identify anomalies and suspicious behavior. They can learn from historical fraud cases to predict which transactions are likely fraudulent based on features like amount, location, time, and merchant category.",
        "contexts": [
            "Machine learning algorithms excel at identifying complex patterns in transaction data that may indicate fraudulent activity.",
            "Supervised learning models can be trained on labeled fraud data to predict the likelihood of fraud in new transactions.",
            "Feature engineering is crucial - models typically use transaction amount, time, location, merchant category, and historical behavior patterns.",
            "Ensemble methods and neural networks have shown high accuracy in fraud detection, reducing false positives while catching genuine fraud cases."
        ]
    },
    {
        "question": "What are the key features used in fraud detection models?",
        "ground_truth": "Key features include transaction amount, transaction time and date, merchant category, geographic location, card holder's historical spending patterns, time since last transaction, and distance from previous transaction location.",
        "contexts": [
            "Transaction amount is a critical feature - unusually high or low amounts compared to historical behavior can indicate fraud.",
            "Temporal features like transaction time, day of week, and time since previous transaction help identify suspicious patterns.",
            "Geographic features including transaction location, distance from home address, and distance from previous transaction are important.",
            "Merchant category codes (MCC) help identify high-risk categories commonly associated with fraud.",
            "Behavioral features capturing the cardholder's typical spending patterns and deviations from normal behavior are essential."
        ]
    },
    {
        "question": "What is the average transaction amount for fraudulent vs non-fraudulent transactions?",
        "ground_truth": "The average transaction amount varies, but fraudulent transactions often have different patterns - they may be higher value to maximize gain, or lower value to avoid detection thresholds.",
        "contexts": [
            "Fraudulent transactions may target specific amount ranges to maximize value while avoiding detection systems.",
            "Statistical analysis shows that fraud patterns vary by merchant category and payment method.",
            "Comparing average amounts between fraud and legitimate transactions reveals important behavioral differences."
        ]
    },
    {
        "question": "Which merchant categories have the highest fraud rates?",
        "ground_truth": "Online shopping, electronics stores, gas stations, and entertainment venues typically have higher fraud rates compared to other categories.",
        "contexts": [
            "E-commerce and online merchants face higher fraud rates due to card-not-present transactions.",
            "Gas stations are frequently targeted for fraud due to automated payment systems and less verification.",
            "Electronics and high-value goods merchants experience elevated fraud due to resale value of products.",
            "Entertainment and travel-related merchants see higher fraud rates from stolen card usage."
        ]
    },
    {
        "question": "How does transaction timing affect fraud likelihood?",
        "ground_truth": "Transactions during unusual hours (late night/early morning), multiple transactions in rapid succession, and transactions shortly after card activation or reported loss show higher fraud likelihood.",
        "contexts": [
            "Timing patterns are crucial indicators - fraudsters often make purchases immediately after obtaining card information.",
            "Late night and early morning transactions, especially if inconsistent with cardholder patterns, may indicate fraud.",
            "Multiple transactions within minutes suggest automated testing or bulk purchasing by fraudsters.",
            "Time intervals between transactions help identify velocity-based fraud patterns."
        ]
    },
    {
        "question": "What is the role of geographic location in fraud detection?",
        "ground_truth": "Geographic location is a key fraud indicator. Transactions from unexpected locations, rapid geographic movement between transactions, and high-risk geographic regions all contribute to fraud risk assessment.",
        "contexts": [
            "Geographic location analysis compares transaction location to cardholder's home, work, and recent transaction locations.",
            "Impossible travel scenarios - transactions in distant locations within implausible timeframes - are strong fraud indicators.",
            "Certain geographic regions have higher fraud rates and are flagged in risk models.",
            "IP address geolocation for online transactions helps verify consistency with billing address."
        ]
    },
    {
        "question": "What are best practices for preventing credit card fraud?",
        "ground_truth": "Best practices include real-time transaction monitoring, multi-factor authentication, velocity checks, geographic verification, behavioral analysis, and setting transaction limits.",
        "contexts": [
            "Real-time monitoring systems can flag and block suspicious transactions before completion.",
            "Multi-factor authentication adds security layers for online and high-value transactions.",
            "Velocity checks limit the number of transactions within specified time windows.",
            "Customer education about secure card usage and prompt fraud reporting is essential.",
            "EMV chip cards and tokenization technology significantly reduce fraud risk."
        ]
    },
    {
        "question": "How do false positives impact fraud detection systems?",
        "ground_truth": "False positives occur when legitimate transactions are flagged as fraud, leading to customer friction, declined sales, and reduced trust. Balancing fraud detection with minimizing false positives is a key challenge.",
        "contexts": [
            "High false positive rates frustrate customers and can lead to account abandonment.",
            "Each declined legitimate transaction represents lost revenue and damaged customer relationships.",
            "Modern systems aim to reduce false positives through machine learning and behavioral analysis.",
            "The cost of false positives must be balanced against the cost of missed fraud cases.",
            "Adaptive models that learn from customer feedback help reduce false positive rates over time."
        ]
    }
]


def get_test_dataset():
    """
    Returns the test dataset for RAGAS evaluation.
    
    Returns:
        list: List of dictionaries containing question, ground_truth, and contexts
    """
    return TEST_DATASET


def get_dataset_size():
    """
    Returns the number of test cases in the dataset.
    
    Returns:
        int: Number of test cases
    """
    return len(TEST_DATASET)


def get_test_case(index):
    """
    Get a specific test case by index.
    
    Args:
        index (int): Index of the test case
        
    Returns:
        dict: Test case dictionary
        
    Raises:
        IndexError: If index is out of range
    """
    if 0 <= index < len(TEST_DATASET):
        return TEST_DATASET[index]
    else:
        raise IndexError(f"Test case index {index} out of range (0-{len(TEST_DATASET)-1})")


if __name__ == "__main__":
    # Print dataset summary
    print(f"Test Dataset Summary")
    print(f"{'='*50}")
    print(f"Total test cases: {get_dataset_size()}")
    print(f"\nSample test cases:")
    for i, case in enumerate(TEST_DATASET[:3], 1):
        print(f"\n{i}. Question: {case['question']}")
        print(f"   Contexts: {len(case['contexts'])} passages")
        print(f"   Has ground truth: {'Yes' if case.get('ground_truth') else 'No'}")
