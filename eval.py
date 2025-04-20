# eval.py
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class EvaluationMetrics:
    def __init__(self, test_queries_path: str, groundtruth_path: str):
        """
        Initialize the evaluation metrics calculator.

        Args:
            test_queries_path: Path to test queries JSON file
            groundtruth_path: Path to ground truth JSON file
        """
        self.test_queries = self.load_json(test_queries_path)
        self.groundtruth = self.load_json(groundtruth_path)

    def load_json(self, path: str) -> Dict:
        """Load JSON data from file."""
        with open(path, "r") as f:
            return json.load(f)

    def calculate_recall_at_k(
        self, query_id: str, predictions: List[Dict[str, Any]], k: int = 3
    ) -> float:
        """
        Calculate Recall@K for a single query.

        Args:
            query_id: ID of the query
            predictions: List of predicted assessments
            k: K value for Recall@K
             k: K value for Recall@K

        Returns:
            Recall@K score
        """
        if query_id not in self.groundtruth:
            print(f"Warning: No ground truth data for query ID {query_id}")
            return 0.0

        # Get relevant assessment IDs from ground truth
        relevant_ids = set(self.groundtruth[query_id])

        # Get predicted assessment IDs (top K)
        predicted_ids = [pred["name"] for pred in predictions[:k]]

        # Calculate recall
        hits = len(set(predicted_ids).intersection(relevant_ids))
        recall = hits / len(relevant_ids) if relevant_ids else 0.0

        return recall

    def calculate_precision_at_k(
        self, query_id: str, predictions: List[Dict[str, Any]], k: int = 3
    ) -> float:
        """
        Calculate Precision@K for a single query.

        Args:
            query_id: ID of the query
            predictions: List of predicted assessments
            k: K value for Precision@K

        Returns:
            Precision@K score
        """
        if query_id not in self.groundtruth:
            print(f"Warning: No ground truth data for query ID {query_id}")
            return 0.0

        # Get relevant assessment IDs from ground truth
        relevant_ids = set(self.groundtruth[query_id])

        # Get predicted assessment IDs (top K)
        predicted_ids = [pred["name"] for pred in predictions[:k]]

        # Calculate precision
        hits = len(set(predicted_ids).intersection(relevant_ids))
        precision = hits / len(predicted_ids[:k]) if predicted_ids[:k] else 0.0

        return precision

    def calculate_ndcg_at_k(
        self, query_id: str, predictions: List[Dict[str, Any]], k: int = 3
    ) -> float:
        """
        Calculate NDCG@K for a single query.

        Args:
            query_id: ID of the query
            predictions: List of predicted assessments
            k: K value for NDCG@K

        Returns:
            NDCG@K score
        """
        if query_id not in self.groundtruth:
            print(f"Warning: No ground truth data for query ID {query_id}")
            return 0.0

        # Get relevant assessment IDs from ground truth
        relevant_ids = set(self.groundtruth[query_id])

        # Get predicted assessment IDs (top K)
        predicted_ids = [pred["name"] for pred in predictions[:k]]

        # Calculate DCG
        dcg = 0.0
        for i, pred_id in enumerate(predicted_ids):
            if pred_id in relevant_ids:
                # Using binary relevance (1 if relevant, 0 if not)
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

        # Calculate IDCG (best possible DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_ids))):
            idcg += 1.0 / np.log2(i + 2)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return ndcg

    def evaluate_all_metrics(
        self, query_processor, k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all metrics for all test queries.

        Args:
            query_processor: QueryProcessor instance
            k_values: List of K values for the metrics

        Returns:
            Dictionary of metrics
        """
        results = {
            "recall": {k: [] for k in k_values},
            "precision": {k: [] for k in k_values},
            "ndcg": {k: [] for k in k_values},
        }

        for query_id, query_text in self.test_queries.items():
            # Process query
            predictions = query_processor.process_query(
                query_text, max_results=max(k_values)
            )

            # Calculate metrics for each K
            for k in k_values:
                recall = self.calculate_recall_at_k(query_id, predictions, k)
                precision = self.calculate_precision_at_k(query_id, predictions, k)
                ndcg = self.calculate_ndcg_at_k(query_id, predictions, k)

                results["recall"][k].append(recall)
                results["precision"][k].append(precision)
                results["ndcg"][k].append(ndcg)

        # Calculate average metrics
        avg_results = {
            "recall": {k: np.mean(values) for k, values in results["recall"].items()},
            "precision": {
                k: np.mean(values) for k, values in results["precision"].items()
            },
            "ndcg": {k: np.mean(values) for k, values in results["ndcg"].items()},
        }

        return avg_results

    def print_results(self, results: Dict[str, Dict[str, float]]):
        """
        Print evaluation results in a readable format.

        Args:
            results: Dictionary of metrics
        """
        print("\n===== EVALUATION RESULTS =====")

        for metric_name, k_values in results.items():
            print(f"\n{metric_name.upper()}:")
            for k, value in sorted(k_values.items()):
                print(f"  @{k}: {value:.4f}")

    def save_results_to_csv(
        self, results: Dict[str, Dict[str, float]], output_path: str
    ):
        """
        Save evaluation results to a CSV file.

        Args:
            results: Dictionary of metrics
            output_path: Path to save CSV file
        """
        # Prepare data for DataFrame
        data = []
        for metric_name, k_values in results.items():
            for k, value in k_values.items():
                data.append({"Metric": metric_name, "K": k, "Value": value})

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


# Example usage
if __name__ == "__main__":
    import argparse
    from scrapper import SHLScraper
    from embedding import EmbeddingEngine
    from queryprocessor import QueryProcessor
    import os

    parser = argparse.ArgumentParser(
        description="Evaluate SHL Assessment Recommendation System"
    )
    parser.add_argument(
        "--test_queries",
        type=str,
        default="data/test_queries.json",
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--groundtruth",
        type=str,
        default="data/groundtruth.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation_results.csv",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Create sample test data if it doesn't exist (for demonstration)
    if not os.path.exists(args.test_queries):
        os.makedirs(os.path.dirname(args.test_queries), exist_ok=True)

        test_queries = {
            "query1": "I need a programming assessment for Java developers that takes less than 30 minutes",
            "query2": "Looking for personality assessments for team collaboration",
            "query3": "Need cognitive assessments for data analyst role",
            "query4": "Python coding test for senior developers",
            "query5": "SQL assessment for database administrators under 45 minutes",
        }

        with open(args.test_queries, "w") as f:
            json.dump(test_queries, f, indent=2)

    # Create sample ground truth data if it doesn't exist
    if not os.path.exists(args.groundtruth):
        os.makedirs(os.path.dirname(args.groundtruth), exist_ok=True)

        groundtruth = {
            "query1": [
                "SHL Coding Simulations - Java",
                "SHL Software Development Aptitude Test",
            ],
            "query2": [
                "SHL OPQ - Occupational Personality Questionnaire",
                "SHL Teamwork Styles Assessment",
            ],
            "query3": [
                "SHL Verify Interactive - Deductive Reasoning",
                "SHL Verify - Numerical Reasoning",
            ],
            "query4": ["SHL Coding Simulations - Python"],
            "query5": ["SHL Coding Simulations - SQL"],
        }

        with open(args.groundtruth, "w") as f:
            json.dump(groundtruth, f, indent=2)

    # Initialize components
    print("Loading data and initializing components...")
    scraper = SHLScraper()
    assessments = scraper.load_data()
    embedding_engine = EmbeddingEngine()
    embedding_engine.process_assessments(assessments)
    query_processor = QueryProcessor(embedding_engine)

    # Initialize evaluation metrics calculator
    evaluator = EvaluationMetrics(args.test_queries, args.groundtruth)

    # Evaluate the system
    print("Evaluating recommendations...")
    results = evaluator.evaluate_all_metrics(query_processor, k_values=[1, 3, 5, 10])

    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results_to_csv(results, args.output)
