# cli.py
import argparse
import os
import json
from scrapper import SHLScraper
from embedding import EmbeddingEngine
from queryprocessor import QueryProcessor
from eval import EvaluationMetrics
import pandas as pd


class CommandLineInterface:
    def __init__(self):
        # Initialize components
        self.scraper = SHLScraper()
        self.assessments = self.scraper.load_data()
        self.embedding_engine = EmbeddingEngine()
        self.embedding_engine.process_assessments(self.assessments)
        self.query_processor = QueryProcessor(self.embedding_engine)

    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n===== SHL Assessment Recommendation System =====")
        print("Type 'exit' to quit, 'help' for commands\n")

        while True:
            query = input("\nEnter your query: ")

            if query.lower() == "exit":
                print("Goodbye!")
                break

            elif query.lower() == "help":
                print("\nAvailable commands:")
                print("  'exit' - Exit the program")
                print("  'help' - Show this help message")
                print("  'eval' - Run evaluation metrics")
                print("  'list' - List all available assessments")
                print("  anything else - Search for relevant assessments")

            elif query.lower() == "eval":
                print("\nRunning evaluation...")
                try:
                    results = run_evaluation()
                    print(
                        "\nEvaluation complete. Results saved to data/eval/results.csv"
                    )
                except Exception as e:
                    print(f"Error running evaluation: {str(e)}")

            elif query.lower() == "list":
                print("\nAvailable Assessments:")
                for i, assessment in enumerate(self.assessments):
                    print(
                        f"{i+1}. {assessment['name']} ({assessment.get('test_type', 'Unknown')})"
                    )

            else:
                constraints = self.query_processor.extract_constraints(query)
                print(f"\nExtracted constraints: {json.dumps(constraints, indent=2)}")

                results = self.query_processor.process_query(query)
                print(f"\nFound {len(results)} relevant assessments:")

                for i, result in enumerate(results):
                    print(f"{i+1}. {result['name']}")
                    print(f"   URL: {result.get('url', '#')}")
                    print(
                        f"   Remote Testing: {result.get('remote_testing', 'Unknown')}"
                    )
                    print(f"   Adaptive/IRT: {result.get('adaptive_irt', 'Unknown')}")
                    print(f"   Duration: {result.get('duration', 'Unknown')}")
                    print(f"   Test Type: {result.get('test_type', 'Unknown')}")
                    print(
                        f"   Description: {result.get('description', 'No description available')[:100]}..."
                    )
                    print()


def main():
    parser = argparse.ArgumentParser(description="SHL Assessment Recommendation CLI")
    parser.add_argument(
        "--scrape", action="store_true", help="Force re-scrape the SHL catalog"
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    args = parser.parse_args()

    # Re-scrape if requested
    if args.scrape:
        scraper = SHLScraper()
        scraper.scrape_catalog()
        print("Catalog re-scraped successfully")

    # Run evaluation if requested
    if args.eval:
        run_evaluation()
        return

    # Otherwise run interactive mode
    cli = CommandLineInterface()
    cli.interactive_mode()


if __name__ == "__main__":
    main()
