from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any


class EmbeddingEngine:
    def __init__(self):
        """Initialize the Embedding Engine with model and database configuration."""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db_path = "data/chroma_db"
        self.collection_name = "shl_assessments"
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Load or create the collection."""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Loaded existing collection with {self.collection.count()} items")
        except Exception as e:
            self.collection = self.client.create_collection(self.collection_name)
            print(f"Created new collection due to error: {e}")

    def create_document_text(self, assessment: Dict[str, Any]) -> str:
        """Create a searchable text representation of an assessment."""
        name = assessment.get("name", "Unknown Name")
        description = assessment.get("description", "No description available")
        test_type = assessment.get("test_type", "Unknown")
        duration = assessment.get("duration", "Unknown")

        return f"{name}. {description} Type: {test_type}. Duration: {duration}."

    def process_assessments(self, assessments: List[Dict[str, Any]]):
        """Process assessments, generate embeddings, and store them in the vector database."""

        if not assessments:
            print("No assessments provided. Skipping processing.")
            return

        if self.collection.count() > 0:
            print(
                "Collection already contains documents, skipping embedding generation."
            )
            return

        print("Processing assessments and generating embeddings...")

        # Prepare data for insertion
        ids = [str(i) for i in range(len(assessments))]

        # Generate document text from assessments
        documents = [
            self.create_document_text(assessment) for assessment in assessments
        ]
        metadatas = assessments  # Store the full assessment data as metadata

        # Debug: Check if all lists are populated correctly
        print(f"Documents: {documents}")
        print(f"IDs: {ids}")
        print(f"Metadatas: {metadatas}")

        # Check if any of the lists are empty
        if not documents or not ids or not metadatas:
            print(
                "Error: One or more lists (documents, ids, metadatas) are empty. Skipping insertion."
            )
            return

        # Insert data into the collection
        try:
            self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
            print(f"Generated embeddings for {len(assessments)} assessments.")
        except Exception as e:
            print(f"Error occurred while adding documents to the collection: {e}")

    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant assessments based on a query."""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)

            # Process results
            if results and "metadatas" in results and results["metadatas"]:
                return results["metadatas"][
                    0
                ]  # Return the first element since it's a single query
            else:
                print("No matching results found.")
                return []
        except Exception as e:
            print(f"Error during search: {e}")
            return []
