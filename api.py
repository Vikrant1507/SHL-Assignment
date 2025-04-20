from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests
from bs4 import BeautifulSoup
import logging

# Internal modules
from scrapper import SHLScraper
from embedding import EmbeddingEngine
from queryprocessor import QueryProcessor

# -------------------- Setup Logging -------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Initialize FastAPI App -------------------- #
app = FastAPI(title="SHL Assessment Recommender API")

# -------------------- CORS Middleware -------------------- #
origins = ["http://localhost:5500", "http://127.0.0.1:5500"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Pydantic Models -------------------- #
class QueryRequest(BaseModel):
    query: str
    url: Optional[str] = None


class AssessmentResponse(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    duration: str
    test_type: str


class RecommendResponse(BaseModel):
    recommendations: List[AssessmentResponse]


# -------------------- Initialize Core Components -------------------- #
scraper = SHLScraper()
embedding_engine = EmbeddingEngine()
query_processor = QueryProcessor(embedding_engine)


# -------------------- Startup Event -------------------- #
@app.on_event("startup")
async def startup_event():
    logger.info("Loading data and generating embeddings...")
    try:
        assessments = scraper.load_data()
        embedding_engine.process_assessments(assessments)
        logger.info("System initialized successfully.")
    except Exception as e:
        logger.exception(f"Startup error: {e}")
        raise RuntimeError(f"Failed during startup: {e}")


# -------------------- Utility: Extract Text From URL -------------------- #
def extract_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()

        raw_text = soup.get_text(separator="\n")
        cleaned_text = "\n".join(
            line.strip() for line in raw_text.splitlines() if line.strip()
        )
        return cleaned_text[:1500] + "..." if len(cleaned_text) > 1500 else cleaned_text

    except Exception as e:
        logger.error(f"URL extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"URL error: {str(e)}")


# -------------- Health Check Endpoint --------------- #
@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "ok", "message": "API is live"}


# -------------- Recommendation Endpoint --------- #
@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: QueryRequest):
    query_text = request.query.strip()

    # Append job description text from URL if provided
    if request.url:
        url_text = extract_text_from_url(request.url)
        query_text = f"{query_text} {url_text}" if query_text else url_text

    if not query_text or len(query_text) < 3:
        raise HTTPException(
            status_code=400, detail="Query must be at least 3 characters long."
        )

    try:
        recommendations = query_processor.process_query(query_text, max_results=10)

        # if not recommendations:
        #     raise HTTPException(
        #         status_code=404, detail="No matching assessments found."
        #     )
        if not recommendations:
            return JSONResponse(
                status_code=404, content={"detail": "No relevant assessments found"}
            )

        response = [
            AssessmentResponse(
                name=rec.get("name", "Unknown"),
                url=rec.get("url", "#"),
                remote_testing=rec.get("remote_testing", "Unknown"),
                adaptive_irt=rec.get("adaptive_irt", "Unknown"),
                duration=rec.get("duration", "Unknown"),
                test_type=rec.get("test_type", "Unknown"),
            )
            for rec in recommendations
        ]

        return RecommendResponse(recommendations=response)

    except Exception as e:
        logger.exception(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8501, reload=True)
