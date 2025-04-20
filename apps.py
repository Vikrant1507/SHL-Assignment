# apps.py
import streamlit as st
import httpx
import os
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI

# Define API URL adjust based on deployment
API_URL = os.environ.get("API_URL", "http://192.168.0.102:8501")

st.set_page_config(
    page_title="SHL Assessment Recommender", page_icon="📊", layout="wide"
)


def check_api_health():
    """Check if the API is running."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            st.error(f"API service returned status code: {response.status_code}")
            return False
    except httpx.RequestError as e:
        st.error(f"Cannot connect to API service. Please check if it's running.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred.")
        return False


def get_recommendations(query: str, url: str = None):
    """Get assessment recommendations from the API."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            f"{API_URL}/recommend",
            json={"query": query, "url": url},
            headers=headers,
        )
        response.raise_for_status()
        return response.json().get("recommendations", [])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            st.warning(
                "No relevant assessments found. Please try different search terms."
            )
        else:
            st.error(f"API Error: {e.response.text}")
        return []
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []


def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Display recommendations in a table format."""
    if not recommendations:
        return

    # Convert to DataFrame for easier display
    df = pd.DataFrame(recommendations)

    # Rename columns for better display
    df = df.rename(
        columns={
            "name": "Assessment Name",
            "url": "URL",
            "remote_testing": "Remote Testing",
            "adaptive_irt": "Adaptive/IRT",
            "duration": "Duration",
            "test_type": "Test Type",
        }
    )

    # Make URLs clickable
    df["URL"] = df["URL"].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

    # Display table with HTML
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


def main():
    st.title("SHL Assessment Recommender")
    st.markdown(
        """
    This application helps you find the most relevant SHL assessments based on your job requirements.
    Enter a description of the role or specific skills you're looking for, or provide a job description URL.
    """
    )

    # Check API health
    api_healthy = check_api_health()
    if not api_healthy:
        st.error(
            "API service is not available. Please check your connection or try again later."
        )
        return

    st.success("API service is running")

    # Input fields
    query = st.text_area(
        "Enter your query or job requirements:",
        placeholder="",
    )

    url = st.text_input(
        "Or enter a job description URL (optional):",
        placeholder="",
    )

    # Search button
    if st.button("Find Relevant Assessments"):
        if not query and not url:
            st.warning("Please enter a query or provide a URL")
            return

        with st.spinner("Searching for relevant assessments..."):
            recommendations = get_recommendations(query, url)

        if recommendations:
            st.subheader("Recommended Assessments")
            display_recommendations(recommendations)


if __name__ == "__main__":
    main()
# streamlit run app.py
# http://localhost:8000/health
# uvicorn app:app --reload
