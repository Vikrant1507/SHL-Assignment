# query_processor.py
import re
from typing import Dict, List, Tuple, Any
import json


class QueryProcessor:
    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine

    def extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints like time limits and skills from the query."""
        constraints = {}

        # Extract time constraints
        time_pattern = r"(\d+)\s*minutes"
        max_time_pattern = r"(less than|max|maximum|under|within|up to)\s*(\d+)\s*min"

        # Check for maximum time constraints
        max_time_match = re.search(max_time_pattern, query.lower())
        if max_time_match:
            constraints["max_duration"] = int(max_time_match.group(2))

        # Check for exact time constraints
        time_match = re.search(time_pattern, query)
        if time_match and not max_time_match:
            constraints["duration"] = int(time_match.group(1))

        # Extract skills/technologies
        tech_skills = [
            "java",
            "python",
            "javascript",
            "js",
            "sql",
            "react",
            "angular",
            "node",
            "c\+\+",
            "c#",
        ]
        found_skills = []

        for skill in tech_skills:
            if re.search(rf"\b{skill}\b", query.lower()):
                found_skills.append(skill)

        if found_skills:
            constraints["skills"] = found_skills

        # Extract test types
        test_types = [
            "cognitive",
            "personality",
            "behavior",
            "situational",
            "technical",
            "coding",
        ]
        found_types = []

        for test_type in test_types:
            if re.search(rf"\b{test_type}\b", query.lower()):
                found_types.append(test_type)

        if found_types:
            constraints["test_types"] = found_types

        return constraints

    def filter_assessments(
        self, assessments: List[Dict[str, Any]], constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter assessments based on extracted constraints."""
        filtered = assessments.copy()

        # Filter by duration if specified
        if "max_duration" in constraints:
            filtered = [
                a
                for a in filtered
                if "duration" in a
                and re.search(r"(\d+)", a["duration"])
                and int(re.search(r"(\d+)", a["duration"]).group(1))
                <= constraints["max_duration"]
            ]
        elif "duration" in constraints:
            filtered = [
                a
                for a in filtered
                if "duration" in a
                and re.search(r"(\d+)", a["duration"])
                and int(re.search(r"(\d+)", a["duration"]).group(1))
                == constraints["duration"]
            ]

        # Filter by skills if specified
        if "skills" in constraints and constraints["skills"]:
            skill_matches = []
            for assessment in filtered:
                # Check if any skill is mentioned in the assessment name or description
                for skill in constraints["skills"]:
                    if re.search(
                        rf"\b{skill}\b", assessment.get("name", "").lower()
                    ) or re.search(
                        rf"\b{skill}\b", assessment.get("description", "").lower()
                    ):
                        skill_matches.append(assessment)
                        break

            if skill_matches:
                filtered = skill_matches

        # Filter by test types if specified
        if "test_types" in constraints and constraints["test_types"]:
            type_matches = []
            for assessment in filtered:
                # Check if any test type is mentioned in the assessment type or description
                test_type = assessment.get("test_type", "").lower()
                for t_type in constraints["test_types"]:
                    if t_type in test_type or re.search(
                        rf"\b{t_type}\b", assessment.get("description", "").lower()
                    ):
                        type_matches.append(assessment)
                        break

            if type_matches:
                filtered = type_matches

        return filtered

    def process_query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Process a query to return recommended assessments."""
        # 1. Extract constraints from the query
        constraints = self.extract_constraints(query)

        # 2. Perform semantic search
        search_results = self.embedding_engine.search(
            query, n_results=20
        )  # Get more results for filtering

        # 3. Apply constraint-based filtering
        filtered_results = self.filter_assessments(search_results, constraints)

        # 4. Return top results (limited by max_results)
        return (
            filtered_results[:max_results]
            if filtered_results
            else search_results[:max_results]
        )
