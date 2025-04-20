# scraper.py

import requests
from bs4 import BeautifulSoup
import json
import os
import re


class SHLScraper:
    def __init__(self):
        self.url = "https://www.shl.com/solutions/products/product-catalog/view/account-manager-solution/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.data_path = "shl_catalog.json"

    def scrape_catalog(self):
        print("Scraping SHL product catalog...")
        response = requests.get(self.url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch catalog page. Status code: {response.status_code}"
            )

        soup = BeautifulSoup(response.text, "html.parser")
        assessments = []

        selectors_to_try = [
            ".product-item",
            ".assessment-card",
            ".product-card",
            ".catalog-item",
            ".test-product",
            ".assessment-item",
            'div[data-product-type="assessment"]',
        ]

        assessment_elements = []
        for selector in selectors_to_try:
            assessment_elements = soup.select(selector)
            if assessment_elements:
                print(f"Found assessment elements using selector: {selector}")
                break

        if not assessment_elements:
            print("Attempting fallback extraction based on page layout...")
            sections = soup.select(
                "section, div.products, div.catalog, div.assessments"
            )
            for section in sections:
                items = section.find_all(["div", "article", "li"], class_=True)
                grouped = {}
                for item in items:
                    class_key = " ".join(item.get("class", []))
                    grouped.setdefault(class_key, []).append(item)

                for class_key, elements in grouped.items():
                    if len(elements) >= 3:
                        print(
                            f"Fallback: found potential elements with class: {class_key}"
                        )
                        assessment_elements = elements
                        break
                if assessment_elements:
                    break

        for element in assessment_elements:
            assessment = {
                "name": None,
                "url": None,
                "description": None,
                "remote_testing": "Unknown",
                "adaptive_irt": "Unknown",
                "duration": "Unknown",
                "test_type": "Unknown",
            }

            # Extract name
            for selector in [
                ".product-name",
                ".title",
                "h2",
                "h3",
                ".card-title",
                "strong",
                ".assessment-title",
            ]:
                name_element = element.select_one(selector)
                if name_element and name_element.text.strip():
                    assessment["name"] = name_element.text.strip()
                    break

            # Extract URL
            link_element = element.select_one("a[href]")
            if link_element and link_element.get("href"):
                href = link_element["href"]
                base_url = "/".join(self.url.split("/")[:3])
                assessment["url"] = (
                    href
                    if href.startswith("http")
                    else f"{base_url}/{href.lstrip('/')}"
                )

            # Extract description
            for selector in [
                ".product-description",
                ".description",
                ".card-text",
                "p",
                ".summary",
                ".assessment-desc",
            ]:
                desc_elements = element.select(selector)
                if desc_elements:
                    texts = [
                        desc.text.strip() for desc in desc_elements if desc.text.strip()
                    ]
                    assessment["description"] = " ".join(texts)
                    break

            # Extract features
            for selector in [
                ".product-features li",
                ".details li",
                ".features li",
                ".specs li",
                "ul li",
                "dl dt, dl dd",
            ]:
                features = element.select(selector)
                if features:
                    for feature in features:
                        text = feature.text.strip().lower()

                        if any(
                            term in text
                            for term in ["remote testing", "remote", "online testing"]
                        ):
                            assessment["remote_testing"] = (
                                "Yes"
                                if any(
                                    term in text
                                    for term in ["yes", "available", "supported"]
                                )
                                else "No"
                            )
                        if any(
                            term in text
                            for term in ["adaptive", "irt", "item response"]
                        ):
                            assessment["adaptive_irt"] = (
                                "Yes"
                                if any(
                                    term in text
                                    for term in ["yes", "available", "supported"]
                                )
                                else "No"
                            )
                        if any(
                            term in text for term in ["duration", "time", "minutes"]
                        ):
                            match = re.search(r"(\d+\s*(minutes|min|m|hours|hr))", text)
                            if match:
                                assessment["duration"] = match.group(1)
                        if any(
                            term in text
                            for term in ["type", "category", "assessment type"]
                        ):
                            match = re.search(r"(?:type|category):\s*([^,\.]+)", text)
                            if match:
                                assessment["test_type"] = match.group(1).strip()
                        else:
                            # Guess type if not explicitly found
                            if any(
                                term in text
                                for term in [
                                    "cognitive",
                                    "aptitude",
                                    "reasoning",
                                    "ability",
                                ]
                            ):
                                assessment["test_type"] = "Cognitive Ability"
                            elif any(
                                term in text
                                for term in ["personality", "behavior", "trait"]
                            ):
                                assessment["test_type"] = "Personality & Behavior"
                            elif any(
                                term in text
                                for term in ["coding", "technical", "programming"]
                            ):
                                assessment["test_type"] = "Coding & Technical"
                            elif any(
                                term in text
                                for term in ["situation", "judgment", "skills"]
                            ):
                                assessment["test_type"] = "Situational & Skills"

                    break  # Exit after first matching feature list

            if assessment["name"] and assessment["description"]:
                assessments.append(assessment)
            else:
                print("Skipping incomplete assessment:", assessment)

        # Save only if there’s something to save
        if assessments:
            dir_name = os.path.dirname(self.data_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(assessments, f, indent=2)

            print(
                f"Scraped {len(assessments)} assessments and saved to {self.data_path}"
            )
        else:
            print("No assessments found to save.")

        return assessments

    def load_data(self):
        """Load from file or scrape if file does not exist."""
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return self.scrape_catalog()
