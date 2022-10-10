import argparse
import csv
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import requests

BUILD_DIR = "build"
BASE_URL = "https://scholar.google.com/scholar"

SEARCH_TERMS = {
    "offline-rl": [
        "offline reinforcement learning",
        "batch reinforcement learning",
        "offline RL",
        "batch RL",
    ],
    "rl": [
        "reinforcement learning",
    ],
}

PATTERN = re.compile("About ([0-9,]*) results")


def get_query_string(search_terms: List[str]) -> str:
    return " OR ".join(map(lambda term: f'"{term}"', search_terms))


def get_num_articles(year: int, search_terms: List[str]) -> int:
    response = requests.get(
        BASE_URL,
        params={
            "hl": "en",
            "as_ylo": year,
            "as_yhi": year,
            "q": get_query_string(search_terms),
        },
    )
    assert (
        response.status_code == 200
    ), f"Error in request: status {response.status_code}"
    match = re.search(PATTERN, response.text)
    assert (
        match is not None
    ), f"Error processing Google Scholar query for url {response.url}."
    num_articles_str = match.groups()[0]
    return int(num_articles_str.replace(",", ""))


def plot_num_articles(
    years: List[int], num_articles_per_year: Dict[str, List[int]]
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()
    for (category, num_articles), ax, color in zip(
        num_articles_per_year.items(), (ax1, ax2), ("blue", "red")
    ):
        ax.plot(years, num_articles, color=color, label=category)
    ax1.legend()
    ax2.legend()
    fig.savefig(os.path.join(BUILD_DIR, "num-articles-per-year.pdf"))


def save_num_articles(
    save_path: str, years: List[int], num_articles_per_year: Dict[str, List[int]]
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf8") as csv_file:
        field_names = ["year"] + list(num_articles_per_year.keys())
        writer = csv.DictWriter(csv_file, fieldnames=field_names)

        writer.writeheader()
        for idx, year in enumerate(years):
            entry = {"year": year}
            for category, num_articles in num_articles_per_year.items():
                entry[category] = num_articles[idx]
            writer.writerow(entry)


def main(save_path: Optional[str], render: bool) -> None:
    years = list(range(2011, 2022))

    num_articles_per_year: Dict[str, List[int]] = {}
    for category, search_terms in SEARCH_TERMS.items():
        num_articles_per_year[category] = [
            get_num_articles(year, search_terms) for year in years
        ]

    if render:
        plot_num_articles(years, num_articles_per_year)

    if save_path is not None:
        save_num_articles(save_path, years, num_articles_per_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script crawls Google Scholar for information on the "
        "number of publications in RL in the past decade."
    )
    parser.add_argument(
        "--save_path",
        help="Path to output csv file with crawled data.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the plot uisng matplotlib.",
    )
    args = parser.parse_args()
    main(**vars(args))
