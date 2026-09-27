"""Versioned canonical row contracts; no dataframe or inference dependencies."""
from dataclasses import dataclass

SCHEMA_VERSION = "wands-canonical-v1"
NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"
CONFLICT_POLICY = "exclude-conflicting-pairs-v1"
RELEVANCE = {"Irrelevant": 0, "Partial": 1, "Exact": 2}
POSITIVE_LABELS = ("Partial", "Exact")
NDCG_GAIN = "2**relevance - 1"
RAW_COLUMNS = {
    "product.csv": ("product_id", "product_name", "product_class", "category hierarchy", "product_description", "product_features", "rating_count", "average_rating", "review_count"),
    "query.csv": ("query_id", "query", "query_class"),
    "label.csv": ("id", "query_id", "product_id", "label"),
}

@dataclass(frozen=True)
class Product:
    product_id: int
    product_name: str
    product_class: str
    category_hierarchy: str
    product_description: str
    product_features: str
    rating_count: int | None
    average_rating: float | None
    review_count: int | None

@dataclass(frozen=True)
class Query:
    query_id: int
    query: str
    query_class: str
    normalized_query: str

@dataclass(frozen=True)
class Judgment:
    query_id: int
    product_id: int
    label: str
    relevance: int

@dataclass(frozen=True)
class Conflict:
    query_id: int
    product_id: int
    labels: tuple[str, ...]
    annotation_ids: tuple[int, ...]

@dataclass(frozen=True)
class Dataset:
    products: tuple[Product, ...]
    queries: tuple[Query, ...]
    judgments: tuple[Judgment, ...]
    conflicts: tuple[Conflict, ...]
