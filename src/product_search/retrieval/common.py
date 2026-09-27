"""Small shared search contract and deterministic ordering."""
from dataclasses import dataclass
from typing import Protocol
import re
import unicodedata
import numpy as np

TEXT_VERSION = "product-fields-v1"
LEXICAL_VERSION = "nfkc-casefold-unicode-alnum-v1"

@dataclass(frozen=True)
class Hit:
    product_id: int
    score: float
    rank: int
    source: str

class Retriever(Protocol):
    def search(self, query: str, top_k: int = 100) -> list[Hit]: ...

def check_request(query, top_k):
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

def tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())

def product_text(product, representation):
    values = [product.product_name]
    if representation in ("B", "C"):
        # Avoid repeating exactly identical class/hierarchy strings.
        for value in (product.product_class, product.category_hierarchy):
            if value.strip() and value not in values:
                values.append(value)
    if representation == "C":
        values.extend((product.product_description, product.product_features))
    if representation not in ("A", "B", "C"):
        raise ValueError("Unknown product representation")
    return "\n".join(v for v in values if v.strip())

def ids_array(ids):
    array = np.asarray(ids)
    if array.ndim != 1 or len(array) == 0 or array.dtype.kind not in "iu" or len(np.unique(array)) != len(array) or np.any(array < 0):
        raise ValueError("Unique nonnegative integer product IDs required")
    return array.astype(np.int64)

def ranked(ids, scores, top_k, source, positive_only=False):
    scores = np.asarray(scores)
    if scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid retrieval scores")
    eligible = np.flatnonzero(scores > 0) if positive_only else np.arange(len(ids))
    order = eligible[np.lexsort((ids[eligible], -scores[eligible]))[:top_k]]
    return [Hit(int(ids[i]), float(scores[i]), rank, source) for rank, i in enumerate(order, 1)]
