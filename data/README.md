# WANDS source and acquisition

Authoritative source: https://github.com/wayfair/WANDS

Audited commit: `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`.
The upstream README and LICENSE specify MIT, copyright (c) 2021 ecir2022.
The complete notice is preserved in WANDS-LICENSE.txt. Retain the copyright and
permission notice with redistributed copies/substantial portions; the license
disclaims warranty. Cite *WANDS: Dataset for Product Search Relevance Assessment*,
Chen et al., ECIR 2022, as requested upstream. Use upstream terms rather than
potentially inconsistent third-party dataset mirror metadata.

From repository root, run:

```powershell
.\.venv\Scripts\python.exe scripts/download_wands.py
```

This downloads three public files without authentication, from immutable revision
URLs recorded in reports/dataset_audit.json, into data/raw and verifies size and
SHA-256. Existing matching files are reused; mismatches fail without overwrite.
Downloaded files total **96,377,307 bytes** (96.38 MB; about 91.91 MiB).
Raw and processed data must not be committed. No model download is needed.

## Verified raw schema

Despite .csv extensions these are **UTF-8 tab-separated files**, with CSV quoting.
Use a real CSV parser to handle quoted text and embedded newlines. The audit does
not interpret string sentinel values as missing; blank counts mean empty or
whitespace-only fields. Future preprocessing must define numeric/null semantics.

| File | Rows | Columns |
|---|---:|---|
| product.csv | 42,994 | product_id, product_name, product_class, category hierarchy, product_description, product_features, rating_count, average_rating, review_count |
| query.csv | 480 | query_id, query, query_class |
| label.csv | 233,448 | id, query_id, product_id, label |

Product IDs, query IDs and annotation IDs are unique in their respective files,
nonblank and represented as integer strings. All annotation foreign keys resolve.
Labels: Exact (25,614), Partial (146,633), Irrelevant (61,201).

There are **1,575 excess repeated query-product rows**, including **14 distinct
query-product pairs with conflicting labels**. Annotation IDs being unique does
not imply query-product pairs are unique. No deduplication has been applied.

Blank product fields: product_class 2,852; category hierarchy 1,556;
product_description 6,008; rating_count, average_rating and review_count 9,452 each.
No blank product IDs, names or features. Six query_class values are blank; query
IDs/text and all annotation fields are nonblank. All queries have judgments;
eight catalog products have none. There are 417 excess duplicate product names
and 41 excess duplicate name-description combinations; do not merge product IDs
based on text alone. No normalized duplicate query text was found using NFKC,
case folding and whitespace collapse. No field-count or CSV parsing errors occurred.

Phase 1 now implements typed normalization, conservative conflict exclusion and
query-group splitting. See ../reports/split_protocol.md for the canonical schemas,
exact split algorithm, missing-value policy, zero-positive handling and freeze.
Raw files remain unchanged. Unjudged query-product pairs are not verified negatives.
