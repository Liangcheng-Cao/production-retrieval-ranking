# Canonical WANDS methodology and frozen split boundary

## Canonical contract and normalization

The sole supported processed interface is product_search.data.load_dataset(directory,
partition="train"). It verifies artifact hashes, schemas and integrity and returns
immutable dataclasses. JSONL uses explicit JSON types, sorted object keys, UTF-8,
LF lines and no nonfinite numbers. Products sort by numeric product_id, queries by
query_id, judgments/conflicts by (query_id, product_id). No Parquet dependency is needed.

Product has 9 fields: integer product_id; string product_name, product_class,
category_hierarchy, product_description, product_features; nullable integer
rating_count/review_count; nullable float average_rating. Query has integer query_id,
original query and query_class strings, plus normalized_query. Judgment has integer
query_id/product_id/relevance and string label. Conflict stores the pair, ordered
labels and original annotation IDs. The manifest records these schemas.

The standard-library CSV reader preserves text with quoting and embedded newlines,
uses tab separation despite .csv names, and checks exact headers and row lengths.
No pandas automatic NA conversion occurs. All raw IDs are canonical nonnegative
integer strings and must be unique in their primary tables. Product identities
are never merged by text.

Blank/whitespace or actual null/NaN text becomes an empty string; nonblank text is
preserved exactly, including surrounding whitespace and sentinel-looking strings.
Raw WANDS contains empty fields but no whitespace-only or sentinel-like missing
values in the inspected product/query columns. Actual nulls do not occur in the CSV
parser output. Do not assume a literal 'NA' is missing product text.

Numeric blank/null becomes JSON null. Decimal parsing rejects malformed, nonfinite,
negative and fractional count values. No imputation is performed. All 33,542
nonmissing values in each numeric column parse successfully; each has 9,452 missing.
These columns are not retrieval features yet.

## Judgment policy and measured audit

Raw annotations: 233,448; unique query-product pairs: 231,873. There are 1,453
same-label duplicate groups and 14 conflicting groups. Deduplicate by
(query_id, product_id, label), ignoring annotation ID: 1,561 redundant rows are
removed. The 14 conflicting pairs contain 30 raw rows or 28 rows after same-label
deduplication. Exclude all those pairs from supervised judgments, retaining their
labels and annotation IDs in judgment_conflicts.jsonl. Final canonical judgments:
231,859. Conflicts are 9 Partial/Exact and 5 Irrelevant/Partial; none Exact/Irrelevant
only or three-way. This avoids inventing ground truth via arbitrary ordering,
majority voting or optimistic/pessimistic relevance choice.

## Relevance and eligibility

Upstream annotation guidelines:
https://github.com/wayfair/WANDS/blob/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/Product%20Search%20Relevance%20Annotation%20Guidelines.pdf

Exact satisfies the full query; Partial matches its target entity while missing
requested modifiers; Irrelevant does not match. Our project maps these ordered
categories to 2/1/0, with NDCG gains 2**grade-1 (3/1/0). Recall positives are Partial
and Exact, measuring broad entity relevance. These numerical/metric choices are
project policy, not a claim that upstream mandates this evaluation convention.
Constants are in schema.py and copied into the manifest.

Query 366 has no positive labels even before conflict processing. Keep all 480
queries in the split and all their canonical judgments. For future Recall/NDCG,
exclude zero-positive queries from macro means (undefined denominator/ideal DCG),
report excluded counts and eligible query counts, and use identical eligibility
for paired comparisons. Do not silently score them as zero or choose another seed.
Unjudged pairs remain unknown, not supervised negatives. Metric cutoffs and any
operational zero-gain convention must be fixed before model selection; no metrics
are implemented here.

## Exact split algorithm

Normalization v1: Unicode NFKC, then casefold, then Python split()/single-space join.
Original query text remains unchanged for later retrieval. Group by normalized key;
all IDs in a group move together. This blocks identical normalized-text leakage,
not every semantic paraphrase overlap.

Algorithm sha256-json-seed-key-largest-remainder-v1, seed 42:
1. Serialize [seed, normalized_key] with json.dumps(ensure_ascii=False,
   sort_keys=True, separators=(",", ":"), allow_nan=False), append one LF, encode UTF-8.
2. Sort groups by SHA-256 hex digest, then normalized key for a hash tie.
3. Compute group quotas with exact rational fractions 3/5, 1/5, 1/5. Floor each,
   then allocate remaining groups by descending fractional remainder, ties in
   train/validation/test order. Take contiguous blocks in that same partition order.
4. Flatten whole groups, sort numeric query IDs in each partition. The membership
   checksum is SHA-256 of the same compact JSON+LF encoding of that sorted ID list.

All 480 normalized keys are unique: 288 train, 96 validation, 96 test. No overlap;
complete coverage. General quotas apply to groups, so query counts may differ from
ratios when groups have multiple IDs. Assignment is independent of input row order.

## Freeze, reconstruction and access discipline

Small data_manifest.json, query_splits.json and data_audit.json are Git-eligible;
all JSONL data (including the small conflict table) remains ignored. The manifest
records revision, raw and processed hashes, source hashes, policies, schemas, IDs,
partition checksums, creation UTC and Python version. No wall-clock time is embedded
in data rows. The first creation timestamp is retained on same-directory reruns.
For byte-identical independent reconstruction, supply the recorded --created-at
value and use the recorded runtime/source. Creation metadata may differ on a new
runtime; never silently replace an established manifest to hide that difference.

Build once with scripts/build_data.py. Frozen manifests may already exist after
checkout without ignored payloads: reconstruct into a separate empty output directory
using --output and the recorded --created-at. Compare its manifest and all recorded
hashes to the frozen manifest before restoring missing payloads. A normal build
refuses differing files or implicit repair of a partially missing frozen output.
There is no force-overwrite option. Preserve versioned provenance explicitly if
future authorized data changes require a new dataset version.

Judgments are physically separated by partition. Default loading returns train;
validation is explicit and test requires allow_test=True. It is a guardrail, not a
security boundary. Future development must not use final-test labels for model
selection, hyperparameters, negative mining, engineering decisions or failure-led
improvements. Engineering benchmarks/replay use train queries; validation supports
selection. All partitions share the full catalog, so this is known-catalog retrieval,
not unseen-product generalization.

Global labels were inspected only for Phase 0/1 structural validation, including
loading all partitions before completing Phase 1. This is recorded, not disguised
as untouched data. After Phase 1, do not routinely rerun raw audits/builds to inspect
test labels. Freeze model/pipeline settings and evaluation code before later final
test access; log future access. No model or final evaluation evidence exists yet.
