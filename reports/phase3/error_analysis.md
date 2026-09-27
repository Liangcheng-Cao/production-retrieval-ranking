# Validation error analysis (descriptive, no retuning)

Six queries were selected by largest positive/negative paired NDCG@10 change.
The small title/label excerpts in error_examples.json support these observations;
they are examples, not an estimate of failure-mode prevalence.

| Query ID | NDCG@10 delta | Evidence-backed observation |
|---|---:|---|
| 138 | +0.447643 | For bathroom glass doors, four Exact shower-door products replace French/closet doors at the top. Product-category disambiguation improves. |
| 442 | +0.409188 | For an above-toilet cabinet, Exact over-toilet cabinets move above Irrelevant tissue holders. The compound product intent is better matched. |
| 14 | +0.387452 | For beds with LEDs, Exact LED-related products move above Partial generic beds. The requested modifier is promoted. |
| 219 | -0.500168 | Four judged Partial towel racks move from ranks 1–4 to 11–14. The new top five Trinsic products are unjudged. This is an observed metric regression under incomplete qrels, not proof that the promoted products are irrelevant. |
| 441 | -0.263588 | Unjudged desk-and-chair sets occupy ranks 1–4, while judged Exact desks move 2→7, 3→19 and 4→15. Again, incomplete judgments limit relevance conclusions; title length is not established as the cause. |
| 406 | -0.202469 | A judged Exact refrigerator moves 5→9 while a judged Partial refrigerator installation accessory moves 9→4. This is a supported graded-relevance ordering regression, plausibly related to accessory/entity or modifier matching; the titles alone do not establish the causal mechanism. |

No new labels were assigned and no model or configuration was changed in response.
The first two regressions illustrate why unjudged does not mean known irrelevant.
The six examples do not support a general claim that long text causes failure:
this selected model receives titles only and none of its measured pairs were truncated.
