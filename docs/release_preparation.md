# Release preparation

This is documentation-only portfolio preparation. No new benchmark, model selection,
final-test evaluation, remote metadata update, commit, tag or publication is performed.

## Provenance

[Release summary](../reports/release_summary.json) records evaluated system commit
`75c72d09e9af756a11888ef216afd4c2ee6a6f0d` and documentation base
`9926c29991dc1ec26072c681d8989f324b3f0837`. Evaluation also used the separately
hash-frozen, then-uncommitted harness recorded in the immutable
[RC manifest](../reports/phase9/release_candidate_manifest.json). Its bytes were
subsequently committed. The final documentation commit is pending authorization.
Do not substitute a later docs commit into historical evaluation evidence.

After approved documentation commits, record their final content commit in the
release summary in a small follow-up metadata commit. That value identifies the
completed documentation content, not the metadata commit itself; this avoids a
self-referential Git hash. Preserve the original documentation base and evaluated identity.

## Historical documentation

The README is the current portfolio overview. Existing phase reports and methodology
documents remain historical evidence. Their phase-relative future tense, uncommitted
status statements and local diagnostic paths describe the time of validation.
In particular, deployment documentation retains early statements that Phase 9 had
not started and Docker OS compatibility was not yet exercised; its later completed
container section and the immutable Docker report establish the successful validation.
These historical statements were not silently rewritten. The current README links
to the completed evidence and states actual provisioning prerequisites.

No public one-command dataset/model reconstruction is claimed. The frozen inference
bundle must be provisioned separately; downloading/rebuilding data is not part of the
serving quick start. Large runtime artifacts remain ignored. Do not upload them as
release attachments without a separate licensing and distribution decision.

## Recommended GitHub metadata

Description: Production-style product search with hybrid retrieval, CrossEncoder
reranking, FastAPI serving, measured benchmarks, observability and GPU Docker validation.

Topics: `machine-learning`, `information-retrieval`, `search`, `ranking`, `bm25`,
`sentence-transformers`, `cross-encoder`, `fastapi`, `mlops`, `docker`.

Recommendations only; remote metadata is unchanged.

## Proposed commits and approval sequence

1. `docs: present final search quality serving and deployment evidence` — README.
2. `docs: add evidence-backed resume and interview material` — the two portfolio documents.
3. `docs: record release provenance and portfolio validation` — this guide, release summary and final check report.

After explicit approval: inspect the final diff, rerun documentation/integrity checks,
commit those groups, then record the documentation content commit in a metadata
follow-up and confirm clean status. Verify the intended remote before pushing main.
Only after separate tag/release authorization, choose an unused version, create an
annotated tag at the reviewed release commit and publish release notes linking the
final evaluation, benchmark and Docker evidence. State evaluated and documentation
identities separately. Do not publish the Docker image as part of this plan.

The current version field is not a recommendation to reuse or create any existing tag.
No tag or GitHub release has been created.
