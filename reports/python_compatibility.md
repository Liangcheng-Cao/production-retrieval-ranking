# Python compatibility decision

**KEEP PYTHON 3.14**, checked 2026-09-27. Phase 1 tests/builds run on Python 3.14.3.
No retrieval/transformer dependencies were installed and no model was downloaded.

Primary evidence:

- https://pytorch.org/get-started/locally/ explicitly lists Windows Python 3.10-3.14.
- PyPI torch 2.14.0 publishes torch-2.14.0-cp314-cp314-win_amd64.whl.
- transformers 5.17.0 and sentence-transformers 6.1.0 declare Python >=3.10
  and publish universal Python wheels. https://huggingface.co/docs/transformers/installation
  and https://www.sbert.net/docs/installation.html describe their supported stack.
- tokenizers 0.23.2 and safetensors 0.8.0 publish Windows cp310-abi3 wheels usable
  by regular CPython 3.14. Candidate wheels and dependency metadata are recorded
  in python_compatibility.json; these are observed versions, not newly pinned dependencies.
- CrossEncoder is part of sentence-transformers, not a separate required runtime.

Local hardware query detected NVIDIA GeForce RTX 4070, driver 581.08. Hardware
presence is not a verified CUDA runtime. Phase 2 must select a supported PyTorch
compute build using the official selector, check driver compatibility, and test
CPU imports plus torch.cuda.is_available() and a small tensor operation if using
CUDA. Do not infer CUDA compatibility from Python wheel availability.

This audit establishes plausible installation support, not complete transitive
resolution, import success, model compatibility or inference performance. There
is no evidence here requiring migration to Python 3.12. If a later selected model
or dependency has a demonstrated 3.14 blocker, create a parallel .venv312, install
the same project, run the full test/build-verification procedure, and preserve
the original environment and frozen artifacts. Do not overwrite the manifest to
hide environment differences; its creation runtime is intentionally recorded.
