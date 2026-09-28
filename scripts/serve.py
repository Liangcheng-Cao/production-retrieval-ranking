"""Local single-worker runner; no reload, benchmark or deployment automation."""
import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(os.environ.get('SEARCH_RUNTIME_CONFIG', ROOT/'configs/runtime.json')))
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import uvicorn
    from product_search.api import create_app
    uvicorn.run(create_app(args.config), host='127.0.0.1', port=args.port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
