"""Strict readiness: DEGRADED remains searchable but container should be unhealthy."""
import json
import sys
from urllib.request import urlopen


def main():
    try:
        with urlopen('http://127.0.0.1:8000/ready', timeout=3) as response:
            state = json.load(response)
            return 0 if response.status == 200 and state.get('ready') is True and state.get('state') == 'READY' else 1
    except Exception:
        return 1


if __name__ == '__main__':
    sys.exit(main())
