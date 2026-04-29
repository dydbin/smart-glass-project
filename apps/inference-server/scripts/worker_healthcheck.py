import json
import sys

from src.health.checks import build_worker_health_payload


def main() -> int:
    status_code, payload = build_worker_health_payload()
    if status_code != 0:
        print(json.dumps(payload, ensure_ascii=True), file=sys.stderr)
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
