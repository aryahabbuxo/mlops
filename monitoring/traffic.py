"""Send demo traffic to the forecast API so the Grafana panels have data.

    python monitoring/traffic.py                      # 200 requests to the local stack
    python monitoring/traffic.py --base https://greenops-api-latest.onrender.com --n 50

Mostly valid forecasts across every site and many dates, plus a few what-if requests, unknown
sites and out-of-range dates so the error-rate panel moves too. Standard library only.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import date, timedelta

SNAPSHOT_START = date(2019, 10, 30)
SNAPSHOT_END = date(2026, 9, 29)
OUT_OF_RANGE = date(2018, 1, 1)


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict | list]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{base}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def random_day(rng: random.Random) -> date:
    span = (SNAPSHOT_END - SNAPSHOT_START).days
    return SNAPSHOT_START + timedelta(days=rng.randrange(span + 1))


def make_request(rng: random.Random, sites: list[str]) -> dict:
    roll = rng.random()
    if roll < 0.03:
        return {"site": "atlantis", "as_of": random_day(rng).isoformat()}
    if roll < 0.06:
        return {"site": rng.choice(sites), "as_of": OUT_OF_RANGE.isoformat()}
    body = {"site": rng.choice(sites), "as_of": random_day(rng).isoformat()}
    if roll < 0.20:
        body["perturb"] = {
            "sun_scale": round(rng.uniform(0.4, 1.2), 2),
            "wind_shift_ms": round(rng.uniform(-3.0, 3.0), 1),
        }
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    status, sites_body = call(base, "GET", "/v1/sites")
    if status != 200:
        print(f"{base}/v1/sites answered {status}; is the API up and ready?", file=sys.stderr)
        return 1
    sites = [site["name"] for site in sites_body]
    rng = random.Random(args.seed)

    statuses: Counter[int] = Counter()
    for i in range(1, args.n + 1):
        status, _ = call(base, "POST", "/v1/forecast", make_request(rng, sites))
        statuses[status] += 1
        if i % 50 == 0:
            print(f"{i}/{args.n} sent")
        time.sleep(args.delay)

    print("status counts:", dict(sorted(statuses.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
