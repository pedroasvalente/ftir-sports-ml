#!/usr/bin/env python3
"""
Cleanup DagsHub MLflow runs by config tag.

Usage:
    python scripts/cleanup_dagshub_runs.py --config study1_quick_test.json [--dry-run]
    python scripts/cleanup_dagshub_runs.py --config study1_group_fam.json  [--dry-run]
    python scripts/cleanup_dagshub_runs.py --list-configs                   # see what exists

Uses DagsHub REST API (Basic auth) — no mlflow Python client needed.

Set DAGSHUB_USER_TOKEN in your environment before running:
    export DAGSHUB_USER_TOKEN=<your_token>
"""

import argparse
import os
import sys

import requests

DAGSHUB_BASE = "https://dagshub.com/pedroasvalente/ftir-sports-ml.mlflow/api/2.0/mlflow"
USERNAME = "pedroasvalente"


def _auth():
    token = os.environ.get("DAGSHUB_USER_TOKEN")
    if not token:
        sys.exit("❌  Set DAGSHUB_USER_TOKEN environment variable first.")
    return (USERNAME, token)


def _get_experiments(auth) -> list[dict]:
    r = requests.get(f"{DAGSHUB_BASE}/experiments/search",
                     auth=auth, params={"max_results": 1000})
    r.raise_for_status()
    return r.json().get("experiments", [])


def _search_runs(exp_id: str, auth, filter_str: str = "") -> list[dict]:
    """Return all runs (with pagination) for a given experiment."""
    runs, page_token = [], None
    while True:
        body = {"experiment_ids": [exp_id], "max_results": 1000}
        if filter_str:
            body["filter"] = filter_str
        if page_token:
            body["page_token"] = page_token
        r = requests.post(f"{DAGSHUB_BASE}/runs/search", auth=auth,
                          headers={"Content-Type": "application/json"}, json=body)
        r.raise_for_status()
        data = r.json()
        runs.extend(data.get("runs", []))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return runs


def _delete_run(run_id: str, auth) -> bool:
    r = requests.post(f"{DAGSHUB_BASE}/runs/delete", auth=auth,
                      headers={"Content-Type": "application/json"},
                      json={"run_id": run_id})
    return r.status_code == 200


def _get_tag(run: dict, key: str) -> str:
    return next((t["value"] for t in run.get("data", {}).get("tags", [])
                 if t["key"] == key), "")


def list_configs(auth):
    """Print all config values found across all experiments."""
    experiments = _get_experiments(auth)
    configs: dict[str, int] = {}
    for exp in experiments:
        runs = _search_runs(exp["experiment_id"], auth)
        for run in runs:
            cfg = _get_tag(run, "config")
            if cfg:
                configs[cfg] = configs.get(cfg, 0) + 1
    print("\n📋  Config values found on DagsHub:\n")
    for cfg, count in sorted(configs.items()):
        print(f"   {cfg:45s}  →  {count} run(s)")
    print()


def delete_by_config(config_name: str, auth, dry_run: bool = True):
    """Delete all runs tagged with a given config name."""
    experiments = _get_experiments(auth)
    total_found = 0
    total_deleted = 0
    total_failed = 0

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Searching runs with config = '{config_name}' …\n")

    for exp in experiments:
        runs = _search_runs(exp["experiment_id"], auth)
        matched = [r for r in runs if _get_tag(r, "config") == config_name]
        if not matched:
            continue

        print(f"  Experiment: {exp['name']}  ({len(matched)} run(s) to delete)")
        for run in matched:
            total_found += 1
            run_id = run["info"]["run_id"]
            model = _get_tag(run, "model") or "parent"
            tp    = _get_tag(run, "timepoints")
            mat   = _get_tag(run, "sample_type")
            print(f"    {'[skip] ' if dry_run else '[delete] '}"
                  f"{run_id[:8]}…  {mat:8s} {tp:12s} {model}")

            if not dry_run:
                ok = _delete_run(run_id, auth)
                if ok:
                    total_deleted += 1
                else:
                    total_failed += 1
                    print(f"             ⚠️  failed to delete {run_id}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Runs found:   {total_found}")
    if not dry_run:
        print(f"  Deleted:      {total_deleted}")
        if total_failed:
            print(f"  Failed:       {total_failed}  ← check token permissions")
    else:
        print("  (no changes made — re-run without --dry-run to delete)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Cleanup DagsHub MLflow runs by config tag")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Config filename to target (e.g. study1_quick_test.json)")
    group.add_argument("--list-configs", action="store_true", help="List all config values on DagsHub")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview without deleting (default: True)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete runs (overrides --dry-run)")
    args = parser.parse_args()

    auth = _auth()

    if args.list_configs:
        list_configs(auth)
    else:
        dry_run = not args.execute
        delete_by_config(args.config, auth, dry_run=dry_run)


if __name__ == "__main__":
    main()
