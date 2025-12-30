"""
Database operations and SQLite integration (replacing MongoDB).

This module provides a shim for MongoDB operations using SQLite for simpler local management.
"""

import os
import glob
import sqlite3
import json
from functools import lru_cache
from typing import Iterable, Any

from omegaconf import OmegaConf
from .data_analysis import get_nested_value, normalize_value_for_grouping

class SQLiteCollection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config TEXT UNIQUE,
                    log_file TEXT,
                    scored_by TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_file ON runs(log_file)")

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def find(self, filter: dict = None, projection: dict = None) -> list[dict]:
        with self._get_conn() as conn:
            if filter and "log_file" in filter:
                cursor = conn.execute("SELECT * FROM runs WHERE log_file = ?", (filter["log_file"],))
            elif filter and "config" in filter:
                config_str = json.dumps(filter["config"], sort_keys=True)
                cursor = conn.execute("SELECT * FROM runs WHERE config = ?", (config_str,))
            else:
                cursor = conn.execute("SELECT * FROM runs")
            
            results = []
            for row in cursor:
                doc = {
                    "config": json.loads(row["config"]),
                    "log_file": row["log_file"],
                    "scored_by": json.loads(row["scored_by"]) if row["scored_by"] else [],
                    "_id": row["id"]
                }
                results.append(doc)
            return results

    def find_one(self, filter: dict = None, projection: dict = None) -> dict | None:
        results = self.find(filter, projection)
        return results[0] if results else None

    def replace_one(self, filter_dict: dict, replacement: dict, upsert: bool = False):
        # We assume the filter is by config, as in log_config_to_db
        config_data = replacement["config"]
        config_str = json.dumps(config_data, sort_keys=True)
        log_file = replacement["log_file"]
        scored_by = json.dumps(replacement.get("scored_by", []))
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs (config, log_file, scored_by)
                VALUES (?, ?, ?)
            """, (config_str, log_file, scored_by))

    def update_many(self, filter_dict: dict, update_dict: dict):
        # Supports {"log_file": path} filter and {"$addToSet": {"scored_by": classifier}} update
        if "$addToSet" in update_dict and "scored_by" in update_dict["$addToSet"]:
            classifier = update_dict["$addToSet"]["scored_by"]
            log_file = filter_dict.get("log_file")
            
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT id, scored_by FROM runs WHERE log_file = ?", (log_file,))
                rows = cursor.fetchall()
                for row in rows:
                    scored_by = json.loads(row["scored_by"]) if row["scored_by"] else []
                    if classifier not in scored_by:
                        scored_by.append(classifier)
                        conn.execute("UPDATE runs SET scored_by = ? WHERE id = ?", 
                                     (json.dumps(scored_by), row["id"]))

    def delete_one(self, filter_dict: dict):
        if "_id" in filter_dict:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM runs WHERE id = ?", (filter_dict["_id"],))

class SQLiteDB:
    def __init__(self, db_path: str):
        self.runs = SQLiteCollection(db_path)

def get_mongodb_connection() -> Any:
    """Get a SQLite-based shim that mimics MongoDB connection."""
    db_path = os.environ.get("SQLITE_DB_PATH", "adversaria.db")
    return SQLiteDB(db_path)


def log_config_to_db(run_config, result, log_file):
    db = get_mongodb_connection()
    collection = db.runs

    idx = run_config.dataset_params.idx
    if idx is None:
        idx = [i for i in range(len(result.runs))]
    elif isinstance(idx, int):
        idx = [idx]

    for i in idx:
        run_config.dataset_params.idx = i
        config_data = {
            "config": OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True),
            "log_file": log_file,
            "scored_by": []
        }
        # If a run with the same config already exists, replace it
        collection.replace_one(
            {"config": config_data["config"]},
            config_data,
            upsert=True
        )


def delete_orphaned_runs(dry_run: bool = True, direction: str = "db_only"):
    db = get_mongodb_connection()

    if direction in ["db_only", "both"]:
        # remove DB entries for missing files
        items = db.runs.find()
        for item in items:
            log_file = item["log_file"]
            if not os.path.exists(log_file):
                print(f"Log file not found: {log_file}, deleting from database")
                if not dry_run:
                    db.runs.delete_one({"_id": item["_id"]})

    if direction in ["files_only", "both"]:
        tracked_files = set()
        items = db.runs.find()  # Projection not implemented in shim, so we get all
        for item in items:
            tracked_files.add(item["log_file"])

        all_run_files = set(glob.glob("outputs/**/run.json", recursive=True))
        all_run_files_absolute = set(os.path.abspath(f) for f in all_run_files)

        # Find untracked files
        untracked_files = all_run_files_absolute - tracked_files
        print(f"Found {len(untracked_files)} untracked files")

        for untracked_file in sorted(list(untracked_files)):
            print(f"Untracked run file found: {untracked_file}")
            if not dry_run:
                print(f"Deleting untracked file: {untracked_file}")
                os.remove(untracked_file)


def check_match(doc_fragment, filter_fragment):
    # --- 1. dict → recurse over its keys --------------------------------------
    if isinstance(filter_fragment, dict):
        if not isinstance(doc_fragment, dict):
            return False
        for k, v in filter_fragment.items():
            if k not in doc_fragment or not check_match(doc_fragment[k], v):
                return False
        return True

    # --- 2. iterable → "any of these values is fine" --------------------------
    if isinstance(filter_fragment, (list, tuple, set)):
        if isinstance(doc_fragment, (list, tuple, set)):
            return filter_fragment == doc_fragment
        return doc_fragment in filter_fragment

    # --- 3. primitive equality ------------------------------------------------
    return doc_fragment == filter_fragment


@lru_cache
def get_all_runs() -> list[dict]:
    db = get_mongodb_connection()
    collection = db.runs
    return list(collection.find())


def get_filtered_and_grouped_paths(filter_by: dict, group_by: Iterable[str]|None = None, force_reload: bool = True) -> dict[tuple[str], list[str]]:
    if force_reload:
        get_all_runs.cache_clear()
    all_results = get_all_runs()

    if filter_by:
        filtered_results = [
            doc for doc in all_results
            if check_match(doc['config'], filter_by)
        ]
    else:
        filtered_results = all_results

    if not group_by:
        return {("all",): [r["log_file"] for r in filtered_results if "log_file" in r]}

    grouped_results = {}
    for result in filtered_results:
        if "config" not in result or "log_file" not in result:
            continue

        config_data = result["config"]
        log_path = result["log_file"]

        group_key_parts = []
        for key_spec in group_by:
            if isinstance(key_spec, str):
                value = get_nested_value(config_data, [key_spec])
                normalized_value = normalize_value_for_grouping(value)
                group_key_parts.append(f"{key_spec}={normalized_value}")
            elif isinstance(key_spec, (list, tuple)):
                value = get_nested_value(config_data, key_spec)
                normalized_value = normalize_value_for_grouping(value)
                key_name = '.'.join(map(str, key_spec))
                group_key_parts.append(f"{key_name}={normalized_value}")
            else:
                group_key_parts.append(f"invalid_group_spec={key_spec}")

        group_key_tuple = tuple(sorted(group_key_parts))

        if group_key_tuple not in grouped_results:
            grouped_results[group_key_tuple] = []
        grouped_results[group_key_tuple].append(log_path)
    return grouped_results
