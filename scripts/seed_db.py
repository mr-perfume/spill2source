"""
seed_db.py
----------------------------------------------------------------------------
Creates the MongoDB database and loads the two static collections the demo
needs. Safe to re-run: it replaces the catalog and vessel documents in place
and leaves any spill_events from previous runs alone unless you ask for them
to go.

MongoDB creates databases and collections lazily, on first write, so there is
no "CREATE DATABASE" step to run first. Starting mongod and running this
script is the whole setup.

    python scripts/seed_db.py                 load catalog + ships
    python scripts/seed_db.py --reset-events  also clear previous pipeline runs
    python scripts/seed_db.py --drop          wipe the database and reload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.errors import ServerSelectionTimeoutError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "oilspill")

CATALOG_PATH = PROJECT_ROOT / "data" / "demo_images" / "catalog.json"
SHIPS_PATH = PROJECT_ROOT / "data" / "ships" / "ships_seed.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-events", action="store_true", help="Clear the spill_events collection")
    ap.add_argument("--drop", action="store_true", help="Drop the whole database first")
    args = ap.parse_args()

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        print(f"Cannot reach MongoDB at {MONGODB_URI}.")
        print("Start it first. On Windows, if you installed MongoDB as a service:")
        print("    net start MongoDB")
        print("or run it in the foreground:")
        print('    "C:\\Program Files\\MongoDB\\Server\\8.0\\bin\\mongod.exe" --dbpath C:\\data\\db')
        sys.exit(1)

    if args.drop:
        client.drop_database(MONGODB_DB)
        print(f"[drop] database {MONGODB_DB} removed")

    db = client[MONGODB_DB]

    # --- demo_images ------------------------------------------------------
    if not CATALOG_PATH.exists():
        print(f"Missing {CATALOG_PATH}. Build it first:")
        print("    python scripts/make_demo_catalog.py")
        sys.exit(1)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    db.demo_images.delete_many({})
    if catalog:
        db.demo_images.insert_many(catalog)
    print(f"[demo_images] {len(catalog)} scene(s) loaded")
    for c in catalog:
        tile = PROJECT_ROOT / c["file_path"]
        mark = "ok " if tile.exists() else "MISSING"
        print(f"  {mark} {c['_id']}  {c['title']:<20} {c['acquisition_timestamp']}")

    # --- ships ------------------------------------------------------------
    if not SHIPS_PATH.exists():
        print(f"Missing {SHIPS_PATH}")
        sys.exit(1)
    ships = json.loads(SHIPS_PATH.read_text(encoding="utf-8"))
    db.ships.delete_many({})
    if ships:
        db.ships.insert_many(ships)
    db.ships.create_index([("ship_id", ASCENDING)], unique=True)
    print(f"[ships] {len(ships)} vessel(s) loaded")

    # A 2dsphere index on the track geometry. The AIS scorer walks paths in
    # Python rather than querying by geometry -- at seven vessels that is
    # faster than a round trip -- but the index is here so a larger dataset
    # can switch to a $near pre-filter without a schema change.
    for ship in ships:
        coords = [[p["lon"], p["lat"]] for p in ship.get("path", [])]
        if len(coords) >= 2:
            db.ships.update_one(
                {"ship_id": ship["ship_id"]},
                {"$set": {"track_geometry": {"type": "LineString", "coordinates": coords}}},
            )
    db.ships.create_index([("track_geometry", "2dsphere")])
    print("[ships] 2dsphere index built on track_geometry")

    # --- spill_events -----------------------------------------------------
    if args.reset_events:
        n = db.spill_events.count_documents({})
        db.spill_events.delete_many({})
        print(f"[spill_events] cleared {n} previous run(s)")
    else:
        db.spill_events.create_index([("created_at", ASCENDING)])
        print(f"[spill_events] {db.spill_events.count_documents({})} previous run(s) kept")

    print(f"\nDatabase '{MONGODB_DB}' ready at {MONGODB_URI}")
    print("Collections:", sorted(db.list_collection_names()))


if __name__ == "__main__":
    main()
