"""
One-off diagnostic: given a doc_id from a failed smoke-test run, check
every plausible database/collection combo on the SAME Atlas cluster to
see where the audit entry actually landed. Run this before assuming the
write is failing -- it may just be landing in a different db than the
test is reading from.

Usage:
  $env:AUDIT_MONGO_URI = "mongodb+srv://..."
  python check_audit_doc_location.py edc60995-a35f-43af-9b79-5c0f375d80b0
"""
import asyncio
import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient
import certifi

CANDIDATE_DBS = ["loan_kyc_audit", "golden_set_audit"]
CANDIDATE_COLLECTIONS = ["audit_log"]


async def main(doc_id: str):
    uri = os.environ["AUDIT_MONGO_URI"]
    client = AsyncIOMotorClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=10_000)

    found_anywhere = False
    for db_name in CANDIDATE_DBS:
        db = client[db_name]
        existing_collections = await db.list_collection_names()
        for coll_name in CANDIDATE_COLLECTIONS:
            coll = db[coll_name]
            doc = await coll.find_one({"doc_id": doc_id})
            status = "FOUND" if doc else "not found"
            print(f"{db_name}.{coll_name}: {status}")
            if doc:
                found_anywhere = True

        # Also report every collection actually present in this db, in
        # case the real collection name differs from "audit_log" too.
        other = [c for c in existing_collections if c not in CANDIDATE_COLLECTIONS]
        if other:
            print(f"  (other collections present in {db_name}: {other})")

    if not found_anywhere:
        print(
            "\nNot found in any candidate db/collection on this cluster. "
            "Either the write is genuinely failing (check ECS task logs for "
            "'[audit_log]' entries), or AUDIT_MONGO_URI here points at a "
            "different cluster than the deployed ECS task uses."
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))