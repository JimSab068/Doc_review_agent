# seed_kb.py
import os
import sys


def check_env():
    missing = []

    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY) -- required for embedding calls")

    if "CHROMA_HOST" not in os.environ:
        missing.append("CHROMA_HOST -- is the SSH tunnel open? (should be 'localhost' if tunneling)")

    if missing:
        print("Missing required environment variables:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    chroma_port = os.environ.get("CHROMA_PORT", "8000")
    chroma_ssl = os.environ.get("CHROMA_SSL", "false")
    print(f"Chroma target: {os.environ['CHROMA_HOST']}:{chroma_port} (ssl={chroma_ssl})")


def main():
    check_env()

    from src.compliance_kb import ComplianceKB
    from src.compliance_fixtures import REG_B_FCRA_FIXTURES

    kb = ComplianceKB(collection_name="compliance_rules")
    kb.seed_initial_compliance_data(REG_B_FCRA_FIXTURES)

    count = kb.collection.count()
    print(f"Total documents in collection: {count}")

    results = kb.collection.query(
        query_texts=["adverse action notice requirements"],
        n_results=2,
    )
    print(results["metadatas"])


if __name__ == "__main__":
    main()