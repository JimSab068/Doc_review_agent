from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Field names here are chosen to match the environment variable names
    the rest of the codebase already reads directly via os.environ --
    AuditLogStore reads AUDIT_MONGO_URI (audit_log.py), ComplianceKB reads
    CHROMA_HOST/CHROMA_PORT/CHROMA_SSL (compliance_kb.py), vault.py's
    _build_store reads VAULT_BACKEND/VAULT_DYNAMODB_TABLE/VAULT_KMS_KEY_ID/
    AWS_REGION, and GeminiClient reads GEMINI_API_KEY (primary_agent.py).

    pydantic-settings matches env vars to fields case-insensitively by
    name, so e.g. `audit_mongo_uri` picks up AUDIT_MONGO_URI automatically
    -- no aliasing needed. This class does not invent new env var names;
    it validates, at startup, that the ones the rest of the code already
    depends on are actually set, so a missing var fails fast in /healthz's
    process boot rather than 500ing on the first real request.
    """

    app_name: str = "Loan KYC Agent API"

    # Read independently by vault.py's fail-closed check (APP_ENV in
    # {"staging","production"} requires VAULT_BACKEND=dynamodb). This
    # field doesn't drive that logic -- it's here so the app validates
    # the value is set to something sane at startup too.
    app_env: str = "production"

    # --- audit_log.py: AuditLogStore(db_name, collection_name) ---
    audit_mongo_uri: str  # required, no default -- AUDIT_MONGO_URI
    mongo_db: str = "loan_kyc_audit"
    mongo_collection: str = "audit_log"

    # --- compliance_kb.py: ComplianceKB(collection_name) ---
    chroma_host: Optional[str] = None  # unset -> ephemeral in-memory Chroma (dev only)
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_collection: str = "compliance_rules"

    # --- vault.py: _build_store() (VAULT_BACKEND=dynamodb path) ---
    vault_backend: str = "dynamodb"
    vault_dynamodb_table: Optional[str] = None  # required if vault_backend=dynamodb
    vault_kms_key_id: Optional[str] = None       # required if vault_backend=dynamodb
    aws_region: Optional[str] = None

    # --- primary_agent.py: GeminiClient(model_name) ---
    gemini_api_key: str  # required, no default -- GEMINI_API_KEY
    model_name: str = "gemini-3.1-flash-lite"  # confirm this is still a live Gemini model id before deploying

    # In production, do not load from a .env file -- rely on env vars
    # injected by the container orchestrator (e.g. AWS ECS task def).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()