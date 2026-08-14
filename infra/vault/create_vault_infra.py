"""
One-time provisioning script for the persistent vault backend.

Creates:
  - A customer-managed KMS key (symmetric, for Encrypt/Decrypt) with an
    alias.
  - A DynamoDB table 'loan-kyc-token-maps' with partition key
    'token_map_id' and TTL enabled on 'expires_at'.
  - A filled-in IAM policy JSON (no placeholders) scoped to exactly that
    table and key, written to vault_iam_policy.generated.json.

Recommended location in your repo: infra/vault/create_vault_infra.py, so
this provisioning script and its generated policy are reviewed alongside
the application code that depends on them, rather than living loose at
the project root.

Run this once per environment (e.g. once for staging, once for prod) --
it is NOT part of the application's runtime path. Requires AWS credentials
with permission to create KMS keys, aliases, DynamoDB tables, and to call
sts:GetCallerIdentity.

Usage:
    python create_vault_infra.py --region us-east-1

After running:
  1. Take the printed KMS key ARN and set it as VAULT_KMS_KEY_ID in your
     ECS task definition / .env, alongside VAULT_DYNAMODB_TABLE and
     VAULT_BACKEND=dynamodb.
  2. Attach vault_iam_policy.generated.json to your ECS task role (as an
     inline policy, or create it as a standalone managed policy and
     attach it). It has real account/region/resource values baked in --
     don't hand-edit placeholders into a static policy file, since that's
     an easy way to end up with a stale or overly broad policy attached
     in production. If you use Terraform/CDK/CloudFormation for the task
     role, prefer generating the policy there directly using the same
     table/key ARNs printed below, and treat this file as a reference.
"""

from __future__ import annotations

import argparse
import json

import boto3


from pathlib import Path

GENERATED_POLICY_PATH = Path(__file__).with_name(
    "vault_iam_policy.generated.json"
)



KMS_KEY_POLICY_NOTE = """
Note: this script creates the key with the default policy (account root has
full access via IAM). The generated IAM policy below is what should
actually be attached to your ECS task role -- don't grant kms:* broadly.
"""

def _resource_names(environment: str) -> tuple[str, str]:
    table_name = f"loan-kyc-token-maps-{environment}"
    key_alias = f"alias/loan-kyc-vault-{environment}"
    return table_name, key_alias

def create_kms_key(kms_client, key_alias: str) -> str:
    existing = kms_client.list_aliases()
    for alias in existing.get("Aliases", []):
        if alias["AliasName"] == key_alias:
            print(f"KMS alias {key_alias} already exists -> {alias['TargetKeyId']}")
            key = kms_client.describe_key(KeyId=alias["TargetKeyId"])
            return key["KeyMetadata"]["Arn"]

    resp = kms_client.create_key(
        Description="loan-kyc-agent vault encryption key (PII token map values)",
        KeyUsage="ENCRYPT_DECRYPT",
        Origin="AWS_KMS",
        Tags=[{"TagKey": "project", "TagValue": "loan-kyc-agent"}],
    )
    key_id = resp["KeyMetadata"]["KeyId"]
    key_arn = resp["KeyMetadata"]["Arn"]

    kms_client.create_alias(AliasName=key_alias, TargetKeyId=key_id)
    print(f"Created KMS key {key_arn} with alias {key_alias}")
    print(KMS_KEY_POLICY_NOTE)
    return key_arn


def create_table(dynamodb_client, table_name: str) -> str:
    existing_tables = dynamodb_client.list_tables().get("TableNames", [])
    if table_name in existing_tables:
        print(f"Table {table_name} already exists")
    else:
        dynamodb_client.create_table(
            TableName=table_name,
            AttributeDefinitions=[{"AttributeName": "token_map_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "token_map_id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
            Tags=[{"Key": "project", "Value": "loan-kyc-agent"}],
        )
        print(f"Creating table {table_name}, waiting for it to become active...")
        waiter = dynamodb_client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)

        dynamodb_client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
        )
        print(f"Table {table_name} is active with TTL enabled on 'expires_at'")

    table_desc = dynamodb_client.describe_table(TableName=table_name)
    return table_desc["Table"]["TableArn"]


def write_iam_policy(table_arn: str, key_arn: str) -> None:
    """Writes a fully filled-in least-privilege policy -- no REGION /
    ACCOUNT_ID / KEY_ID placeholders left for someone to forget to
    replace. Only the actions _DynamoDBStore actually uses are granted:
    GetItem, UpdateItem, DeleteItem. PutItem is intentionally omitted --
    put() uses UpdateItem so the item and its nested 'tokens' map can be
    created or extended in the same call."""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VaultDynamoDBAccess",
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem"],
                "Resource": table_arn,
            },
            {
                "Sid": "VaultKMSAccess",
                "Effect": "Allow",
                "Action": ["kms:Encrypt", "kms:Decrypt"],
                "Resource": key_arn,
            },
        ],
    }
    with open(GENERATED_POLICY_PATH, "w") as f:
        json.dump(policy, f, indent=2)
    print(f"Wrote least-privilege IAM policy to {GENERATED_POLICY_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
    "--environment",
    required=True,
    choices=["dev", "staging", "production"],
    help="Namespaces every resource created so environments sharing an AWS "
    "account/region never collide on the same vault table or key.",
)
    args = parser.parse_args()
    table_name, key_alias = _resource_names(args.environment)

    session = boto3.session.Session(region_name=args.region)
    kms_client = session.client("kms")
    dynamodb_client = session.client("dynamodb")

    key_arn = create_kms_key(kms_client, key_alias)
    table_arn = create_table(dynamodb_client, table_name)
    write_iam_policy(table_arn, key_arn)

    print("\nDone. Set these in your environment:")
    print(f"  VAULT_BACKEND=dynamodb")
    print(f"  VAULT_DYNAMODB_TABLE={table_name}")
    print(f"  VAULT_KMS_KEY_ID={key_arn}")
    print(f"  AWS_REGION={args.region}")
    print(f"\nAttach {GENERATED_POLICY_PATH} to your ECS task role.")


if __name__ == "__main__":
    main()