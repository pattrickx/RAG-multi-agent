"""
S3 client — wrapper boto3 para SeaweedFS / S3 compativel.

Funcoes:
    create_bucket()       — creates bucket with versioning
    upload_file()         — upload with auto-create bucket
    upload_folder()       — recursive folder upload
    list_bucket_contents() — lists objects
    delete_file()         — deletes object
    delete_bucket()       — deletes bucket and all contents
"""

import os

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")


# ---------------------------------------------------------------------------
# Client (module-level singleton)
# ---------------------------------------------------------------------------

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=ENDPOINT_URL,
)


# ---------------------------------------------------------------------------
# Bucket operations
# ---------------------------------------------------------------------------

def create_bucket(bucket_name: str) -> None:
    """Creates bucket with versioning enabled."""
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created.")
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={"Status": "Enabled"},
        )
        print(f"Versioning enabled for '{bucket_name}'.")
    except s3.exceptions.BucketAlreadyExists as e:
        print(f"Bucket already exists: {e}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket '{bucket_name}' already owned by you.")
    except ClientError as e:
        print(f"Failed to create bucket: {e}")
        print(f"Check ENDPOINT_URL '{ENDPOINT_URL}' points to SeaweedFS S3 API.")


def delete_bucket(bucket_name: str) -> None:
    """Deletes bucket and all its contents."""
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if "Contents" in response:
            for obj in response["Contents"]:
                s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
                print(f"Deleted {obj['Key']} from '{bucket_name}'")
        s3.delete_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' deleted.")
    except s3.exceptions.NoSuchBucket:
        print(f"Bucket '{bucket_name}' does not exist.")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials.")


def list_bucket_contents(bucket_name: str) -> list[dict]:
    """Lists bucket objects. Returns list of dicts with Key and Size."""
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if "Contents" in response:
            print(f"\nContents of '{bucket_name}':")
            for obj in response["Contents"]:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
            return response["Contents"]
        else:
            print(f"Bucket '{bucket_name}' is empty.")
            return []
    except s3.exceptions.NoSuchBucket:
        print(f"Bucket '{bucket_name}' does not exist.")
        return []
    except NoCredentialsError:
        print("AWS credentials not found.")
        return []
    except PartialCredentialsError:
        print("Incomplete AWS credentials.")
        return []


# ---------------------------------------------------------------------------
# Object operations
# ---------------------------------------------------------------------------

def upload_file(bucket_name: str, file_path: str, key: str) -> None:
    """Uploads file. Creates bucket if it does not exist."""
    try:
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                create_bucket(bucket_name)
            else:
                print(f"Error checking bucket: {e}")
                return
        s3.upload_file(file_path, bucket_name, key)
        print(f"Uploaded {file_path} as {key}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials.")


def upload_folder(bucket_name: str, folder_path: str) -> None:
    """Recursive folder upload."""
    try:
        for root, _dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                key = os.path.relpath(file_path, start=folder_path).replace("\\", "/")
                s3.upload_file(file_path, bucket_name, key)
                print(f"Uploaded {file_path} as {key}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials.")


def delete_file(bucket_name: str, key: str) -> None:
    """Deletes object from bucket."""
    try:
        s3.delete_object(Bucket=bucket_name, Key=key)
        print(f"Deleted {key} from '{bucket_name}'")
    except s3.exceptions.NoSuchBucket:
        print(f"Bucket '{bucket_name}' does not exist.")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bucket_name = "my-test-bucket"

    create_bucket(bucket_name)
    upload_file(bucket_name, "test_file.txt", "test_file.txt")
    list_bucket_contents(bucket_name)
    delete_file(bucket_name, "test_file.txt")
    list_bucket_contents(bucket_name)
    delete_bucket(bucket_name)
