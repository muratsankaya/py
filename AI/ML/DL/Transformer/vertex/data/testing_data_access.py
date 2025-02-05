from google.cloud import storage
import pandas as pd


def list_bucket_files(bucket_name):
    """Lists all the files in the given bucket."""
    client = storage.Client()

    # Access the private bucket
    bucket = client.bucket(bucket_name)

    # List all objects in the bucket
    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)


def download_file_from_bucket(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Retrieve blob (file) from the bucket
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"File {source_blob_name} downloaded to {destination_file_name}.")

    # Convert data saved file .csv from .parquet
    df = pd.read_parquet(destination_file_name)
    df.to_csv(
        destination_file_name[: destination_file_name.rfind(".")] + ".csv", index=False
    )


if __name__ == "__main__":
    # Replace with your private bucket name and file paths
    BUCKET_NAME = "transformer_demo_wmt14"
    SOURCE_BLOB_NAME = "test/test-00000-of-00001.parquet"
    DESTINATION_FILE_NAME = "vertex/data/test-00000-of-00001.parquet"

    # List files in the private bucket
    list_bucket_files(BUCKET_NAME)

    # Download a file from the private bucket
    download_file_from_bucket(BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME)
