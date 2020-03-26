import os
import sys
from google.cloud import storage

devKeyPath = os.getenv("devKey")
devKey = str(open(devKeyPath, "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getenv("HOME"), "Downloads", "google-creds.json")

PARSED_BUCKET = "parsed_transcript_bucket"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return


if __name__ == "__main__":
    upload_blob(PARSED_BUCKET, sys.argv[1], sys.argv[1])