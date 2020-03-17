import os

from google.oauth2 import service_account
import googleapiclient.discovery

def list_keys(service_account_email):
    """Lists all keys for a service account."""

    credentials = service_account.Credentials.from_service_account_file(
        filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
        scopes=['https://www.googleapis.com/auth/cloud-platform'])

    service = googleapiclient.discovery.build(
        'iam', 'v1', credentials=credentials)

    keys = service.projects().serviceAccounts().keys().list(
        name='projects/-/serviceAccounts/' + service_account_email).execute()

    for key in keys['keys']:
        print('Key: ' + key['name'])