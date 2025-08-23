import io
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import time  # Corrected import — you had `from datetime import time` which doesn't work

async def upload_html_to_s3(html_content, object_name):
    """
    Upload HTML content to an S3 bucket using an in-memory buffer.
    
    :param html_content: HTML string to upload
    :param object_name: S3 object key name in the bucket
    :return: True if upload succeeded, False otherwise
    """
    bucket_name = "clustering-results-for-foursides-clustering-graphs"

    session = boto3.Session(profile_name='kacha')
    s3 = session.client('s3', region_name='ap-northeast-1')

    # Convert string to BytesIO
    buffer = io.BytesIO(html_content.encode("utf-8"))

    try:
        s3.upload_fileobj(
            buffer,
            bucket_name,
            object_name,
            ExtraArgs={
                'ContentType': 'text/html',
                'ACL': 'bucket-owner-full-control'
            }
        )
        url=f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        print(f"✅ File uploaded to https://{bucket_name}.s3.amazonaws.com/{object_name}")
        return url
    except NoCredentialsError:
        print("❌ AWS credentials not available.")
    except ClientError as e:
        print(f"❌ Unexpected error: {e}")
