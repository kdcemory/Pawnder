# s3_sync.py
# Utility for syncing data between Google Drive and AWS S3

import boto3
import os
import json
import logging
import subprocess
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('s3_sync')

class S3Sync:
    """
    A utility class for synchronizing data between Google Drive and AWS S3.
    Handles uploads, downloads, and model checkpoint management.
    """
    
    def __init__(self, bucket_name, aws_access_key=None, aws_secret_key=None, region=None):
        """
        Initialize S3 sync utility.
        
        Parameters:
            bucket_name (str): S3 bucket name
            aws_access_key (str, optional): AWS access key ID
            aws_secret_key (str, optional): AWS secret access key
            region (str, optional): AWS region name
        """
        # Try to load credentials from various sources
        self.aws_access_key = aws_access_key or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = aws_secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.region = region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Try to load from Colab secrets if available
        if not (self.aws_access_key and self.aws_secret_key):
            try:
                from google.colab import userdata
                self.aws_access_key = userdata.get('AWS_ACCESS_KEY_ID')
                self.aws_secret_key = userdata.get('AWS_SECRET_ACCESS_KEY')
                logger.info("Loaded AWS credentials from Colab secrets")
            except (ImportError, Exception) as e:
                logger.debug(f"Could not load credentials from Colab secrets: {str(e)}")
        
        self.bucket_name = bucket_name
        
        # Initialize S3 client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region
        )
        
        logger.info(f"S3Sync initialized for bucket: {bucket_name}")
    
    def test_connection(self):
        """
        Test the AWS S3 connection by listing the bucket contents.
        
        Returns:
            bool: True if connection succeeds, False otherwise
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to S3: {str(e)}")
            return False
    
    def download_folder(self, s3_prefix, local_path):
        """
        Download a folder from S3 to local storage.
        
        Parameters:
            s3_prefix (str): S3 path prefix (folder path)
            local_path (str): Local directory path
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure the directory exists
            Path(local_path).mkdir(parents=True, exist_ok=True)
            
            # Use AWS CLI for efficient directory sync
            cmd = f"aws s3 sync s3://{self.bucket_name}/{s3_prefix} {local_path}"
            logger.info(f"Running: {cmd}")
            
            # Run the command
            process = subprocess.run(cmd, shell=True, check=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Downloaded files from s3://{self.bucket_name}/{s3_prefix} to {local_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"AWS CLI error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            
            # Fallback to using boto3 directly
            logger.info("Falling back to boto3 for downloads...")
            try:
                # List all objects in the prefix
                paginator = self.s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
                
                # Download each object
                downloaded_count = 0
                for page in pages:
                    if 'Contents' not in page:
                        continue
                        
                    for obj in page['Contents']:
                        # Get relative path
                        key = obj['Key']
                        if not key.startswith(s3_prefix):
                            continue
                            
                        rel_path = key[len(s3_prefix):].lstrip('/')
                        if not rel_path:  # Skip the directory itself
                            continue
                            
                        # Create directory if needed
                        local_file_path = os.path.join(local_path, rel_path)
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        
                        # Download the file
                        logger.debug(f"Downloading {key} to {local_file_path}")
                        self.s3.download_file(self.bucket_name, key, local_file_path)
                        downloaded_count += 1
                
                logger.info(f"Downloaded {downloaded_count} files using boto3")
                return True
                
            except Exception as inner_e:
                logger.error(f"Failed to download from S3 using boto3: {str(inner_e)}")
                return False
    
    def upload_folder(self, local_path, s3_prefix):
        """
        Upload a folder to S3.
        
        Parameters:
            local_path (str): Local directory path
            s3_prefix (str): S3 path prefix (folder path)
            
        Returns:
            bool: Success status
        """
        try:
            # Check if directory exists
            if not os.path.exists(local_path):
                logger.error(f"Local path does not exist: {local_path}")
                return False
            
            # Use AWS CLI for efficient directory sync
            cmd = f"aws s3 sync {local_path} s3://{self.bucket_name}/{s3_prefix}"
            logger.info(f"Running: {cmd}")
            
            # Run the command
            process = subprocess.run(cmd, shell=True, check=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Uploaded files from {local_path} to s3://{self.bucket_name}/{s3_prefix}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"AWS CLI error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            
            # Fallback to using boto3 directly
            logger.info("Falling back to boto3 for uploads...")
            try:
                # Walk the local directory
                uploaded_count = 0
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        # Get the full local path
                        local_file_path = os.path.join(root, file)
                        
                        # Get the relative path to construct S3 key
                        rel_path = os.path.relpath(local_file_path, local_path)
                        s3_key = f"{s3_prefix}/{rel_path}".replace('\\', '/')
                        
                        # Upload the file
                        logger.debug(f"Uploading {local_file_path} to s3://{self.bucket_name}/{s3_key}")
                        self.s3.upload_file(local_file_path, self.bucket_name, s3_key)
                        uploaded_count += 1
                
                logger.info(f"Uploaded {uploaded_count} files using boto3")
                return True
                
            except Exception as inner_e:
                logger.error(f"Failed to upload to S3 using boto3: {str(inner_e)}")
                return False
    
    def upload_file(self, local_file_path, s3_key):
        """
        Upload a single file to S3.
        
        Parameters:
            local_file_path (str): Path to local file
            s3_key (str): S3 key (path) for the file
            
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(local_file_path):
                logger.error(f"Local file does not exist: {local_file_path}")
                return False
            
            logger.info(f"Uploading {local_file_path} to s3://{self.bucket_name}/{s3_key}")
            self.s3.upload_file(local_file_path, self.bucket_name, s3_key)
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {str(e)}")
            return False
    
    def download_file(self, s3_key, local_file_path):
        """
        Download a single file from S3.
        
        Parameters:
            s3_key (str): S3 key (path) for the file
            local_file_path (str): Path to save local file
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_file_path}")
            self.s3.download_file(self.bucket_name, s3_key, local_file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download file from S3: {str(e)}")
            return False
    
    def sync_model_checkpoints(self, local_checkpoint_dir, model_name):
        """
        Sync model checkpoints bidirectionally between local and S3.
        
        Parameters:
            local_checkpoint_dir (str): Local directory with checkpoint files
            model_name (str): Name of the model for S3 path
            
        Returns:
            bool: Success status
        """
        s3_prefix = f"models/{model_name}/checkpoints"
        
        # First, download any checkpoints that don't exist locally
        logger.info(f"Syncing checkpoints from S3 to local...")
        download_success = self.download_folder(s3_prefix, local_checkpoint_dir)
        
        # Then upload any new local checkpoints to S3
        logger.info(f"Syncing checkpoints from local to S3...")
        upload_success = self.upload_folder(local_checkpoint_dir, s3_prefix)
        
        return download_success and upload_success
    
    def list_models(self):
        """
        List all models in S3.
        
        Returns:
            list: List of model names
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='models/',
                Delimiter='/'
            )
            
            models = []
            for prefix in response.get('CommonPrefixes', []):
                # Extract model name from prefix
                prefix_str = prefix.get('Prefix')
                if prefix_str.startswith('models/') and prefix_str.endswith('/'):
                    model_name = prefix_str[7:-1]  # Remove 'models/' and trailing '/'
                    models.append(model_name)
            
            logger.info(f"Found {len(models)} models in S3")
            return models
        except Exception as e:
            logger.error(f"Failed to list models in S3: {str(e)}")
            return []

    def check_file_exists(self, s3_key):
        """
        Check if a file exists in S3.
        
        Parameters:
            s3_key (str): S3 key (path) to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False

if __name__ == "__main__":
    # Example usage
    sync = S3Sync('pawnder-media-storage')
    if sync.test_connection():
        print("Connection to S3 successful!")
        models = sync.list_models()
        print(f"Available models: {models}")
    else:
        print("Failed to connect to S3.")
