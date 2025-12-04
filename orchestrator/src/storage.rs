//! Object Storage Integration
//!
//! Supports uploading artifacts to S3 (AWS/MinIO) or GCS.

use std::path::Path;
use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Unsupported storage scheme: {0}")]
    UnsupportedScheme(String),
    #[error("Invalid URI: {0}")]
    InvalidUri(String),
    #[cfg(feature = "storage-s3")]
    #[error("S3 error: {0}")]
    S3Error(String),
    #[cfg(feature = "storage-gcs")]
    #[error("GCS error: {0}")]
    GcsError(String),
    #[error("Storage feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

/// Upload a file to object storage
///
/// Supports:
/// - `s3://bucket/key` - AWS S3 or MinIO
/// - `gs://bucket/key` - Google Cloud Storage
/// - `file://path` - Local file (just copies)
pub async fn upload(local_path: &Path, uri: &str) -> Result<(), StorageError> {
    info!("Uploading {} to {}", local_path.display(), uri);

    if uri.starts_with("s3://") {
        upload_s3(local_path, uri).await
    } else if uri.starts_with("gs://") {
        upload_gcs(local_path, uri).await
    } else if uri.starts_with("file://") {
        upload_local(local_path, uri).await
    } else {
        Err(StorageError::UnsupportedScheme(uri.to_string()))
    }
}

/// Upload to S3 (requires storage-s3 feature)
#[cfg(feature = "storage-s3")]
async fn upload_s3(local_path: &Path, uri: &str) -> Result<(), StorageError> {
    use aws_sdk_s3::primitives::ByteStream;

    // Parse URI: s3://bucket/key
    let uri = uri.strip_prefix("s3://").unwrap();
    let (bucket, key) = uri
        .split_once('/')
        .ok_or_else(|| StorageError::InvalidUri(format!("s3://{}", uri)))?;

    // Load AWS config
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);

    // Read file
    let body = ByteStream::from_path(local_path)
        .await
        .map_err(|e| StorageError::S3Error(e.to_string()))?;

    // Upload
    client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(body)
        .send()
        .await
        .map_err(|e| StorageError::S3Error(e.to_string()))?;

    info!("Uploaded to s3://{}/{}", bucket, key);
    Ok(())
}

#[cfg(not(feature = "storage-s3"))]
async fn upload_s3(_local_path: &Path, uri: &str) -> Result<(), StorageError> {
    Err(StorageError::FeatureNotEnabled(format!(
        "S3 storage not enabled. Enable 'storage-s3' feature to upload to {}",
        uri
    )))
}

/// Upload to GCS (requires storage-gcs feature)
#[cfg(feature = "storage-gcs")]
async fn upload_gcs(local_path: &Path, uri: &str) -> Result<(), StorageError> {
    use cloud_storage::Client;
    use tokio::fs::File;
    use tokio::io::AsyncReadExt;

    // Parse URI: gs://bucket/key
    let uri = uri.strip_prefix("gs://").unwrap();
    let (bucket, key) = uri
        .split_once('/')
        .ok_or_else(|| StorageError::InvalidUri(format!("gs://{}", uri)))?;

    // Read file
    let mut file = File::open(local_path).await?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).await?;

    // Upload
    let client = Client::default();
    client
        .object()
        .create(bucket, data, key, "application/octet-stream")
        .await
        .map_err(|e| StorageError::GcsError(e.to_string()))?;

    info!("Uploaded to gs://{}/{}", bucket, key);
    Ok(())
}

#[cfg(not(feature = "storage-gcs"))]
async fn upload_gcs(_local_path: &Path, uri: &str) -> Result<(), StorageError> {
    Err(StorageError::FeatureNotEnabled(format!(
        "GCS storage not enabled. Enable 'storage-gcs' feature to upload to {}",
        uri
    )))
}

/// Local file copy (for testing)
async fn upload_local(local_path: &Path, uri: &str) -> Result<(), StorageError> {
    let dest = uri.strip_prefix("file://").unwrap();
    let dest_path = Path::new(dest);

    // Create parent directory
    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    tokio::fs::copy(local_path, dest_path).await?;
    info!("Copied to {}", dest);
    Ok(())
}

/// Download a file from object storage
pub async fn download(uri: &str, local_path: &Path) -> Result<(), StorageError> {
    info!("Downloading {} to {}", uri, local_path.display());

    if uri.starts_with("s3://") {
        download_s3(uri, local_path).await
    } else if uri.starts_with("gs://") {
        download_gcs(uri, local_path).await
    } else if uri.starts_with("file://") {
        download_local(uri, local_path).await
    } else {
        Err(StorageError::UnsupportedScheme(uri.to_string()))
    }
}

#[cfg(feature = "storage-s3")]
async fn download_s3(uri: &str, local_path: &Path) -> Result<(), StorageError> {
    use tokio::io::AsyncWriteExt;

    let uri = uri.strip_prefix("s3://").unwrap();
    let (bucket, key) = uri
        .split_once('/')
        .ok_or_else(|| StorageError::InvalidUri(format!("s3://{}", uri)))?;

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);

    let resp = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| StorageError::S3Error(e.to_string()))?;

    let data = resp
        .body
        .collect()
        .await
        .map_err(|e| StorageError::S3Error(e.to_string()))?;

    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut file = tokio::fs::File::create(local_path).await?;
    file.write_all(&data.into_bytes()).await?;

    Ok(())
}

#[cfg(not(feature = "storage-s3"))]
async fn download_s3(uri: &str, _local_path: &Path) -> Result<(), StorageError> {
    Err(StorageError::FeatureNotEnabled(format!(
        "S3 storage not enabled. Enable 'storage-s3' feature to download from {}",
        uri
    )))
}

#[cfg(feature = "storage-gcs")]
async fn download_gcs(uri: &str, local_path: &Path) -> Result<(), StorageError> {
    use cloud_storage::Client;
    use tokio::io::AsyncWriteExt;

    let uri = uri.strip_prefix("gs://").unwrap();
    let (bucket, key) = uri
        .split_once('/')
        .ok_or_else(|| StorageError::InvalidUri(format!("gs://{}", uri)))?;

    let client = Client::default();
    let data = client
        .object()
        .download(bucket, key)
        .await
        .map_err(|e| StorageError::GcsError(e.to_string()))?;

    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut file = tokio::fs::File::create(local_path).await?;
    file.write_all(&data).await?;

    Ok(())
}

#[cfg(not(feature = "storage-gcs"))]
async fn download_gcs(uri: &str, _local_path: &Path) -> Result<(), StorageError> {
    Err(StorageError::FeatureNotEnabled(format!(
        "GCS storage not enabled. Enable 'storage-gcs' feature to download from {}",
        uri
    )))
}

async fn download_local(uri: &str, local_path: &Path) -> Result<(), StorageError> {
    let src = uri.strip_prefix("file://").unwrap();
    let src_path = Path::new(src);

    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    tokio::fs::copy(src_path, local_path).await?;
    Ok(())
}

