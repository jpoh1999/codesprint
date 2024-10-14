resource "google_storage_bucket" "data_lake_bucket" {
  name                        = var.bucket_name
  location                    = var.bucket_location
  force_destroy               = true  # Allows bucket deletion even if not empty
  storage_class               = var.storage_class

  # Enforce uniform bucket-level access (recommended for controlling permissions)
  uniform_bucket_level_access = true

  versioning {
    enabled = var.enable_versioning
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = var.file_retention_days  # Automatically delete objects older than specified days
    }
  }
}

# IAM binding for the Cloud Storage bucket to allow public read access or specific service roles
resource "google_storage_bucket_iam_member" "storage_admin_role" {
  bucket = google_storage_bucket.data_lake_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.service_account_email}"
}

resource "google_storage_bucket_iam_member" "storage_writer_role" {
  bucket = google_storage_bucket.data_lake_bucket.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${var.service_account_email}"
}

resource "google_storage_bucket_iam_member" "storage_reader_role" {
  bucket = google_storage_bucket.data_lake_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.service_account_email}"
}

# Optional: Public access to the bucket
resource "google_storage_bucket_iam_member" "public_access" {
  count  = var.public_access ? 1 : 0
  bucket = google_storage_bucket.data_lake_bucket.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}