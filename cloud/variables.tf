variable "bucket_name" {
  description = "The name of the GCS bucket"
  type        = string
  default     = "codesprint-data-lake"
}

variable "bucket_location" {
  description = "The GCP region to create the bucket in"
  type        = string
  default     = "US"
}

variable "storage_class" {
  description = "The storage class for the bucket"
  type        = string
  default     = "STANDARD"
}

variable "enable_versioning" {
  description = "Whether to enable versioning on the bucket"
  type        = bool
  default     = true
}

variable "file_retention_days" {
  description = "Number of days after which objects should be deleted"
  type        = number
  default     = 365
}

variable "service_account_email" {
  description = "The service account email that will manage the bucket"
  type        = string
}

variable "public_access" {
  description = "Set to true to allow public read access to the bucket"
  type        = bool
  default     = true
}

variable "credentials_file_path" {
  description = "Path to the service account credentials JSON file"
  type        = string
}