provider "google" {
  credentials = file(var.credentials_file_path) 
  project     = var.project_id
  region      = var.region
}

variable "project_id" {
  description = "The ID of your GCP project"
  type        = string
}

variable "region" {
  description = "The region for GCP resources"
  type        = string
  default     = "us-central1"
}
