variable "subscription_id" {
  description = "Azure subscription to deploy into."
  type        = string
}

variable "location" {
  description = "Azure region. SE3 is the modelled bidding zone, so the data stays in the same country as the market it describes."
  type        = string
  default     = "swedencentral"
}

variable "name_prefix" {
  description = "Prefix for every resource name. Keep it short: storage account names are capped at 24 characters and allow only lowercase letters and digits."
  type        = string
  default     = "dapm"

  validation {
    condition     = can(regex("^[a-z][a-z0-9]{1,10}$", var.name_prefix))
    error_message = "name_prefix must be 2-11 characters, lowercase letters and digits, starting with a letter."
  }
}

variable "environment" {
  description = "Environment name, used in tags and in the MLflow experiment prefix."
  type        = string
  default     = "dev"
}

variable "postgres_sku_name" {
  description = "Flexible Server compute tier. B1ms is the smallest burstable size and is what the MLflow backend store needs at this scale; the registry is metadata, not a workload."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "postgres_storage_mb" {
  description = "Flexible Server storage. 32768 is the floor."
  type        = number
  default     = 32768
}

variable "postgres_version" {
  description = "PostgreSQL major version."
  type        = string
  default     = "16"
}

variable "storage_replication_type" {
  description = "Artifact store redundancy. LRS is sufficient for artifacts that are reproducible from the training code; ZRS or GRS if a lost artifact would mean a lost audit trail."
  type        = string
  default     = "LRS"
}

variable "allowed_client_ip" {
  description = "Single public IP allowed through the Postgres firewall, for running migrations or psql from a workstation. Leave null to allow no direct client access."
  type        = string
  default     = null
}

variable "tracking_server_ingress_external" {
  description = "Expose the MLflow tracking server to the public internet. Defaults to false: Container Apps ingress carries no authentication of its own, and MLflow has no user model, so an external endpoint is an unauthenticated registry. See infra/README.md."
  type        = bool
  default     = false
}

variable "mlflow_image" {
  description = "Container image for the tracking server. The -full variant, because the default image installs bare mlflow: no Postgres driver and no Azure Blob client. Pinned by tag rather than digest here; a real deployment pins the digest so a retag cannot change what runs."
  type        = string
  default     = "ghcr.io/mlflow/mlflow:v3.15.0-full"
}

variable "entsoe_api_key" {
  description = "ENTSO-E transparency platform token, stored in Key Vault. Leave null to create the vault without the secret and set it out of band."
  type        = string
  default     = null
  sensitive   = true
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
