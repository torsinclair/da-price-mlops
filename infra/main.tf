data "azurerm_client_config" "current" {}

# Storage accounts and key vaults share one global namespace, so the names
# need a suffix that is stable across plans but unique across deployments.
resource "random_string" "suffix" {
  length  = 6
  lower   = true
  upper   = false
  numeric = true
  special = false
}

# Azure requires three of four character classes. Special characters are
# excluded because this password goes into a postgresql:// URI, and
# percent-encoding it correctly in two places is a worse trade than
# spending the entropy on length.
resource "random_password" "postgres" {
  length  = 40
  lower   = true
  upper   = true
  numeric = true
  special = false
}

locals {
  suffix = random_string.suffix.result

  tags = merge({
    project     = "da-price-mlops"
    environment = var.environment
    managed_by  = "terraform"
  }, var.tags)

  artifact_container = "mlartifacts"

  # MLflow addresses blob storage with the wasbs scheme.
  artifact_root = "wasbs://${local.artifact_container}@${azurerm_storage_account.artifacts.name}.blob.core.windows.net"

  postgres_dsn = join("", [
    "postgresql://",
    "${azurerm_postgresql_flexible_server.mlflow.administrator_login}:",
    "${random_password.postgres.result}@",
    "${azurerm_postgresql_flexible_server.mlflow.fqdn}:5432/",
    "${azurerm_postgresql_flexible_server_database.mlflow.name}",
    "?sslmode=require",
  ])
}

resource "azurerm_resource_group" "main" {
  name     = "${var.name_prefix}-${var.environment}-rg"
  location = var.location
  tags     = local.tags
}

# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------

resource "azurerm_storage_account" "artifacts" {
  name                = "${var.name_prefix}${var.environment}${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier             = "Standard"
  account_replication_type = var.storage_replication_type
  account_kind             = "StorageV2"

  https_traffic_only_enabled      = true
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false

  # The tracking server authenticates with its managed identity, so the
  # shared account keys are never needed and are therefore disabled.
  shared_access_key_enabled = false

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 7
    }
  }

  tags = local.tags
}

resource "azurerm_storage_container" "artifacts" {
  name                  = local.artifact_container
  storage_account_id    = azurerm_storage_account.artifacts.id
  container_access_type = "private"
}

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

resource "azurerm_key_vault" "main" {
  name                = "${var.name_prefix}-${var.environment}-kv-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  # RBAC rather than access policies: access policies are per-vault ACLs
  # that no central role assignment can audit.
  rbac_authorization_enabled = true

  purge_protection_enabled   = false
  soft_delete_retention_days = 7

  tags = local.tags
}

# The principal running terraform needs to be able to write the secrets it
# creates. On a first apply this race is real: role assignments take up to
# a minute to propagate and the secret writes below can fail once. Re-run.
resource "azurerm_role_assignment" "deployer_secrets_officer" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_key_vault_secret" "postgres_dsn" {
  name         = "mlflow-backend-store-uri"
  value        = local.postgres_dsn
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_role_assignment.deployer_secrets_officer]
}

resource "azurerm_key_vault_secret" "entsoe_api_key" {
  count = var.entsoe_api_key == null ? 0 : 1

  name         = "entsoe-api-key"
  value        = var.entsoe_api_key
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_role_assignment.deployer_secrets_officer]
}

# ---------------------------------------------------------------------------
# Backend store
# ---------------------------------------------------------------------------

resource "azurerm_postgresql_flexible_server" "mlflow" {
  name                = "${var.name_prefix}-${var.environment}-pg-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  version                       = var.postgres_version
  administrator_login           = "mlflow"
  administrator_password        = random_password.postgres.result
  sku_name                      = var.postgres_sku_name
  storage_mb                    = var.postgres_storage_mb
  backup_retention_days         = 7
  public_network_access_enabled = true
  zone                          = "1"

  tags = local.tags

  lifecycle {
    # The server is created in whichever zone has capacity, and a later plan
    # that moves it would destroy the backend store.
    ignore_changes = [zone]
  }
}

resource "azurerm_postgresql_flexible_server_database" "mlflow" {
  name      = "mlflow"
  server_id = azurerm_postgresql_flexible_server.mlflow.id
  charset   = "UTF8"
  collation = "en_US.utf8"

  lifecycle {
    prevent_destroy = false
  }
}

# Public access with a firewall, not a private endpoint. The all-zero rule is
# Azure's documented marker for "other Azure services", which is how the
# Container App reaches the server without a VNet. A production estate puts
# both ends in a VNet and deletes this rule; see infra/README.md.
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.mlflow.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "client" {
  count = var.allowed_client_ip == null ? 0 : 1

  name             = "allow-client"
  server_id        = azurerm_postgresql_flexible_server.mlflow.id
  start_ip_address = var.allowed_client_ip
  end_ip_address   = var.allowed_client_ip
}

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

resource "azurerm_user_assigned_identity" "tracking" {
  name                = "${var.name_prefix}-${var.environment}-tracking-id"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_role_assignment" "tracking_blob_contributor" {
  scope                = azurerm_storage_account.artifacts.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.tracking.principal_id
}

resource "azurerm_role_assignment" "tracking_secrets_user" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.tracking.principal_id
}

# ---------------------------------------------------------------------------
# Tracking server
# ---------------------------------------------------------------------------

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.name_prefix}-${var.environment}-law"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

resource "azurerm_container_app_environment" "main" {
  name                       = "${var.name_prefix}-${var.environment}-cae"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags
}

resource "azurerm_container_app" "tracking" {
  name                         = "${var.name_prefix}-${var.environment}-mlflow"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.tracking.id]
  }

  secret {
    name                = "backend-store-uri"
    key_vault_secret_id = azurerm_key_vault_secret.postgres_dsn.id
    identity            = azurerm_user_assigned_identity.tracking.id
  }

  ingress {
    external_enabled = var.tracking_server_ingress_external
    target_port      = 5000
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 1
    max_replicas = 1

    container {
      name   = "mlflow"
      image  = var.mlflow_image
      cpu    = 0.5
      memory = "1Gi"

      command = ["mlflow"]
      args = [
        "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--default-artifact-root", local.artifact_root,
        "--serve-artifacts",
      ]

      # The DSN carries the database password, so it arrives as a secret
      # reference rather than an argument: container arguments are readable
      # in the resource definition by anyone with reader on the group.
      env {
        name        = "MLFLOW_BACKEND_STORE_URI"
        secret_name = "backend-store-uri"
      }

      # Tells the Azure SDK inside MLflow which user-assigned identity to
      # present when it reaches for the artifact container.
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.tracking.client_id
      }

      # MLflow 3.x rejects any Host header outside localhost and private IPs.
      # Requests through Container Apps ingress carry the app's FQDN, which
      # sits under the environment's default domain.
      env {
        name  = "MLFLOW_SERVER_ALLOWED_HOSTS"
        value = "*.${azurerm_container_app_environment.main.default_domain}"
      }
    }
  }

  depends_on = [
    azurerm_role_assignment.tracking_blob_contributor,
    azurerm_role_assignment.tracking_secrets_user,
  ]
}
