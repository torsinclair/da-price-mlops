# The outputs are deliberately the five environment variables the README
# names as the cost of portability, so that what terraform produces lines up
# with what the application already expects to be told.

output "mlflow_tracking_uri" {
  description = "Value for MLFLOW_TRACKING_URI. Internal ingress resolves only from inside the Container Apps environment."
  value       = "https://${azurerm_container_app.tracking.ingress[0].fqdn}"
}

output "artifact_root" {
  description = "Blob container the tracking server writes artifacts to."
  value       = local.artifact_root
}

output "key_vault_name" {
  description = "Vault holding the backend store URI and, if supplied, the ENTSO-E token."
  value       = azurerm_key_vault.main.name
}

output "backend_store_uri_secret_id" {
  description = "Key Vault secret holding the Postgres DSN. The value is not an output: it would then sit in state in plaintext twice."
  value       = azurerm_key_vault_secret.postgres_dsn.id
}

output "tracking_identity_client_id" {
  description = "Client ID of the user-assigned identity, for AZURE_CLIENT_ID in any workload that writes artifacts directly."
  value       = azurerm_user_assigned_identity.tracking.client_id
}

output "postgres_fqdn" {
  description = "Flexible Server hostname, for running migrations or psql from an allowed client IP."
  value       = azurerm_postgresql_flexible_server.mlflow.fqdn
}

output "resource_group_name" {
  description = "Resource group holding everything, for a single-command teardown."
  value       = azurerm_resource_group.main.name
}
