# infra

Terraform for the persistent backend the root README describes as the
intended production shape: *"SQLite is the zero-infrastructure database
backend; production would swap the URI for Postgres"* and *"pointing the
same job at a persistent tracking server is a one-env-var change."*

This directory is that sentence made executable. It provisions the target
the loop points at; it does not change a line of the loop.

## What it provisions

| Resource | Why |
|---|---|
| PostgreSQL Flexible Server + database | MLflow backend store. The registry never worked on the filesystem store, and SQLite does not survive a container restart. |
| Storage account + private container | Artifact store, replacing local `mlartifacts/`. Versioning on, soft delete 7 days. |
| Key Vault (RBAC) | The Postgres DSN and the ENTSO-E token, which today live in `.env`. |
| User-assigned managed identity | Lets the tracking server read the vault and write artifacts without a secret in the repo. Account keys on the storage account are disabled, so identity is the only path. |
| Log Analytics workspace | Container Apps log destination. |
| Container App | The MLflow tracking server itself. Internal ingress by default. |

Outputs are the environment variables the application already expects, so
`terraform output` lines up with the `.env.databricks` pattern rather than
inventing a second configuration vocabulary.

## Running it

    cd infra
    cp terraform.tfvars.example terraform.tfvars   # fill in subscription_id
    terraform init
    terraform plan
    terraform apply

Teardown is `terraform destroy`, or deleting the one resource group that
holds everything.

CI runs `fmt -check`, `init -backend=false` and `validate` on every change
under `infra/`. It does not run `plan` or `apply`: that needs credentials,
and a repository that can plan against a subscription is a repository whose
secrets are worth stealing.

## What this is not

**It has not been applied.** The configuration is written and validated in
CI against the real provider schema; it has never been run against a live
subscription. Validation proves the configuration is well-formed and
schema-correct, not that the resulting system comes up. The gap between
those two is exactly the sort of thing this repo says out loud elsewhere, so
it says it here too.

**No authentication in front of the tracking server.** MLflow has no user
model and Container Apps ingress adds none, so an externally reachable
endpoint is an unauthenticated registry that anyone can move `@champion` on.
Ingress therefore defaults to internal. Making it external is a variable,
but the honest production answer is Container Apps' built-in auth against
Entra ID, or a private endpoint and a VPN — not flipping the flag.

**Public network access on Postgres, with a firewall.** The all-zero
firewall rule is Azure's marker for "other Azure services", which is how the
Container App reaches the database without a VNet. The production shape is
both ends in a VNet with a private endpoint and that rule deleted. This
version trades an attack surface for about sixty lines of networking.

**State is local.** The remote backend is written out and commented in
`versions.tf`. It matters more than it looks: state holds the Postgres admin
password in plaintext, so a shared apply needs an encrypted, locking backend
before a second person touches it.

**One environment.** `environment` is a variable and names flow from it, but
there are no workspaces, no per-environment tfvars and no promotion path
between them.

**The image is pinned by tag, not digest.** A retag upstream changes what
runs on the next revision. A deployment that cares pins the digest.

## Known rough edges

- First `apply` can fail once on the Key Vault secret writes. The deployer's
  "Key Vault Secrets Officer" assignment is created in the same run and RBAC
  propagation lags it by up to a minute. Re-running succeeds. The fix is a
  separate bootstrap apply for the vault, which is more machinery than one
  retry is worth here.
- `zone` on the Flexible Server is in `ignore_changes`. Azure assigns
  whichever zone has capacity, and a later plan that moved it would destroy
  the backend store.
- The Postgres password is generated without special characters. It goes
  into a `postgresql://` URI, and percent-encoding it correctly in both the
  DSN and the vault is a worse trade than spending the entropy on length:
  40 alphanumeric characters is ~206 bits.
