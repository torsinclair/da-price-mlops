terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.42"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }

  # State lives locally until this is applied for real. The remote backend
  # below is the shape it takes then: state carries the Postgres admin
  # password, so a shared apply needs a backend with encryption and
  # locking rather than a file in the working directory.
  #
  # backend "azurerm" {
  #   resource_group_name  = "dapm-tfstate"
  #   storage_account_name = "dapmtfstate"
  #   container_name       = "tfstate"
  #   key                  = "dapm.tfstate"
  #   use_azuread_auth     = true
  # }
}

provider "azurerm" {
  features {}

  # Required from azurerm 4.x; no longer inferred from the CLI context.
  subscription_id = var.subscription_id
}
