terraform {
  required_version = ">= 1.0"
  
  required_providers {
    ec = {
      source  = "elastic/ec"
      version = "~> 0.13"
    }
  }
}

provider "ec" {
  apikey = var.elastic_cloud_api_key
}

resource "ec_elasticsearch_project" "demo_project" {
  region_id     = var.region
  name          = "demo_project"
  optimized_for = "general_purpose"
}
