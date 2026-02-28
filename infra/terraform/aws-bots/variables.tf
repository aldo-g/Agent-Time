variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project slug used in names and tags."
  type        = string
  default     = "agent-time"
}

variable "environment" {
  description = "Environment name (e.g. dev, prod)."
  type        = string
  default     = "dev"
}

variable "default_instance_type" {
  description = "Default EC2 instance type for bots."
  type        = string
  default     = "t3.micro"
}

variable "root_volume_size_gb" {
  description = "Root EBS size in GB."
  type        = number
  default     = 16
}

variable "image_tag" {
  description = "ECR image tag to run on all bots."
  type        = string
  default     = "latest"
}

variable "market_limit" {
  description = "Market limit passed to single_runner."
  type        = number
  default     = 25
}

variable "enable_shared_market_cache" {
  description = "When true, bots use a shared S3 market snapshot prepared by a dedicated fetcher instance."
  type        = bool
  default     = true
}

variable "create_market_fetcher_instance" {
  description = "When true, create a dedicated EC2 instance that fetches markets and uploads to S3."
  type        = bool
  default     = true
}

variable "shared_market_cache_object_key" {
  description = "S3 object key used for the shared market cache JSON."
  type        = string
  default     = "shared/shared_markets.json"
}

variable "shared_market_cache_force_destroy" {
  description = "Allow Terraform to delete the market cache bucket even if it still has objects."
  type        = bool
  default     = false
}

variable "market_fetcher_instance_type" {
  description = "EC2 instance type for the dedicated market fetcher host."
  type        = string
  default     = "t3.micro"
}

variable "market_fetcher_schedule" {
  description = "Systemd OnCalendar schedule for the dedicated market fetcher."
  type        = string
  default     = "hourly"
}

variable "market_fetcher_run_service_once_on_boot" {
  description = "Run one market-fetch cycle on boot for the dedicated fetcher host."
  type        = bool
  default     = true
}

variable "market_cache_max_age_seconds" {
  description = "Maximum acceptable age for shared market cache object when bots download it."
  type        = number
  default     = 7200
}

variable "market_cache_wait_seconds" {
  description = "How long non-fetcher bots wait for the shared market cache to be refreshed."
  type        = number
  default     = 180
}

variable "bot_skip_market_fetch" {
  description = "When true, bot runs require a pre-fetched market cache and pass --skip-market-fetch."
  type        = bool
  default     = true
}

variable "max_attempts" {
  description = "Max attempts per run passed to single_runner."
  type        = number
  default     = 2
}

variable "create_public_eips" {
  description = "Allocate and attach Elastic IPs to each bot."
  type        = bool
  default     = false
}

variable "enable_timers" {
  description = "Enable systemd timer on each bot host."
  type        = bool
  default     = true
}

variable "run_service_once_on_boot" {
  description = "Run one bot session on first boot."
  type        = bool
  default     = true
}

variable "enable_dashboard_service" {
  description = "Run the Predict Arena API/UI service on each bot EC2 host."
  type        = bool
  default     = false
}

variable "dashboard_port" {
  description = "Container and host port for the dashboard service."
  type        = number
  default     = 3000
}

variable "dashboard_host" {
  description = "Bind host used by api_server inside the dashboard container."
  type        = string
  default     = "0.0.0.0"
}

variable "dashboard_bind_address" {
  description = "EC2 host bind address for published dashboard port. Keep 127.0.0.1 for private access via SSM."
  type        = string
  default     = "127.0.0.1"
}

variable "dashboard_ingress_cidr" {
  description = "Optional CIDR allowed to access dashboard_port on bot instances. Set to 0.0.0.0/0 for public access."
  type        = string
  default     = null
}

variable "enable_dashboard_alb" {
  description = "Expose dashboard through an internet-facing Application Load Balancer (recommended for HTTPS + domain)."
  type        = bool
  default     = false
}

variable "dashboard_alb_ingress_cidr" {
  description = "CIDR allowed to access dashboard ALB on ports 80/443."
  type        = string
  default     = "0.0.0.0/0"
}

variable "dashboard_target_bot_key" {
  description = "Bot key whose dashboard service is registered as ALB target."
  type        = string
  default     = "gpt"
}

variable "dashboard_domain_name" {
  description = "Public DNS name for dashboard HTTPS endpoint (for example app.example.com)."
  type        = string
  default     = null
}

variable "dashboard_hosted_zone_name" {
  description = "Optional Route53 hosted zone name used to discover zone ID (for example example.com)."
  type        = string
  default     = null
}

variable "dashboard_route53_zone_id" {
  description = "Optional Route53 hosted zone ID. Preferred over hosted zone name when provided."
  type        = string
  default     = null
}

variable "dashboard_certificate_arn" {
  description = "Optional ACM certificate ARN for HTTPS listener. If unset, Terraform can create and validate one via Route53."
  type        = string
  default     = null
}

variable "create_dashboard_dns_record" {
  description = "Create Route53 alias records for dashboard_domain_name to the ALB."
  type        = bool
  default     = true
}

variable "dashboard_ssl_policy" {
  description = "SSL policy applied to dashboard HTTPS listener."
  type        = string
  default     = "ELBSecurityPolicy-2016-08"
}

variable "default_schedule" {
  description = "Default systemd OnCalendar schedule."
  type        = string
  default     = "hourly"
}

variable "enable_rds_postgres" {
  description = "When true, provision a managed RDS Postgres instance and store DATABASE_URL in SSM."
  type        = bool
  default     = false
}

variable "rds_instance_class" {
  description = "RDS instance class for managed Postgres."
  type        = string
  default     = "db.t4g.micro"
}

variable "rds_engine_version" {
  description = "Optional Postgres engine version for RDS. Set null to let AWS choose a region-supported default."
  type        = string
  default     = null
}

variable "rds_db_name" {
  description = "Database name created on the RDS instance."
  type        = string
  default     = "agent_time"
}

variable "rds_username" {
  description = "Master username for RDS Postgres."
  type        = string
  default     = "agent_time"
}

variable "rds_allocated_storage_gb" {
  description = "Initial allocated storage (GB) for RDS Postgres."
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage_gb" {
  description = "Autoscaling max storage (GB) for RDS Postgres."
  type        = number
  default     = 100
}

variable "rds_backup_retention_days" {
  description = "Backup retention period in days for RDS Postgres."
  type        = number
  default     = 1
}

variable "rds_publicly_accessible" {
  description = "Whether the RDS instance gets a public endpoint. Keep false for private-only DB access."
  type        = bool
  default     = false
}

variable "rds_deletion_protection" {
  description = "Enable deletion protection on the RDS instance."
  type        = bool
  default     = false
}

variable "rds_skip_final_snapshot" {
  description = "Skip final snapshot when destroying RDS. Keep true for low-friction dev/test teardown."
  type        = bool
  default     = true
}

variable "database_url_param_name" {
  description = "Optional SSM SecureString parameter name for shared DATABASE_URL (e.g. Supabase Postgres URL). Leave null to skip DB wiring."
  type        = string
  default     = null
}

variable "require_database_url" {
  description = "If true and database_url_param_name is set, fail bot startup when DATABASE_URL cannot be loaded."
  type        = bool
  default     = false
}

variable "ssh_ingress_cidr" {
  description = "Optional CIDR for SSH access (e.g. 1.2.3.4/32). Keep null to disable inbound SSH."
  type        = string
  default     = null
}

variable "common_env" {
  description = "Additional non-secret env vars added to all bot env files."
  type        = map(string)
  default = {
    AGENT_DISABLE_DOTENV                = "1"
    AGENT_SKIP_SCHEMA_INIT              = "1"
    AGENT_VERBOSE                       = "1"
    AGENT_LLM_LOG                       = "1"
  }
}

variable "bots" {
  description = "Bot definitions. SSM parameter names are derived from bot key and env var names."
  type = map(object({
    agent_name       = string
    llm_env_var      = string
    manifold_env_var = string
    instance_type    = optional(string)
    schedule         = optional(string)
  }))

  default = {
    gpt = {
      agent_name       = "gpt-runner"
      llm_env_var      = "OPENAI_API_KEY"
      manifold_env_var = "MANIFOLD_API_KEY_OPENAI"
    }
  }
}
