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
  description = "Market limit passed to multi_runner."
  type        = number
  default     = 10
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

variable "max_attempts" {
  description = "Max attempts per run passed to multi_runner."
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

variable "default_schedule" {
  description = "Default systemd OnCalendar schedule."
  type        = string
  default     = "hourly"
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
    MANIFOLD_VERIFY_BEFORE_EACH_REQUEST = "1"
  }
}

variable "bots" {
  description = "Bot definitions. SSM parameter names are derived from bot key and env var names."
  type = map(object({
    agent_name       = string
    expected_wallet  = string
    llm_env_var      = string
    manifold_env_var = string
    instance_type    = optional(string)
    schedule         = optional(string)
  }))

  default = {
    gpt = {
      agent_name       = "gpt-runner"
      expected_wallet  = "AgentChatGPT"
      llm_env_var      = "OPENAI_API_KEY"
      manifold_env_var = "MANIFOLD_API_KEY_OPENAI"
    }
    claude = {
      agent_name       = "claude-runner"
      expected_wallet  = "AgentClaude"
      llm_env_var      = "CLAUDE_API_KEY"
      manifold_env_var = "MANIFOLD_API_KEY_CLAUDE"
    }
    gemini = {
      agent_name       = "gemini-runner"
      expected_wallet  = "AgentGemini"
      llm_env_var      = "GEMINI_API_KEY"
      manifold_env_var = "MANIFOLD_API_KEY_GEMINI"
    }
  }
}
