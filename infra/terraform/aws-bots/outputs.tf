output "ecr_repository_url" {
  description = "ECR repository URL for bot images."
  value       = aws_ecr_repository.bots.repository_url
}

output "ecr_image_uri" {
  description = "Image URI expected by instances."
  value       = "${aws_ecr_repository.bots.repository_url}:${var.image_tag}"
}

output "cloudwatch_log_groups" {
  description = "CloudWatch log groups for bot container logs."
  value = {
    for key, lg in aws_cloudwatch_log_group.bots : key => lg.name
  }
}

output "market_fetcher_log_group" {
  description = "CloudWatch log group for dedicated market fetcher instance."
  value       = try(aws_cloudwatch_log_group.market_fetcher[0].name, null)
}

output "shared_market_cache_bucket" {
  description = "S3 bucket used for shared market cache snapshots."
  value       = aws_s3_bucket.market_cache.bucket
}

output "shared_market_cache_s3_uri" {
  description = "S3 URI of the shared market cache object."
  value       = "s3://${aws_s3_bucket.market_cache.bucket}/${var.shared_market_cache_object_key}"
}

output "bot_public_ips" {
  description = "Public IP per bot."
  value = {
    for key, inst in aws_instance.bot :
    key => (var.create_public_eips ? aws_eip.bot[key].public_ip : inst.public_ip)
  }
}

output "dashboard_public_urls" {
  description = "Public dashboard URL per bot when dashboard service is enabled."
  value = var.enable_dashboard_service ? {
    for key, inst in aws_instance.bot :
    key => "http://${var.create_public_eips ? aws_eip.bot[key].public_ip : inst.public_ip}:${var.dashboard_port}"
  } : {}
}

output "dashboard_alb_dns_name" {
  description = "ALB DNS name for dashboard HTTPS endpoint."
  value       = try(aws_lb.dashboard[0].dns_name, null)
}

output "dashboard_https_url" {
  description = "Preferred HTTPS dashboard URL when ALB is enabled."
  value       = local.dashboard_alb_enabled && var.dashboard_domain_name != null ? "https://${var.dashboard_domain_name}" : null
}

output "dashboard_domain_name" {
  description = "Configured custom domain for dashboard."
  value       = var.dashboard_domain_name
}

output "bot_instance_ids" {
  description = "EC2 instance IDs by bot key."
  value = {
    for key, inst in aws_instance.bot : key => inst.id
  }
}

output "market_fetcher_instance_id" {
  description = "EC2 instance ID for dedicated market fetcher."
  value       = try(aws_instance.market_fetcher[0].id, null)
}

output "market_fetcher_public_ip" {
  description = "Public IP for dedicated market fetcher."
  value       = try(var.create_public_eips ? aws_eip.market_fetcher[0].public_ip : aws_instance.market_fetcher[0].public_ip, null)
}

output "ssm_parameter_names" {
  description = "SSM parameter names required for each bot."
  value = {
    for key, bot in local.bots : key => {
      llm_api_key      = bot.llm_param_name
      manifold_api_key = bot.manifold_param_name
    }
  }
}

output "database_url_ssm_parameter_name" {
  description = "Optional SSM parameter name used to load DATABASE_URL into all bots."
  value       = local.database_url_param_name_effective
}

output "rds_instance_id" {
  description = "RDS instance identifier when managed Postgres is enabled."
  value       = try(aws_db_instance.postgres[0].id, null)
}

output "rds_endpoint" {
  description = "RDS Postgres endpoint when managed Postgres is enabled."
  value       = try(aws_db_instance.postgres[0].address, null)
}

output "rds_port" {
  description = "RDS Postgres port when managed Postgres is enabled."
  value       = try(aws_db_instance.postgres[0].port, null)
}

output "rds_db_name" {
  description = "RDS Postgres database name when managed Postgres is enabled."
  value       = var.enable_rds_postgres ? var.rds_db_name : null
}

output "rds_username" {
  description = "RDS Postgres master username when managed Postgres is enabled."
  value       = var.enable_rds_postgres ? var.rds_username : null
}
