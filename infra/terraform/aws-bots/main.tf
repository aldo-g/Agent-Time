data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_subnet" "default" {
  for_each = toset(data.aws_subnets.default.ids)
  id       = each.value
}

data "aws_ec2_instance_type_offerings" "by_type" {
  for_each = toset(
    distinct(
      concat(
        [for _, bot in var.bots : coalesce(try(bot.instance_type, null), var.default_instance_type)],
        var.enable_shared_market_cache && var.create_market_fetcher_instance ? [var.market_fetcher_instance_type] : []
      )
    )
  )

  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [each.value]
  }
}

data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64"
}

locals {
  name_prefix                   = "${var.project_name}-${var.environment}"
  subnet_ids                    = sort(data.aws_subnets.default.ids)
  bot_keys                      = sort(keys(var.bots))
  shared_market_fetcher_enabled = var.enable_shared_market_cache && var.create_market_fetcher_instance
  shared_market_cache_bucket_name = lower(
    replace(
      "${local.name_prefix}-${data.aws_caller_identity.current.account_id}-${var.aws_region}-market-cache",
      "_",
      "-"
    )
  )

  ssm_path_prefix = "/${var.project_name}/${var.environment}"
  ssm_path_arn    = trim(local.ssm_path_prefix, "/")

  bots = {
    for key, bot in var.bots : key => merge(bot, {
      instance_type       = coalesce(try(bot.instance_type, null), var.default_instance_type)
      schedule            = coalesce(try(bot.schedule, null), var.default_schedule)
      llm_param_name      = "${local.ssm_path_prefix}/${key}/${bot.llm_env_var}"
      manifold_param_name = "${local.ssm_path_prefix}/${key}/${bot.manifold_env_var}"
    })
  }

  compatible_subnet_ids_by_type = {
    for instance_type, offerings in data.aws_ec2_instance_type_offerings.by_type :
    instance_type => [
      for subnet_id in local.subnet_ids : subnet_id
      if contains(offerings.locations, data.aws_subnet.default[subnet_id].availability_zone)
    ]
  }
}

resource "aws_ecr_repository" "bots" {
  name                 = "${local.name_prefix}-bots"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_ecr_lifecycle_policy" "bots" {
  repository = aws_ecr_repository.bots.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "bots" {
  for_each          = local.bots
  name              = "/${var.project_name}/${var.environment}/bots/${each.key}"
  retention_in_days = 14

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Bot         = each.key
  }
}

resource "aws_cloudwatch_log_group" "market_fetcher" {
  count             = local.shared_market_fetcher_enabled ? 1 : 0
  name              = "/${var.project_name}/${var.environment}/market-fetcher"
  retention_in_days = 14

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Bot         = "market-fetcher"
  }
}

resource "aws_s3_bucket" "market_cache" {
  bucket        = local.shared_market_cache_bucket_name
  force_destroy = var.shared_market_cache_force_destroy

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "shared-market-cache"
  }
}

resource "aws_s3_bucket_public_access_block" "market_cache" {
  bucket = aws_s3_bucket.market_cache.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "market_cache" {
  bucket = aws_s3_bucket.market_cache.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_security_group" "bots" {
  name        = "${local.name_prefix}-bots-sg"
  description = "Security group for Agent-Time cloud bots"
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = var.ssh_ingress_cidr == null ? [] : [var.ssh_ingress_cidr]
    content {
      description = "Optional SSH access"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role" "bot_instance" {
  name = "${local.name_prefix}-bot-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.bot_instance.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "bot_access" {
  name = "${local.name_prefix}-bot-access"
  role = aws_iam_role.bot_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadBotParameters"
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/${local.ssm_path_arn}/*"
      },
      {
        Sid      = "DecryptSecureString"
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "ssm.${var.aws_region}.amazonaws.com"
          }
        }
      },
      {
        Sid      = "ECRToken"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer"
        ]
        Resource = aws_ecr_repository.bots.arn
      },
      {
        Sid    = "CloudWatchLogsWrite"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = concat(
          [for _, lg in aws_cloudwatch_log_group.bots : "${lg.arn}:*"],
          local.shared_market_fetcher_enabled ? ["${aws_cloudwatch_log_group.market_fetcher[0].arn}:*"] : []
        )
      },
      {
        Sid    = "SharedMarketCacheReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.market_cache.arn}/${var.shared_market_cache_object_key}"
      },
      {
        Sid    = "SharedMarketCacheList"
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.market_cache.arn
        Condition = {
          StringLike = {
            "s3:prefix" = [var.shared_market_cache_object_key]
          }
        }
      }
    ]
  })
}

resource "aws_iam_instance_profile" "bot_instance" {
  name = "${local.name_prefix}-bot-instance-profile"
  role = aws_iam_role.bot_instance.name
}

resource "aws_instance" "bot" {
  for_each = local.bots

  ami                         = data.aws_ssm_parameter.al2023_ami.value
  instance_type               = each.value.instance_type
  user_data_replace_on_change = true
  subnet_id = element(
    local.compatible_subnet_ids_by_type[each.value.instance_type],
    index(local.bot_keys, each.key) % length(local.compatible_subnet_ids_by_type[each.value.instance_type])
  )
  vpc_security_group_ids      = [aws_security_group.bots.id]
  iam_instance_profile        = aws_iam_instance_profile.bot_instance.name
  associate_public_ip_address = true

  lifecycle {
    precondition {
      condition     = length(local.compatible_subnet_ids_by_type[each.value.instance_type]) > 0
      error_message = "No default subnet supports instance type '${each.value.instance_type}' in region '${var.aws_region}'."
    }
  }

  root_block_device {
    volume_size = var.root_volume_size_gb
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tmpl", {
    aws_region                     = var.aws_region
    bot_key                        = each.key
    agent_name                     = each.value.agent_name
    expected_wallet                = each.value.expected_wallet
    llm_env_var                    = each.value.llm_env_var
    manifold_env_var               = each.value.manifold_env_var
    llm_param_name                 = each.value.llm_param_name
    manifold_param_name            = each.value.manifold_param_name
    ecr_image_uri                  = "${aws_ecr_repository.bots.repository_url}:${var.image_tag}"
    ecr_registry                   = split("/", aws_ecr_repository.bots.repository_url)[0]
    log_group_name                 = aws_cloudwatch_log_group.bots[each.key].name
    log_stream_name                = each.key
    market_limit                   = var.market_limit
    max_attempts                   = var.max_attempts
    enable_shared_market_cache     = var.enable_shared_market_cache ? "true" : "false"
    shared_market_cache_bucket     = aws_s3_bucket.market_cache.bucket
    shared_market_cache_object_key = var.shared_market_cache_object_key
    market_cache_wait_seconds      = var.market_cache_wait_seconds
    market_cache_max_age_seconds   = var.market_cache_max_age_seconds
    enable_timers                  = var.enable_timers ? "true" : "false"
    run_service_once_on_boot       = var.run_service_once_on_boot ? "true" : "false"
    schedule                       = each.value.schedule
    common_env                     = var.common_env
  })

  tags = {
    Name        = "${local.name_prefix}-${each.key}"
    Project     = var.project_name
    Environment = var.environment
    Bot         = each.key
  }
}

resource "aws_instance" "market_fetcher" {
  count = local.shared_market_fetcher_enabled ? 1 : 0

  ami                         = data.aws_ssm_parameter.al2023_ami.value
  instance_type               = var.market_fetcher_instance_type
  user_data_replace_on_change = true
  subnet_id = element(
    local.compatible_subnet_ids_by_type[var.market_fetcher_instance_type],
    0
  )
  vpc_security_group_ids      = [aws_security_group.bots.id]
  iam_instance_profile        = aws_iam_instance_profile.bot_instance.name
  associate_public_ip_address = true

  lifecycle {
    precondition {
      condition     = length(local.compatible_subnet_ids_by_type[var.market_fetcher_instance_type]) > 0
      error_message = "No default subnet supports market_fetcher_instance_type '${var.market_fetcher_instance_type}' in region '${var.aws_region}'."
    }
  }

  root_block_device {
    volume_size = var.root_volume_size_gb
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data_market_fetcher.sh.tmpl", {
    aws_region                     = var.aws_region
    ecr_image_uri                  = "${aws_ecr_repository.bots.repository_url}:${var.image_tag}"
    ecr_registry                   = split("/", aws_ecr_repository.bots.repository_url)[0]
    log_group_name                 = aws_cloudwatch_log_group.market_fetcher[0].name
    log_stream_name                = "market-fetcher"
    market_limit                   = var.market_limit
    shared_market_cache_bucket     = aws_s3_bucket.market_cache.bucket
    shared_market_cache_object_key = var.shared_market_cache_object_key
    enable_timers                  = var.enable_timers ? "true" : "false"
    run_service_once_on_boot       = var.market_fetcher_run_service_once_on_boot ? "true" : "false"
    schedule                       = var.market_fetcher_schedule
  })

  tags = {
    Name        = "${local.name_prefix}-market-fetcher"
    Project     = var.project_name
    Environment = var.environment
    Bot         = "market-fetcher"
  }
}

resource "aws_eip" "bot" {
  for_each = var.create_public_eips ? aws_instance.bot : {}

  instance = each.value.id
  domain   = "vpc"

  tags = {
    Name        = "${local.name_prefix}-${each.key}-eip"
    Project     = var.project_name
    Environment = var.environment
    Bot         = each.key
  }
}

resource "aws_eip" "market_fetcher" {
  count = var.create_public_eips && local.shared_market_fetcher_enabled ? 1 : 0

  instance = aws_instance.market_fetcher[0].id
  domain   = "vpc"

  tags = {
    Name        = "${local.name_prefix}-market-fetcher-eip"
    Project     = var.project_name
    Environment = var.environment
    Bot         = "market-fetcher"
  }
}
