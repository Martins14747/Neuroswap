# Neuroswap on AWS (ECS Fargate + ALB + HTTPS + S3)

This project already ships with a production Dockerfile and a `/health` endpoint. The AWS deploy path below uses:

- ECS Fargate (one container)
- Application Load Balancer (ALB) with ACM TLS cert
- S3 for durable uploads/videos/outputs
- SSM Parameter Store (or Secrets Manager) for `NEUROSWAP_API_KEY`
- CloudWatch Logs for container logs

## 1) Create S3 bucket (private)

- Bucket: `S3_BUCKET` (private)
- Recommended: enable default encryption (SSE-S3 or SSE-KMS)
- Optional but recommended: lifecycle rule to expire old `neuroswap/videos/*` and `neuroswap/outputs/*`

The app writes to:

- `neuroswap/uploads/*` (target images)
- `neuroswap/videos/*` (uploaded videos)
- `neuroswap/outputs/*` (rendered outputs)

## 2) Store API key in SSM

Create a secure string parameter:

- Name: `/neuroswap/api_key`
- Type: SecureString

## 3) IAM role for ECS task (S3 access)

- Create an IAM policy from [iam-s3-policy.json](file:///home/me/Documents/SOAIR/face-swap-web/deploy/aws/iam-s3-policy.json) (replace placeholders)
- Attach it to a task role (example name: `neuroswapTaskRole`)

The container should not use long-lived access keys. Use the task role.

## 4) Build and push image to ECR

High-level steps:

- Create an ECR repo named `neuroswap`
- Build the Docker image using the repo root `Dockerfile`
- Tag the image and push to ECR

## 5) ECS task definition

Use [ecs-taskdef.json](file:///home/me/Documents/SOAIR/face-swap-web/deploy/aws/ecs-taskdef.json) as a starting point:

- Set the ECR image URI
- Set `NEUROSWAP_S3_BUCKET`
- Set your domain in `CORS_ORIGINS`, `NEUROSWAP_PUBLIC_API_URL`, `NEUROSWAP_PUBLIC_WS_URL`
- Point the `NEUROSWAP_API_KEY` secret to your SSM parameter ARN

Notes:

- `--workers 1` is strongly recommended for this app because each worker loads large ML models.
- Increase `ephemeralStorage.sizeInGiB` if you process large videos.

## 6) ALB + HTTPS + WebSockets

- Request an ACM certificate for your domain (in the same region as the ALB)
- Create an ALB (internet-facing)
- Create a target group (HTTP) for the ECS service on port `8000`
- Configure ALB health check path: `/health`
- Create an HTTPS listener (443) forwarding to the target group

ALB settings to double-check:

- Idle timeout high enough for WebSockets and long uploads (e.g. 120s+)
- Security group: allow inbound 443 from the internet; allow ALB → ECS task SG on 8000

## 7) ECS service

- Launch type: Fargate
- Network: `awsvpc` with 2+ subnets
- Attach to ALB target group
- Desired count: start with 1

## 8) Runtime configuration summary

Minimum env/config for this AWS setup:

- `NEUROSWAP_STORAGE=s3`
- `NEUROSWAP_S3_BUCKET=...`
- `NEUROSWAP_S3_PREFIX=neuroswap`
- `NEUROSWAP_API_KEY` (via SSM/Secrets)
- `CORS_ORIGINS=https://<your-domain>`
- `NEUROSWAP_PUBLIC_API_URL=https://<your-domain>`
- `NEUROSWAP_PUBLIC_WS_URL=wss://<your-domain>`

## 9) Behavior on restart (S3 restore)

On startup, the server re-hydrates recent target faces from `neuroswap/uploads/*` in S3 (bounded by `NEUROSWAP_S3_RESTORE_LIMIT` and `MAX_RECENT_TARGETS`).
