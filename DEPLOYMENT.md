# AWS Deployment Guide

## Overview
This guide covers deploying your RAG application to AWS using ECS Fargate with persistent EFS storage for your Chroma database.

## Step 1: AWS Account Setup

### Get AWS Account
1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Follow signup process (requires credit card)
4. **Free tier available** - first 12 months include free usage

### Install AWS CLI
```bash
# macOS
brew install awscli

# Or download installer
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

### Configure AWS CLI
```bash
# Configure with your credentials
aws configure

# You'll need:
# - AWS Access Key ID (from AWS Console > IAM > Users > Security credentials)
# - AWS Secret Access Key 
# - Default region name (e.g., us-east-1)
# - Default output format (json)
```

### Create IAM User (Security Best Practice)
```bash
# Create user for deployment
aws iam create-user --user-name wells-rag-deployer

# Attach necessary policies
aws iam attach-user-policy --user-name wells-rag-deployer --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess
aws iam attach-user-policy --user-name wells-rag-deployer --policy-arn arn:aws:iam::aws:policy/AmazonElasticFileSystemFullAccess
aws iam attach-user-policy --user-name wells-rag-deployer --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
aws iam attach-user-policy --user-name wells-rag-deployer --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

# Create access keys
aws iam create-access-key --user-name wells-rag-deployer
```

## Prerequisites
- AWS account setup (above)
- AWS CLI configured
- Docker installed
- Your existing `chroma_db/` folder (66MB with 1 document)

## Local Testing First

### 1. Test with Docker Compose
```bash
# Build and run locally 
docker-compose up --build

# Test the application
curl http://localhost:8123/health
```

## AWS Infrastructure Setup

### 1. Create EFS File System
```bash
# Create EFS for persistent storage
aws efs create-file-system \
    --performance-mode generalPurpose \
    --tags Key=Name,Value=wells-rag-storage

# Note the FileSystemId (fs-xxxxxxxx)
```

### 2. Create ECR Repository
```bash
# Create container registry
aws ecr create-repository --repository-name wells-rag

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

### 3. Store Secrets in AWS Secrets Manager
```bash
# Store OpenAI API key
aws secretsmanager create-secret \
    --name "wells-rag/openai-api-key" \
    --secret-string "your-openai-api-key"

# Store LangSmith API key  
aws secretsmanager create-secret \
    --name "wells-rag/langsmith-api-key" \
    --secret-string "your-langsmith-api-key"
```

## Deploy Your Application

### 1. Build and Push Container
```bash
# Build the image
docker build -t wells-rag .

# Tag for ECR
docker tag wells-rag:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/wells-rag:latest

# Push to ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/wells-rag:latest
```

### 2. Copy Your Existing Data to EFS
```bash
# Mount EFS locally to copy your data
sudo mkdir /mnt/efs
sudo mount -t efs fs-YOUR_EFS_ID:/ /mnt/efs

# Copy your existing chroma database
sudo cp -r ./chroma_db /mnt/efs/
sudo chown -R 1000:1000 /mnt/efs/chroma_db
```

### 3. Update Task Definition
Edit `aws-ecs-task-definition.json` and replace:
- `YOUR_ACCOUNT_ID` with your AWS account ID
- `YOUR_REGION` with your AWS region
- `fs-YOUR_EFS_ID` with your EFS file system ID
- `fsap-YOUR_ACCESS_POINT_ID` with your EFS access point ID

### 4. Create ECS Service
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition.json

# Create cluster
aws ecs create-cluster --cluster-name wells-rag-cluster

# Create service
aws ecs create-service \
    --cluster wells-rag-cluster \
    --service-name wells-rag-service \
    --task-definition wells-rag-task:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

## Monitoring Your Application

### View Logs
```bash
# Stream logs
aws logs tail /ecs/wells-rag --follow
```

### Check Service Status
```bash
# Check service health
aws ecs describe-services --cluster wells-rag-cluster --services wells-rag-service
```

### Connect to Container (for debugging)
```bash
# Get task ARN
TASK_ARN=$(aws ecs list-tasks --cluster wells-rag-cluster --service-name wells-rag-service --query 'taskArns[0]' --output text)

# Connect to container
aws ecs execute-command \
    --cluster wells-rag-cluster \
    --task $TASK_ARN \
    --container wells-rag-container \
    --interactive \
    --command "/bin/bash"
```

## Cost Estimation
- **ECS Fargate (512 CPU, 1GB RAM)**: ~$15/month
- **EFS Storage (3GB)**: ~$1/month  
- **CloudWatch Logs**: ~$2/month
- **Total**: ~$18/month

## Data Management

### Backup Your Database
```bash
# Backup EFS data
aws datasync create-task --source-location-arn arn:aws:datasync:region:account:location/YOUR_EFS_LOCATION --destination-location-arn arn:aws:datasync:region:account:location/YOUR_S3_LOCATION
```

### Scale Up for More Documents
When you ingest your remaining 25-30 documents:
- Storage will grow to ~2GB (still under $1/month)
- Consider upgrading to 1GB RAM → 2GB RAM if needed (~$30/month)

## Troubleshooting

### Container Won't Start
1. Check CloudWatch logs: `/ecs/wells-rag`
2. Verify secrets are accessible
3. Ensure EFS is mounted correctly

### Performance Issues
1. Monitor CPU/Memory in CloudWatch
2. Scale up task resources if needed
3. Check EFS performance metrics

### Database Issues  
1. Verify EFS permissions: `ls -la /efs/chroma_db`
2. Check if chroma.sqlite3 file exists and is readable
3. Test database connectivity within container