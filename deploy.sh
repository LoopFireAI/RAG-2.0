#!/bin/bash

# Wells RAG 2.0 AWS ECS Deployment Script
# This script helps deploy the Wells RAG application to AWS ECS

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Wells RAG 2.0 AWS ECS Deployment Script${NC}"
echo "=========================================="

# Check if required tools are installed
command -v aws >/dev/null 2>&1 || { echo -e "${RED}Error: AWS CLI is required but not installed.${NC}" >&2; exit 1; }
command -v envsubst >/dev/null 2>&1 || { echo -e "${RED}Error: envsubst is required but not installed.${NC}" >&2; exit 1; }

# Check if environment variables are set
required_vars=("AWS_ACCOUNT_ID" "AWS_REGION" "EFS_FILE_SYSTEM_ID" "EFS_ACCESS_POINT_ID")
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo -e "${RED}Error: Environment variable $var is not set${NC}"
    echo "Please set all required environment variables:"
    echo "  export AWS_ACCOUNT_ID=123456789012"
    echo "  export AWS_REGION=us-east-1"
    echo "  export EFS_FILE_SYSTEM_ID=fs-abc123def456"
    echo "  export EFS_ACCESS_POINT_ID=fsap-123456789abcdef0"
    exit 1
  fi
done

echo -e "${GREEN}✓ Environment variables validated${NC}"

# Generate the actual task definition from template
echo "Generating ECS task definition..."
envsubst < aws-ecs-task-definition.template.json > aws-ecs-task-definition.json

echo -e "${GREEN}✓ Task definition generated${NC}"

# Build and push Docker image
echo "Building Docker image..."
docker build -t wells-rag:latest .

echo "Tagging image for ECR..."
docker tag wells-rag:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/wells-rag:latest

echo "Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/wells-rag:latest

echo -e "${GREEN}✓ Docker image built and pushed${NC}"

# Register task definition
echo "Registering ECS task definition..."
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-definition.json

echo -e "${GREEN}✓ Task definition registered${NC}"

# Update service (if it exists)
if aws ecs describe-services --cluster wells-rag-cluster --services wells-rag-service >/dev/null 2>&1; then
  echo "Updating existing ECS service..."
  aws ecs update-service --cluster wells-rag-cluster --service wells-rag-service --task-definition wells-rag-task
  echo -e "${GREEN}✓ Service updated${NC}"
else
  echo -e "${YELLOW}Note: ECS service 'wells-rag-service' not found. You may need to create it manually.${NC}"
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Verify the service is running: aws ecs describe-services --cluster wells-rag-cluster --services wells-rag-service"
echo "2. Check logs: aws logs tail /ecs/wells-rag --follow"
echo "3. Access the application via the load balancer endpoint"