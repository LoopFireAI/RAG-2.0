#!/bin/bash

echo "🚀 Starting Wells RAG Development Environment"
echo "============================================="
echo "This will start LangGraph Studio for development and testing"
echo "Access at: http://localhost:8123"
echo ""

# Start development environment
docker-compose -f development/docker-compose.dev.yml up --build

echo ""
echo "LangGraph Studio is now running!"
echo "Open your browser to: http://localhost:8123" 