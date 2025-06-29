#!/bin/bash

echo "🤖 Starting Wells RAG Slack Bot"
echo "==============================="
echo "This will start the production Slack bot"
echo "API will be available at: http://localhost:8000"
echo ""

# Start Slack bot
docker-compose -f production/docker-compose.prod.yml up --build

echo ""
echo "Slack bot is now running!"
echo "Configure your Slack app to point to this server" 