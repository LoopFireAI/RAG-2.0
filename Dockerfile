FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY langgraph.json ./
# NOTE: .env file removed for security - use environment variables at runtime

# Create directories for persistent data
RUN mkdir -p /app/chroma_db /app/credentials

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from src.rag_2_0.agents.rag_agent import vector_store; print('healthy')" || exit 1

# Expose port for LangGraph Studio
EXPOSE 8123

# Run the application
CMD ["python", "-m", "langgraph", "up", "--config", "langgraph.json", "--host", "0.0.0.0", "--port", "8123"]