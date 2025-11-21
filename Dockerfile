# Multi-stage build for RAG Agent
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
#COPY test_a2a_client.py .
COPY run_tests.py .

# Create non-root user for security
RUN useradd -m -u 1000 agent && \
    chown -R agent:agent /app

USER agent

# Expose port
EXPOSE 9998

# Health check
#HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#    CMD curl -f http://localhost:9998/.well-known/agent-card.json || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# Run the agent
CMD ["python", "-m", "src.agent"]
