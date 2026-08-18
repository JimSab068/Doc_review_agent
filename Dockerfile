# Use official Python slim image
FROM python:3.12-slim

# Create a non-root user
RUN groupadd -r kycuser && useradd -r -g kycuser kycuser

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for PyMuPDF, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn fastapi python-multipart

# Copy source code (excluding items in .dockerignore)
# pipeline.py lives at src/pipeline.py alongside primary_agent.py, vault.py,
# etc. -- it imports them via `from src.xxx`, so it needs to be in the same
# package. Adjust this if your actual repo layout differs.
COPY src/ ./src/

# Set ownership to the non-root user
RUN chown -R kycuser:kycuser /app

ENV HOME=/home/kycuser   

# Switch to non-root user
USER kycuser

# Expose standard production port
EXPOSE 8080

# Run the FastAPI application via Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]