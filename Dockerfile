FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib (if needed) and git for some pip installs
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command runs the script
ENTRYPOINT ["python", "llm_data_narrative3.py"]
CMD ["--help"]
