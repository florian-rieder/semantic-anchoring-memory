# Define environment
# bookworm for compatibility with chromadb: https://docs.trychroma.com/troubleshooting#sqlite
FROM python:3.11-slim-bookworm

# Set the working directory
WORKDIR "/semantic-anchoring-memory"

# Copy the current directory contents into the container work
# directory excluding any files or directories that match the patterns
# specified in .dockerignore
COPY . "/semantic-anchoring-memory"

# Install poetry
RUN pip install poetry

# Use poetry to install dependencies
RUN poetry install --no-root

# Expose port
EXPOSE 8000


# Launch the uvicorn server
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
