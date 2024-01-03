# Define environment (alpine for lightweight image)
#FROM python:3.11-alpine
FROM python:3.10-slim-bullseye


# Set the working directory
WORKDIR "/semantic-anchoring-memory"

# Copy the current directory contents into the container work
# directory excluding any files or directories that match the patterns
# specified in .dockerignore
COPY . "/semantic-anchoring-memory"

# RUN apk update && apk add python3-dev \
#                         gcc \
#                         libc-dev

# RUN apk update
# RUN apk add make automake gcc g++ subversion python3-dev

# Install necessary dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Launch the gunicorn server and bind it to the right port
CMD ["uvicorn", "main:app"]
