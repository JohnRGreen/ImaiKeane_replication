FROM python:3.10-slim

# Install basic requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Copy the entire directory
COPY . /usr/src/app/

# Make the reproduction script executable
RUN chmod +x reproduce.sh

# The default command will run the reproduction script
CMD ["./reproduce.sh"]
