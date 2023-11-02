# Use Python 3.x as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to the container
COPY . .

# Create log/data folder
RUN mkdir -p /app/log
RUN mkdir -p /app/data

# Copy data to data folder
COPY data/*.csv /app/data

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
