# Use official slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install system dependencies required by matplotlib and networkx
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY ./app /code/app

# Expose port
EXPOSE 8000

# Run FastAPI app with uvicorn (production mode)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]




# FROM python:3.11-slim

# # Set working directory
# WORKDIR /code

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy app source code
# COPY ./app /code/app

# # Run FastAPI app with uvicorn in reload mode
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]















# FROM python:3.11-slim

# # Set working directory
# WORKDIR /code

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy app source code
# COPY ./app /code/app

# # Run FastAPI app with uvicorn
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]







# FROM python:3.11-slim

# WORKDIR /code
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY ./app /code/app
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
