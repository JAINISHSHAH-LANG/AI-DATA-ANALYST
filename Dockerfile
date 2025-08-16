FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY ./app /code/app

# Run FastAPI app with uvicorn in reload mode
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]















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
