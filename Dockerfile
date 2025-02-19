# Use the official Python slim image as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the /app directory
COPY req_for_docker/requirements.txt /app/requirements.txt

# Install python dependencies
RUN pip install  -r /app/requirements.txt

# Copy only main.py(and any other necessary files) into the container
COPY main.py /app/main.py

# Expose the application port
EXPOSE 8000

# Run the application using uvicorn
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
