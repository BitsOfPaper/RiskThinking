# Use an official Python runtime as the base image
FROM python:3.9

# Create the directory
RUN mkdir -p /Stock_ETF

# Set the working directory
WORKDIR /Stock_ETF

# Copy the Python script to the working directory
COPY riskt.py .

# Install any necessary dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the Kaggle API credentials as environment variables
ENV KAGGLE_USERNAME=<your-kaggle-username>
ENV KAGGLE_KEY=<your-kaggle-api-key>

# Run the Python script
CMD ["python", "riskt.py"]
