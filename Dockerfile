# Step 1: Choose Your Base Ingredient
# Use an official Python image. The "slim" version is smaller and good for production.
# Make sure the Python version matches the one you developed with (e.g., 3.10).
FROM python:3.10-slim

# Step 2: Set Up Your Workspace
# Create and set the main working directory inside the container.
WORKDIR /app

# Step 3: Add Your Ingredients, Smallest First (for caching)
# Copy only the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies.
# --no-cache-dir makes the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Add the Rest of Your Ingredients
# Copy the rest of your application's source code into the container.
# This includes your 'app' folder, 'core' folder, etc.
COPY . .

# (Optional but good practice) Tell Docker the container listens on port 8000.
EXPOSE 8000

# Step 5: Set the Startup Command
# This is the command that will run when the container starts.
# We use "--host 0.0.0.0" to make the server accessible from outside the container.
# This is CRITICAL for Docker networking to work.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]