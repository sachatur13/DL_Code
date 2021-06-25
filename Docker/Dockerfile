# DockerFile, Image, Container
FROM python:3.8

## This is required when deploying apps 
WORKDIR /testapp

## Copy the requirement.txt in the root directory
COPY requirements.txt .

# download the required libraries in the file
RUN pip install -r requirements.txt

## Copy the app on local machine to container file
COPY ./ ./app

## What to run in the container terminal
CMD ["python","./app/Test.py"]
