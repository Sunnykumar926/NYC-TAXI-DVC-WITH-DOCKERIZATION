# Build argument and Base Images
ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# prevent python fron generating .pyc files and saving space.
ENV PYTHONDOTWRITEBYTECODE=1

# Disabel the output buffering for stdout and stderr
# Because it is important for real time monitoring.
ENV PYTHONUNBUFFERED=1

# directory inside the container where command will be executed
WORKDIR /app

# create a non-root user that the app will run under
# and it is to enhance security becoze running container
# without non-root user is dangerous

ARG UID=10001
RUN adduser \
    --disabled-password \  
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \ 
    appuser

# Caching dependencies across builds . it speed up pip install
# mount the requirements.txt directly into container
# this avoid copying files repeatedly in seprate layers, which saves build time

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt


# Ensure the application is run with non-root user
USER appuser

# moves the application files into the container's app directory
COPY ./container_models/ ./container_models/
COPY app.py .
COPY data_models.py .
COPY requirements.txt .

# Expose the port that the applications listen on means. our FastAPI app
# which runs on port 8000, will be accessible through given port inside the container.
EXPOSE 8000

# command to run the app.py when the container starts
CMD ["python", "app.py"]

