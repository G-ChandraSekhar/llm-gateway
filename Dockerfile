FROM python:3.12-slim

WORKDIR /app

# build-essential covers cases where a dependency doesn't ship a prebuilt
# wheel for this exact base image. Most of what's here (asyncpg, uvloop,
# httptools) do ship prebuilt wheels for this platform, but this is cheap
# insurance — I can't build-test this Dockerfile myself (no Docker in my
# own environment, see tasks/todo.md), so I'm erring toward robustness
# over a slightly smaller image.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
