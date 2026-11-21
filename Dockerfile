FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY . .

# Mount a Railway Volume here and set DATABASE_PATH=/data/senjougahara.sqlite3.
RUN mkdir -p /data

CMD ["python", "bot.py"]
