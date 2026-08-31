FROM python:3.12-slim

# Встановлюємо системні пакети для роботи з таймзонами та медіа
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Kyiv
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
