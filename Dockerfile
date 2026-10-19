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

# Fail the build if a required service was omitted from the repository/context.
RUN python -m compileall -q app main.py \
    && python -c "from importlib.util import find_spec; required = ('app.services.summarizer', 'app.services.text_quality', 'app.services.quality_audit'); missing = [name for name in required if find_spec(name) is None]; assert not missing, f'Missing required modules: {missing}'"

CMD ["python", "main.py"]
