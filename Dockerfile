# CPU image for clinic / PHC deployment (runs on a modest laptop or mini-PC, no GPU needed)
FROM python:3.11-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/srv/src \
    DR_MODEL=/srv/models/best.pt DR_DB=/srv/data/screenings.db

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY app ./app
# Mount your trained checkpoint at /srv/models/best.pt and a volume at /srv/data
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
