FROM python:3.11-slim

WORKDIR /code/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
