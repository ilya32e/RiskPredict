FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY credit_default_labeled2.csv eda.ipynb model.pkl scaler.pkl features.pkl ./ 
COPY api ./api

EXPOSE 8000

CMD ["uvicorn", "api.model_api:app", "--host", "0.0.0.0", "--port", "8000"]

