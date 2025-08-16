# AI Data Analyst 🚀

A FastAPI agent that answers data questions using CSV uploads, plotting, and simple DuckDB queries.

## Run locally
```bash
uvicorn app.main:app --reload
```

## Run in Docker
```bash
docker build -t ai-data-analyst .
docker run -p 8000:8000 ai-data-analyst
```
