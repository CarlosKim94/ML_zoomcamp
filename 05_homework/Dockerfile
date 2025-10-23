FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /app
COPY ["pyproject.toml", "uv.lock", "./"]

RUN uv sync

COPY ["predict.py", "pipeline_v1.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "uvicorn", "--bind=0.0.0.0:9696", "predict:app"]