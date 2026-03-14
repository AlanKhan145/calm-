FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .
COPY src/ src/
COPY configs/ configs/
ENV PYTHONPATH=/app/src
CMD ["calm", "--help"]
