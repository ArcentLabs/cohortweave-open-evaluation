FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash", "-lc", "pytest && python scripts/verify_frozen_results.py && python scripts/check_public_release.py"]
