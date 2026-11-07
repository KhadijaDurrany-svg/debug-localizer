FROM python:3.12-slim

# Create a non-root user to run the tool.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# Install the package first (leveraging layer caching for deps).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

USER appuser

ENTRYPOINT ["faultrank"]
CMD ["bench", "--no-write"]
