# CPU-only. No CUDA, no torch, no build-essential.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Hugging Face Spaces require a non-root user; UID 1000 is also the local dev UID.
RUN groupadd --gid 1000 user \
    && useradd --uid 1000 --gid user --home-dir /home/user --create-home --shell /bin/bash user

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user pyproject.toml README.md LICENSE ./
COPY --chown=user:user src ./src
RUN pip install --no-cache-dir .

COPY --chown=user:user app ./app
COPY --chown=user:user configs ./configs
COPY --chown=user:user scripts ./scripts

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PORT=7860 \
    OUT_DIR=/home/user/app/out \
    HOME=/home/user

RUN mkdir -p /home/user/app/out && chown -R user:user /home/user/app

USER 1000
EXPOSE 7860

# `docker run <image>` serves the viewer;
# `docker run <image> avdatamate run --max-samples 50` runs the pipeline instead.
CMD ["python", "app/app.py"]
