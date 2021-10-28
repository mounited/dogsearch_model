FROM debian:10.11

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-rus \
    wget \
    && rm -rf /var/apt/lists/*

ENV VIRTUAL_ENV /venv
RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --upgrade pip wheel

ENV TF_CPP_MIN_LOG_LEVEL=3

COPY code /code

RUN pip install /code && rm -rf /code

COPY data /data

ENV DATA /data

COPY prepare.sh /

RUN /prepare.sh

RUN mkdir /work

WORKDIR /work

ENTRYPOINT ["python", "-m", "dogsearch.model"]
