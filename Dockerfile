FROM debian:11.1

RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  libgl1-mesa-glx \
  tesseract-ocr \
  && rm -rf /var/apt/lists/*

ENV DATA /code/data

COPY . /code

WORKDIR /code

RUN pip3 install .

RUN mkdir /work

WORKDIR /work


ENTRYPOINT ["python3", "-m", "dogsearch.model"]
# CMD ["python3", "-m", "dogsearch.model"]
