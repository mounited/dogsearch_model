# DogSearch Model

## Usage

```
python -m dogsearch.model random /path_to_image/image.jpg
```

## Docker

```
docker build -t dogsearch_model .
docker run --rm -it -v/path_to_image:/work random image.jpg
```
