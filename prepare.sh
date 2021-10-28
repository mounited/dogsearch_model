#! /usr/bin/env sh

python -c 'from keras.applications.inception_v3 import InceptionV3; InceptionV3(include_top=False)'
python -c 'from keras.applications.xception import Xception; Xception(include_top=False)'
python -c 'from keras.applications.inception_resnet_v2 import InceptionResNetV2; InceptionResNetV2(include_top=False)'

cd "${DATA}"
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
