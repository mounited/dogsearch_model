import gc
from collections import Counter
import keras
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Lambda, Input

from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inception_preprocessor,
)
from keras.applications.xception import (
    Xception,
    preprocess_input as xception_preprocessor,
)
from keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input as inc_resnet_preprocessor,
)


def most_frequent(ls):
    # most frequent element of code
    occurence_count = Counter(ls)
    return occurence_count.most_common(1)[0][0]


# FEATURE EXTRACTION OF VALIDAION AND TESTING ARRAYS
def get_valfeatures(model_name, data_preprocessor, data):
    dataset = tf.data.Dataset.from_tensor_slices(data)

    ds = dataset.batch(64)

    input_size = data.shape[1:]
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)

    base_model = model_name(
        weights="imagenet", include_top=False, input_shape=input_size
    )(preprocessor)

    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    feature_maps = feature_extractor.predict(ds, verbose=0)
    return feature_maps


def get_concat_features(feat_func, models, preprocs, array):
    feats_list = []

    for i in range(len(models)):
        feats_list.append(feat_func(models[i], preprocs[i], array))

    final_feats = np.concatenate(feats_list, axis=-1)
    del (feats_list, array)
    gc.collect()

    return final_feats


class TailModel:
    def __init__(self, base_dir, img_size):
        self.img_size = img_size
        self.trained_models = []
        for i in range(0, 3):
            model_fname = base_dir + "/breed/models/{}.h5".format(i)
            self.trained_models.append(keras.models.load_model(model_fname))

        data_df = pd.read_csv(base_dir + "/breed/breeds_labels.csv")
        self.class_names = sorted(data_df["breed"].unique())

        self.tails_dict = pd.read_csv(base_dir + "/breed/tails.csv")
        self.tails_dict = self.tails_dict.set_index("breed")

        self.models = [InceptionV3, InceptionResNetV2, Xception]
        self.preprocs = [inception_preprocessor, inc_resnet_preprocessor, xception_preprocessor]

    def predict_breed(self, objects):
        X = []
        i = 0
        for image in objects:
            orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res_image = cv2.resize(orig_image, (self.img_size, self.img_size))
            X.append(res_image)
            i += 1

        Xtesarr = np.array(X)
        del X
        gc.collect()

        # FEATURE EXTRACTION OF TEST IMAGES
        test_features = get_concat_features(get_valfeatures, self.models, self.preprocs, Xtesarr)
        del Xtesarr
        gc.collect()

        y_pred_norm = self.trained_models[0].predict(test_features, batch_size=128) / 3
        for dnn in self.trained_models[1:]:
            y_pred_norm += dnn.predict(test_features, batch_size=128) / 3

        tmp = np.argmax(y_pred_norm, axis=1)
        tmp = pd.DataFrame(tmp, columns=["Predicted class"])
        tmp["Predicted class name"] = tmp["Predicted class"].apply(
            lambda x: self.class_names[x]
        )
        return most_frequent(tmp["Predicted class name"])

    def predict(self, objects):
        breed = self.predict_breed(objects)
        return [breed, self.tails_dict.loc[breed, "tail"]]
