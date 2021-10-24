from sklearn.cluster import KMeans
from collections import Counter
import gc
import numpy as np
#import tensorflow as tf
#from keras.models import Model
#from keras.layers import GlobalAveragePooling2D, Lambda, Input
import pytesseract
import cv2

## Text detection
def filter_image(image):
    # grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # inverse b/w
    invert = cv2.bitwise_not(gray)
    # apply Otsu threshold
    thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_OTSU)[1]
    return thresh

def get_contours(img):
    contours, _ = cv2.findContours(filter(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    r1, r2 = sorted(contours, key=cv2.contourArea)[-3:-1]
    x, y, w, h = cv2.boundingRect(np.r_[r1, r2])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

def tesseract(image):
    thr = filter_image(image)
    ocr = pytesseract.image_to_string(thr, lang = "eng+rus", config = "--psm 1")
    if len(ocr.splitlines()) > 2:
        ocr_lines = ocr.splitlines()
        cam_id = ocr_lines[0].split(' ', 1)[1]
        addr = ocr_lines[1] + ' ' + ocr_lines[2]
        addr = addr.lower()
        return [cam_id, addr]
    else:
        return ["", ""]


## Detect color
def detect_color(image):
    # detect color
    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    # получим высоту и ширину изображения
    (h, w) = image.shape[:2]
    # вырежем участок изображения используя срезы
    cropped = image[int( h /4):int( h * 3 /4), int( w /4):int( w * 3 /4)]
    # кластеризуем пиксели изображения
    number_of_colors = 2
    modified_image = cropped.reshape(cropped.shape[0 ] *cropped.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    # 0.5 норма, заменила на 0.7
    color = 'light' if (1 -
                (0.299 * rgb_colors[0][0] + 0.587 * rgb_colors[0][1] + 0.114 * rgb_colors[0][2]) / 255 < 0.73) else 'dark'
    return color

def most_frequent_color(color_list):
    # most frequent color
    # 0, 1 (темный), 2 (светлый), 3 (разноцветный)
    light = color_list.count('light')
    dark = color_list.count('dark')
    if light == dark:
        return 3
    if light > dark:
        return 2
    if light < dark:
        return 1


def most_frequent(List):
    # most frequent element of code
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

# Breed prediction
# FEATURE EXTRACTION OF TRAINING ARRAYS
'''
AUTO = tf.data.experimental.AUTOTUNE

def get_features(model_name, data_preprocessor, data):

    dataset = tf.data.Dataset.from_tensor_slices(data)

    def preprocess(x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.5)
        return x

    ds = dataset.map(preprocess, num_parallel_calls=AUTO).batch(64)

    input_size = data.shape[1:]
    # Prepare pipeline.
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)

    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)

    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    # Extract feature.
    feature_maps = feature_extractor.predict(ds, verbose=1)
    # print('Feature maps shape: ', feature_maps.shape)

    # deleting variables
    del (feature_extractor, base_model, preprocessor, dataset)
    gc.collect()
    return feature_maps


# FEATURE EXTRACTION OF VALIDAION AND TESTING ARRAYS
def get_valfeatures(model_name, data_preprocessor, data):
    dataset = tf.data.Dataset.from_tensor_slices(data)

    ds = dataset.batch(64)

    input_size = data.shape[1:]
    # Prepare pipeline.
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)

    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)

    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    # Extract feature.
    feature_maps = feature_extractor.predict(ds, verbose=1)
    #print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


# RETURNING CONCATENATED FEATURES USING MODELS AND PREPROCESSORS
def get_concat_features(feat_func, models, preprocs, array):
    # print(f"Beggining extraction with {feat_func.__name__}\n")
    feats_list = []

    for i in range(len(models)):
        # print(f"\nStarting feature extraction with {models[i].__name__} using {preprocs[i].__name__}\n")
        # applying the above function and storing in list
        feats_list.append(feat_func(models[i], preprocs[i], array))

    # features concatenating
    final_feats = np.concatenate(feats_list, axis=-1)
    # memory saving
    del (feats_list, array)
    gc.collect()

    return final_feats

'''