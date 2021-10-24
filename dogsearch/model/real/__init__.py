from dogsearch.model import Model

from pathlib import Path
from imageai.Detection import ObjectDetection
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from utils import dget_valfeatures, get_concat_features
from dogsearch.model.real.utils import detect_color, most_frequent, tesseract, most_frequent_color

# for garbage collection
import gc

# for warnings
import warnings
warnings.filterwarnings("ignore")

# utility libraries
import numpy as np
import pandas as pd
import cv2

# keras libraries
import keras

# DEFINING models and preprocessors imports
'''
from keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input

from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input

from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input

models = [InceptionV3,  InceptionResNetV2, Xception, ]
preprocs = [inception_preprocessor,  inc_resnet_preprocessor,
            xception_preprocessor, ]
'''

class RealModel(Model):
    def __init__(self):
        # load trained models for breeds
        self.trained_models_ = []
        for i in range(0, 2, 1):
            self.model_file_name = self.base_dir + '/breed/trained_model_' + str(i)
            # trained_models_.append(keras.models.load_model(model_file_name))

        self.img_size = 331
        data_df = pd.read_csv(self.base_dir + '/breed/breeds_labels.csv')
        self.class_names = sorted(data_df['breed'].unique())

        # load models for object detection
        self.base_dir = os.environ("DATA")

        # Breeds-tails dictionary
        self.tails_dict = pd.read_csv(self.base_dir + '/breed/tails.csv')
        self.tails_dict = tails_dict.set_index('breed')

        # RetinaNet
        self.detector_retina = ObjectDetection()
        self.detector_retina.setModelTypeAsRetinaNet()
        self.detector_retina.setModelPath(self.base_dir + "/resnet50_coco_best_v2.1.0.h5")
        self.custom_retina = self.detector_retina.CustomObjects(person=True, dog=True)
        self.detector_retina.loadModel()

        # YOLO
        self.detector_yolo = ObjectDetection()
        self.detector_yolo.setModelTypeAsYOLOv3()
        self.detector_yolo.setModelPath(self.base_dir + "/yolo.h5")
        self.custom_yolo = self.detector_yolo.CustomObjects(person=True, dog=True, bird=True)
        self.detector_yolo.loadModel()



    def process(self, data, ext):

        path_to_file = "/tmp/image.{}".format(ext)
        with open(path_to_file, 'wb') as f:
            f.write(data)
        image = mpimg.imread(path_to_file)

        filename = ''
        is_animal_there = 0
        is_it_a_dog = 0
        is_the_owner_there = 0
        is_animal_there_retina = 0
        is_it_a_dog_retina = 0
        is_the_owner_there_retina = 0
        is_animal_there_yolo = 0
        is_it_a_dog_yolo = 0
        is_the_owner_there_yolo = 0
        color = 0
        tail = 0
        address = ''
        cam_id = ''

        # detect objects
        # RetinaNet
        _, objects_retina = self.detector_retina.detectObjectsFromImage(custom_objects=self.custom_retina
                                                                   , input_image=image
                                                                   , input_type='array'
                                                                   , minimum_percentage_probability=5
                                                                   , output_type='array')
        if len(objects_retina) > 0:
            objects_retina = pd.DataFrame(objects_retina)
            objects_retina.columns = ['name', 'percentage_probability', 'box_points']
            mask_animal = (objects_retina['name'] == 'dog') & (objects_retina['percentage_probability'] >= 7)
            is_animal_there_retina = 1 if objects_retina[mask_animal].shape[0] > 0 else 0
            mask_dog = (objects_retina['name'] == 'dog') & (objects_retina['percentage_probability'] >= 9)
            is_it_a_dog_retina = 1 if objects_retina[mask_dog].shape[0] > 0 else 0
            mask_person = (objects_retina['name'] == 'person') & (objects_retina['percentage_probability'] >= 52)
            is_the_owner_there_retina = 1 if objects_retina[mask_person].shape[0] > 0 else 0
            objects_retina = objects_retina[mask_dog]
            objects_retina['source'] = 'retina'

        # detect objects

        # YOLO
        _, objects_yolo = self.detector_yolo.detectObjectsFromImage(custom_objects=self.custom_yolo
                                                               , input_image=image
                                                               , input_type='array'
                                                               , minimum_percentage_probability=5
                                                               , output_type='array')
        if len(objects_yolo) > 0:
            objects_yolo = pd.DataFrame(objects_yolo)
            objects_yolo.columns = ['name', 'percentage_probability', 'box_points']
            mask_animal = ((objects_yolo['name'] == 'dog') & (objects_yolo['percentage_probability'] >= 5)) | \
                          ((objects_yolo['name'] == 'bird') & (objects_yolo['percentage_probability'] >= 5))
            is_animal_there_yolo = 1 if objects_yolo[mask_animal].shape[0] > 0 else 0
            mask_dog = (objects_yolo['name'] == 'dog') & (objects_yolo['percentage_probability'] >= 5)
            is_it_a_dog_yolo = 1 if objects_yolo[mask_dog].shape[0] > 0 else 0
            mask_person = (objects_yolo['name'] == 'person') & (objects_yolo['percentage_probability'] >= 60)
            is_the_owner_there_yolo = 1 if objects_yolo[mask_person].shape[0] > 0 else 0
            objects_yolo = objects_yolo[mask_dog]
            objects_yolo['source'] = 'yolo'

        objects_data = pd.DataFrame()
        # concatenate objects
        if len(objects_retina) > 0 and len(objects_yolo) > 0:
            objects_data = pd.concat([objects_retina, objects_yolo])
        else:
            if len(objects_retina) > 0:
                objects_data = objects_retina
            if len(objects_yolo) > 0:
                objects_data = objects_yolo

        # is_animal_there, is_it_a_dog, is_the_owner_there aggregate
        is_animal_there = np.max([is_animal_there_retina, is_animal_there_yolo])
        is_it_a_dog = np.max([is_it_a_dog_retina, is_it_a_dog_yolo])
        is_the_owner_there = np.max([is_the_owner_there_retina, is_the_owner_there_yolo])

        if len(objects_data) > 0:
            # crop images
            objects = []
            for box in objects_data['box_points']:
                cropped_image = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                objects.append(cropped_image)
                # plt.imshow(cropped_image)
                # plt.show()

            # predict color
            colors = []
            for im in objects:
                colors.append(detect_color(image))
            color = most_frequent_color(colors)

            def predict_breed(objects):
                X = []
                i = 0
                for image in objects:
                    orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    res_image = cv2.resize(orig_image, (self.img_size, self.img_size))
                    X.append(res_image)
                    i += 1

                Xtesarr = np.array(X)

                del (X)
                gc.collect()

                # FEATURE EXTRACTION OF TEST IMAGES
                test_features = get_concat_features(get_valfeatures, models, preprocs, Xtesarr)

                del (Xtesarr)
                gc.collect()
                # print('Final feature maps shape', test_features.shape)

                y_pred_norm_ = trained_models_[0].predict(test_features, batch_size=128) / 3
                for dnn in trained_models_[1:]:
                    y_pred_norm_ += dnn.predict(test_features, batch_size=128) / 3

                tmp = np.argmax(y_pred_norm_, axis=1)
                tmp = pd.DataFrame(tmp, columns=['Predicted class'])
                tmp['Predicted class name'] = tmp['Predicted class'].apply(lambda x: self.class_names[x])
                breed = most_frequent(tmp['Predicted class name'])
                return breed

            # predict breed + tail (1 (короткий/нет хвоста), 2 (длинный))
            # breed = predict_breed(objects)
            # tail = self.tails_dict.loc[breed, 'tail']
            tail = 'long'

            geo = tesseract(image)
            address = geo[1]
            cam_id = geo[0]

        res = {filename, is_animal_there, is_it_a_dog, is_the_owner_there, color, tail, address, cam_id}
        return res
