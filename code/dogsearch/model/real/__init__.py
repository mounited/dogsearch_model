from dogsearch.model import Model

import os
import io
from imageai.Detection import ObjectDetection
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from dogsearch.model.real.color import predict as predict_color
from dogsearch.model.real.tail import TailModel
from dogsearch.model.real.text import predict as predict_text


class RealModel(Model):
    def __init__(self):
        base_dir = os.environ["DATA"]

        self.img_size = 331

        # load models for object detection
        # RetinaNet
        self.detector_retina = ObjectDetection()
        self.detector_retina.setModelTypeAsRetinaNet()
        self.detector_retina.setModelPath(base_dir + "/resnet50_coco_best_v2.1.0.h5")
        self.custom_retina = self.detector_retina.CustomObjects(person=True, dog=True)
        self.detector_retina.loadModel()
        # YOLO
        self.detector_yolo = ObjectDetection()
        self.detector_yolo.setModelTypeAsYOLOv3()
        self.detector_yolo.setModelPath(base_dir + "/yolo.h5")
        self.custom_yolo = self.detector_yolo.CustomObjects(
            person=True, dog=True, bird=True
        )
        self.detector_yolo.loadModel()

        self.tail_model = TailModel(base_dir, self.img_size)

    def process(self, data, ext):
        with io.BytesIO(data) as f:
            image = mpimg.imread(f, format=ext)

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
        address = ""
        cam_id = ""

        # detect objects
        # RetinaNet
        _, objs_retina = self.detector_retina.detectObjectsFromImage(
            custom_objects=self.custom_retina,
            input_image=image,
            input_type="array",
            minimum_percentage_probability=5,
            output_type="array",
        )
        if len(objs_retina) > 0:
            objs_retina = pd.DataFrame(objs_retina)
            objs_retina.columns = ["name", "percentage_prob", "box_points"]
            mask_animal = (objs_retina["name"] == "dog") & (
                objs_retina["percentage_prob"] >= 7
            )
            is_animal_there_retina = 1 if objs_retina[mask_animal].shape[0] > 0 else 0
            mask_dog = (objs_retina["name"] == "dog") & (
                objs_retina["percentage_prob"] >= 9
            )
            is_it_a_dog_retina = 1 if objs_retina[mask_dog].shape[0] > 0 else 0
            mask_person = (objs_retina["name"] == "person") & (
                objs_retina["percentage_prob"] >= 52
            )
            is_the_owner_there_retina = (
                1 if objs_retina[mask_person].shape[0] > 0 else 0
            )
            objs_retina = objs_retina[mask_dog]
            objs_retina["source"] = "retina"

        # detect objects

        # YOLO
        _, objects_yolo = self.detector_yolo.detectObjectsFromImage(
            custom_objects=self.custom_yolo,
            input_image=image,
            input_type="array",
            minimum_percentage_probability=5,
            output_type="array",
        )
        if len(objects_yolo) > 0:
            objects_yolo = pd.DataFrame(objects_yolo)
            objects_yolo.columns = [
                "name",
                "perc_prob",
                "box_points",
            ]
            mask_animal = (
                (objects_yolo["name"] == "dog") & (objects_yolo["perc_prob"] >= 5)
            ) | ((objects_yolo["name"] == "bird") & (objects_yolo["perc_prob"] >= 5))
            is_animal_there_yolo = 1 if objects_yolo[mask_animal].shape[0] > 0 else 0
            mask_dog = (objects_yolo["name"] == "dog") & (
                objects_yolo["perc_prob"] >= 5
            )
            is_it_a_dog_yolo = 1 if objects_yolo[mask_dog].shape[0] > 0 else 0
            mask_person = (objects_yolo["name"] == "person") & (
                objects_yolo["perc_prob"] >= 60
            )
            is_the_owner_there_yolo = 1 if objects_yolo[mask_person].shape[0] > 0 else 0
            objects_yolo = objects_yolo[mask_dog]
            objects_yolo["source"] = "yolo"

        objects_data = pd.DataFrame()
        # concatenate objects
        if len(objs_retina) > 0 and len(objects_yolo) > 0:
            objects_data = pd.concat([objs_retina, objects_yolo])
        else:
            if len(objs_retina) > 0:
                objects_data = objs_retina
            if len(objects_yolo) > 0:
                objects_data = objects_yolo

        # is_animal_there, is_it_a_dog, is_the_owner_there aggregate
        is_animal_there = np.max([is_animal_there_retina, is_animal_there_yolo])
        if is_animal_there > 0:
            is_it_a_dog = np.max([is_it_a_dog_retina, is_it_a_dog_yolo])
            if is_it_a_dog > 0:
                is_the_owner_there = np.max(
                    [is_the_owner_there_retina, is_the_owner_there_yolo]
                )

        if len(objects_data) > 0 and is_it_a_dog > 0:
            # crop images
            objects = []
            for box in objects_data["box_points"]:
                cropped_image = image[
                    box[1] : (box[1] + box[3]), box[0] : (box[0] + box[2])
                ]
                objects.append(cropped_image)

            color = predict_color(objects)
            tail = 1 if self.tail_model.predict(objects) == "short" else 2

        cam_id, address = predict_text(image)

        res = {
            "is_animal_there": is_animal_there,
            "is_it_a_dog": is_it_a_dog,
            "is_the_owner_there": is_the_owner_there,
            "color": color,
            "tail": tail,
            "address": address,
            "cam_id": cam_id,
        }
        return res
