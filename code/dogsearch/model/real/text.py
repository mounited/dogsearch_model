import pytesseract
import cv2
import pandas as pd
import numpy as np
from Levenshtein import distance as ldistance

THR = 2


class TextModel:
    def __init__(self, base_dir):
        path = "{}/cams_adr.csv".format(base_dir)
        cam_addr_dict = pd.read_csv(path)
        self.cam_ids = list(cam_addr_dict["id_dict"])
        self.addresses = list(cam_addr_dict["address_dict_cleaned"])

    def predict(self, image):
        def filter_image(image):
            # grayscale image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # inverse b/w
            invert = cv2.bitwise_not(gray)
            # apply Otsu threshold
            thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_OTSU)[1]
            return thresh

        def cam_id_to_address(cam_id):
            cam_id_clean = cam_id.translate({" ": "_", ".": "_"})
            distances = [ldistance(cam_id_clean, id) for id in self.cam_ids]
            idx = np.argmin(distances)
            distance = distances[idx]
            if distance > THR:
                return None
            return self.cam_ids[idx], self.addresses[idx]

        thr = filter_image(image)
        ocr = pytesseract.image_to_string(thr, lang="eng+rus", config="--psm 1")
        ocr_lines = ocr.splitlines()
        if len(ocr_lines) > 2:
            cam_tokens = ocr_lines[0].split(" ", 1)
            if len(cam_tokens) > 1:
                cam_id = cam_tokens[1]
            else:
                cam_id = ""
            addr = ocr_lines[1] + " " + ocr_lines[2]
            addr = addr.lower()

            def clean_cam_line(line):
                line = line.replace(",", " ")
                line = line.replace("  ", " ")
                return line if len(line) > 0 else " "
            cam_id = clean_cam_line(cam_id)

            if cam_id != " ":
                r = cam_id_to_address(cam_id)
                if r is not None:
                    cam_id = r[0]
                    addr = r[1]

            addr = clean_cam_line(addr)

            return [cam_id, addr]
        else:
            return [" ", " "]
