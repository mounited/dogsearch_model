import pytesseract
import cv2


# Text detection
def filter_image(image):
    # grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # inverse b/w
    invert = cv2.bitwise_not(gray)
    # apply Otsu threshold
    thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_OTSU)[1]
    return thresh


def predict(image):
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
        return [cam_id, addr]
    else:
        return ["", ""]
