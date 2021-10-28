from sklearn.cluster import KMeans
from collections import Counter


# Detect color
def detect_color(image):
    # detect color
    def RGB2HEX(color):
        c0 = int(color[0])
        c1 = int(color[1])
        c2 = int(color[2])
        return "#{:02x}{:02x}{:02x}".format(c0, c1, c2)

    # получим высоту и ширину изображения
    (h, w) = image.shape[:2]
    # вырежем участок изображения используя срезы
    cropped = image[int(h / 4) : int(h * 3 / 4), int(w / 4) : int(w * 3 / 4)]
    # кластеризуем пиксели изображения
    number_of_colors = 2
    modified_image = cropped.reshape(cropped.shape[0] * cropped.shape[1], 3)
    clf = KMeans(n_clusters=number_of_colors, random_state=7)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    # 0.5 норма, заменила на 0.7
    color = (
        "light"
        if (
            1
            - (
                0.299 * rgb_colors[0][0]
                + 0.587 * rgb_colors[0][1]
                + 0.114 * rgb_colors[0][2]
            )
            / 255
            < 0.73
        )
        else "dark"
    )
    return color


def most_frequent_color(color_list):
    # most frequent color
    # 0, 1 (темный), 2 (светлый), 3 (разноцветный)
    light = color_list.count("light")
    dark = color_list.count("dark")
    if light == dark:
        return 3
    if light > dark:
        return 2
    if light < dark:
        return 1


def predict(objects):
    colors = []
    for image in objects:
        colors.append(detect_color(image))
    return most_frequent_color(colors)
