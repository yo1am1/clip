import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

dataset = [
    {"class": "cat", "image": cv2.imread("my_dataset/img.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_1.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_2.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_3.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_4.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_5.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_18.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_19.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_20.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_21.png")},
    {"class": "cat", "image": cv2.imread("my_dataset/img_22.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_6.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_7.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_8.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_9.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_23.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_24.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_25.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_26.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_27.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/img_28.png")},
    {"class": "car", "image": cv2.imread("my_dataset/img_10.png")},
    {"class": "car", "image": cv2.imread("my_dataset/img_11.png")},
    {"class": "car", "image": cv2.imread("my_dataset/img_12.png")},
    {"class": "car", "image": cv2.imread("my_dataset/img_13.png")},
    {"class": "car", "image": cv2.imread("my_dataset/img_14.png")},
    {"class": "car", "image": cv2.imread("my_dataset/img_15.png")},
    {"class": "car", "image": cv2.imread("my_dataset/img_17.png")},
]

common_size = (224, 224)

image_vectors = []
class_labels = []
for item in dataset:
    image = cv2.cvtColor(item["image"], cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, common_size)
    vector = resized_image.flatten().reshape(1, -1)
    image_vectors.append(vector)
    class_labels.append(item["class"])

image_vectors = np.concatenate(image_vectors, axis=0)
class_labels = np.array(class_labels)

# Building the Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn_model.fit(image_vectors)

# Test dataset
test_dataset = [
    {"class": "cat", "image": cv2.imread("my_dataset/test_image.png")},
    {"class": "dog", "image": cv2.imread("my_dataset/test_image_1.png")},
    {"class": "car", "image": cv2.imread("my_dataset/test_image_2.png")}
]

for test_item in test_dataset:
    test_image = cv2.cvtColor(test_item["image"], cv2.COLOR_BGR2GRAY)
    resized_test_image = cv2.resize(test_image, common_size)
    test_vector = resized_test_image.flatten().reshape(1, -1)

    distances, indices = knn_model.kneighbors(test_vector)

    print(f"Test Image: {test_item['class']}")
    print("Nearest Images:")
    for i, index in enumerate(indices.flatten()):
        print(f"  - Class: {class_labels[index]}, Distance: {distances.flatten()[i]}, Index: {index}")
