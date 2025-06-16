from typing import List, Callable
from sklearn import svm
import numpy as np
import torch
import cv2
from PIL import Image

class SVM:
    def __init__(self, backbone, embedding: List, y: List, transform: Callable):
        self.clf = svm.SVC(gamma='scale')
        self.clf.fit(np.array(embedding).reshape(-1, 384), y)
        self.device = "cuda" if torch.is_available() else "cpu"
        self.backbone = backbone
        self.transform = transform

    def load_img(self, img_path:str) -> torch.Tensor:
        img = Image.open(img_path)

        transformed_img = self.transform(img)[:3].unsqueeze(0)

        return transformed_img

    def predict(self, img_path:str):
        import matplotlib.pyplot as plt

        img = self.load_img(img_path)

        with torch.no_grad():
            embedding = self.backbone(img.to(self.device))

        prediction = self.clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

        print("\nPredicted class: " + prediction[0])
