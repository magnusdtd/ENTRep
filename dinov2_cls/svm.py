from typing import List, Callable, Dict
from sklearn import svm
import numpy as np
import torch
import cv2
from PIL import Image
import json
import zipfile
import datetime
import os

class SVM:
    def __init__(self, backbone, embedding: List, y: List, transform: Callable):
        self.clf = svm.SVC(gamma='scale')
        self.clf.fit(np.array(embedding).reshape(-1, 384), y)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    def make_submission(
            self, 
            test_file_path:str, 
            img_folder_path:str, 
            label_map: Dict[str, int],
            output_folder_path: str = './results'
        ):
        test_df = pd.read_csv(test_file_path, header=None, names=['Path'])
        predictions = {}
        for img_name in test_df['Path']:
            img_path = os.path.join(img_folder_path, img_name)
            img = self.load_img(img_path)

            with torch.no_grad():
                embedding = self.backbone(img.to(self.device))

            prediction = self.clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))
            predictions[img_name] = label_map[prediction[0]]
        
        # Generate unique JSON filename with model_name as prefix
        daytime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file_name = f'DINOv2_SVM_{daytime}.json'
        json_file_path = os.path.join(output_folder_path, json_file_name)

        # Save predictions to JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(predictions, json_file)

        # Create ZIP archive with the same name as the JSON file
        zip_file_path = os.path.join(output_folder_path, f'DINOv2_SVM_{daytime}.zip')
        with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
            zip_file.write(json_file_path, arcname=json_file_name)

        print(f"Submission file created at: {zip_file_path}")
            