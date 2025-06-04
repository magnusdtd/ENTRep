import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd

def visualize_img(df: pd.DataFrame):
  sample_df = df.sample(9)
  plt.figure(figsize=(12, 12))
  for idx, (_, row) in enumerate(sample_df.iterrows()):
    img_path = os.path.join('Dataset/images', row['Path'])
    img = Image.open(img_path)
    plt.subplot(3, 3, idx + 1)
    plt.imshow(img)
    plt.title(row['Classification'])
    desc = f"Type: {row['Type']}\n" + str(row['Description']).replace('\r\n', '\n')
    desc_en = str(row['DescriptionEN']).replace('\r\n', '\n')
    plt.text(0.5, -0.1, f"{desc}\n{desc_en}", 
       fontsize=9, color='black', ha='center', va='top', transform=plt.gca().transAxes, wrap=True)
    plt.axis('off')
  plt.tight_layout()
  plt.show()
