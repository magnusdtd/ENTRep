from FAISS.pipeline import Pipeline
from FAISS.make_submission import make_submission
from FAISS.BioCLIP import BioCLIP_FE

def main():
  pipeline = Pipeline(
    "Dataset/train/cls.json", 
    class_feature_map = {
      "nose-right": 0,
      "nose-left": 1,
      "ear-right": 2,
      "ear-left": 3,
      "vc-open": 4,
      "vc-closed": 5,
      "throat": 6,
    },
    feature_extractor=BioCLIP_FE(
      "hf-hub:magnusdtd/bio-clip-ft"
    )
  )
  pipeline.run()

  make_submission(
    feature_extractor=BioCLIP_FE(
      "hf-hub:magnusdtd/bio-clip-ft",
      img_folder_path="Dataset/test/imgs"
    ),
    model_name="BioCLIP_FAISS",
    test_file_path="Dataset/test/i2i.csv",
  )

if __name__ == "__main__":
  main()
