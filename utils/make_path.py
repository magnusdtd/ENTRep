import os
from file import File


def make_path():
  """This function will create paths to run the notebook in Kaggle"""

  # First, run the EDA.ipynb to create a cleaned dataset
  os.system('jupyter nbconvert --to notebook --execute ../EDA/EDA.ipynb --inplace')

  # Copy all repo files to the current directory
  File.copy_files(
  "/kaggle/input/ENTRep",
  "/kaggle/working/"
  )

  # Copy the Kaggle dataset to the current directoru
  File.copy_files(
    "/kaggle/input/entrep-public-dataset",
    "/kaggle/working/Dataset"
  )

  
