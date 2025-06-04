import os
import shutil
from pathlib import Path


class File:
    @staticmethod
    def list_files_in_directory(directory):
        try:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"The directory {directory} does not exist.")
            
            files = os.listdir(directory)
            if not files:
                print("The directory is empty.")
            else:
                print("Files in the directory:")
                for file_name in files:
                    print(file_name)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    @staticmethod
    def copy_files(source_folder, destination_folder):
        try:
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            files = os.listdir(source_folder)
        
            for file_name in files:
                source_path = os.path.join(source_folder, file_name)
                destination_path = os.path.join(destination_folder, file_name)
                
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied: {file_name}")
                
            print(f"Successfully copied all files from {source_folder} to {destination_folder}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    @staticmethod
    def remove_directory_and_contents(directory_path):
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"The directory {directory_path} does not exist")
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"The path {directory_path} is not a directory or is a symlink")

        shutil.rmtree(directory)

    @staticmethod
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

  