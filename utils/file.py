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
            items = os.listdir(source_folder)

            for item_name in items:
                source_path = os.path.join(source_folder, item_name)
                destination_path = os.path.join(destination_folder, item_name)

                if os.path.isfile(source_path):
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied file: {item_name}")
                elif os.path.isdir(source_path):
                    if not os.path.exists(destination_path):
                        os.makedirs(destination_path)
                    File.copy_files(source_path, destination_path)

            print(f"Successfully copied all files and folders recursively from {source_folder} to {destination_folder}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    @staticmethod
    def remove_directory_and_contents(directory_path):
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"The directory {directory_path} does not exist")
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"The path {directory_path} is not a directory or is a symlink")

        shutil.rmtree(directory, ignore_errors=True)
