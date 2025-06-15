from utils.file import File
import os

class Local(File):
  def __init__(self):
    """
    This function will create paths to run the notebook locally. 
    """
    super().__init__()

    if not os.path.exists('pure_bioclip'):
      os.system('git clone https://huggingface.co/imageomics/bioclip pure_bioclip')
    if not os.path.exists('open_clip'):
      os.system('git clone https://github.com/mlfoundations/open_clip.git')

    current_path = os.getcwd()
    print("Current path:", current_path)
