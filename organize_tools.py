import os
import shutil

def delete_env_dir(env_dir):
    if os.path.exists(env_dir):
        shutil.rmtree(env_dir)
        print("The directory has been deleted successfully")
    else:
        print("The directory does not exist")