import os.path
import shutil
from options.train_options import TrainOptions

def delete_all_in_directory(path):
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory
            print(f"Deleted: {item_path}")
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")

def prepare(opts):

    dataset_dir = opts.dataroot
    source_domain = opts.sourcedomain
    target_domain = opts.targetdomain
    pacs_dir = os.path.join(dataset_dir, "pacs")
    train_dirs = [os.path.join(dataset_dir, f"train{l}") for l in ['A', 'B']]


    if os.path.exists(train_dirs[0]):
        shutil.rmtree(train_dirs[0])
    shutil.copytree(os.path.join(pacs_dir, source_domain), train_dirs[0])

    if os.path.exists(train_dirs[1]):
        shutil.rmtree(train_dirs[1])
    shutil.copytree(os.path.join(pacs_dir, target_domain), train_dirs[1])
