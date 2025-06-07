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
    target_domain = opts.targetdomain
    pacs_dir = os.path.join(dataset_dir, "pacs")
    train_dirs = [os.path.join(dataset_dir, f"train{l}") for l in ['A', 'B', 'C', 'D', 'E']]

    pacs_in_domain_dirs = [os.path.join(pacs_dir, x) for x in os.listdir(pacs_dir) if x != target_domain]

    for i, train_dir in enumerate(train_dirs):
        print(train_dir)

        pacs_idx = i % len(pacs_in_domain_dirs)

        #clean dir and create
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)

        shutil.copytree(pacs_in_domain_dirs[pacs_idx], train_dir)
