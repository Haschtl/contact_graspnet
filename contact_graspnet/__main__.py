
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-object-dataset',
                        help='Create object dataset like acronym for a gripper based on the ShapeNet dataset')
    parser.add_argument('--create-contact-infos',
                        help='Create contact infos for a gripper based on the acronym dataset')
    parser.add_argument('--create-table-top-scenes',
                        help='Generate scenes for a gripper with objects from the acronym dataset')
    parser.add_argument('--train',
                        help='Train the model')
    parser.add_argument('--inference',
                        help='Inference')
    parser.add_argument('--test',
                        help='Test script')
    return parser.parse_args()


def test():
    import mayavi.mlab as mlab
    from .gripper.PandaGripper import PandaGripper
    from .gripper.DavidSimpleGripper import DavidSimpleGripper
    g = DavidSimpleGripper()
    g.plot_gripper()
    mlab.show()


def create_object_dataset():
    print("""python tools/create_object?dataset.py /path/to/shapenet""")

def create_contact_infos():
    print("""python tools/create_contact_infos.py /path/to/acronym""")

def create_table_top_scenes():
    print("""python tools/create_table_top_scenes.py /path/to/acronym""")

def inference():
    print("""python contact_graspnet/inference.py \
    --np_path=test_data/*.npy \
    --local_regions --filter_grasps""")

def train():
    print("""python contact_graspnet/train.py --ckpt_dir checkpoints/your_model_name \
                                 --data_path /path/to/acronym/data""")
if __name__ == "__main__":
    args = parse_args()
    if args.inference:
        inference()
    elif args.train:
        train()
    elif args.create_table_top_scenes:
        create_table_top_scenes()
    elif args.create_contact_infos:
        create_contact_infos()
    elif args.create_object_dataset:
        create_object_dataset()
    elif args.test:
        test()
    else:
        print("No argument provided. Enter `-h` for help.")
