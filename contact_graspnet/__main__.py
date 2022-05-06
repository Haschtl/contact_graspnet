
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cod', '--create-object-dataset',
                        help='Create object dataset like acronym for a gripper based on the ShapeNet dataset', action='store_true')
    parser.add_argument('-cci', '--create-contact-infos',
                        help='Create contact infos for a gripper based on the acronym dataset', action='store_true')
    parser.add_argument('-ctts', '--create-table-top-scenes',
                        help='Generate scenes for a gripper with objects from the acronym dataset', action='store_true')
    parser.add_argument('-t', '--train',
                        help='Train the model', action='store_true')
    parser.add_argument('-i', '--inference',
                        help='Inference', action='store_true')
    parser.add_argument('--test',
                        help='Test script', action='store_true')
    return parser.parse_args()


def test():
    import mayavi.mlab as mlab
    from .gripper.PandaGripper import PandaGripper
    from .gripper.DavidSimpleGripper import DavidSimpleGripper
    g = DavidSimpleGripper()
    g.plot_gripper()
    mlab.show()


def create_object_dataset():
    print("""python tools/create_object_dataset.py /path/to/shapenet""")


def create_contact_infos():
    print("""python tools/create_contact_infos.py /path/to/acronym""")
    from .scene.create_contact_infos import commandline as cci
    cci()


def create_table_top_scenes():
    print("""python tools/create_table_top_scenes.py /path/to/acronym""")
    from .scene.create_table_top_scenes import commandline as ctts
    ctts()


def inference():
    print("""python contact_graspnet/inference.py \
    --np_path=test_data/*.npy \
    --local_regions --filter_grasps""")
    from .model.inference import commandline as inf
    inf()


def train():
    print("""python contact_graspnet/train.py --ckpt_dir checkpoints/your_model_name \
                                 --data_path /path/to/acronym/data""")
    from .model.train import commandline as tr
    tr()


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
