
import os
from .PandaGripper import PandaGripper
from .DavidSimpleGripper import DavidSimpleGripper


def create_gripper(name, configuration=None, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    if name.lower() == 'david-simple':
        return DavidSimpleGripper(q=configuration, root_folder=root_folder)
    else:
        raise FileNotFoundError("Unknown gripper: {}".format(name))