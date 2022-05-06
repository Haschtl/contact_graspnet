import mayavi.mlab as mlab
from contact_graspnet.gripper.PandaGripper import PandaGripper
from contact_graspnet.gripper.DavidSimpleGripper import DavidSimpleGripper


if __name__ == "__main__":
    g = DavidSimpleGripper()
    g.plot_gripper()
    mlab.show()
