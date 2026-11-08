"""Run the fixed joint source trainer with the registered wider network."""
from experiments.phase1_joint_particle import train as joint_train

WIDTH = 384


def main():
    joint_train.WIDTH = WIDTH
    joint_train.main()


if __name__ == '__main__':
    main()
