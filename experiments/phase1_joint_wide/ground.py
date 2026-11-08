"""Use the fixed joint evaluator with the registered wider checkpoint."""
from experiments.phase1_joint_particle import ground as joint_ground

WIDTH = 384


def main():
    joint_ground.WIDTH = WIDTH
    joint_ground.main()


if __name__ == '__main__':
    main()
