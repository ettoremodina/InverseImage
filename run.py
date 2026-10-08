"""One command from image to animation: SCA skeleton, NCA training, timeline render.

    python run.py images/jellyfish.png
    python run.py images/jellyfish.png --skip-train   # render again with the trained model
    python run.py                                     # uses target_image in config/pipeline.py

Settings stay in config/pipeline.py; the image argument only replaces target_image.
"""
import argparse
import os
import subprocess
import sys

STAGES = (
    ('SCA skeleton', ['train_sca.py']),
    ('NCA training', ['train_nca.py']),
    ('Timeline render', ['render.py', '--mode', 'timeline']),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', nargs='?', help='target image (default: target_image in config/pipeline.py)')
    parser.add_argument('--skip-train', action='store_true',
                        help='skip SCA and NCA and render with what is already in outputs/')
    args = parser.parse_args()

    env = dict(os.environ)
    if args.image:
        env['INVERSEIMAGE_TARGET'] = args.image
    stages = STAGES[2:] if args.skip_train else STAGES
    for number, (label, command) in enumerate(stages, 1):
        print(f'[{number}/{len(stages)}] {label}', flush=True)
        if subprocess.run([sys.executable, *command], env=env).returncode:
            sys.exit(f'{label} failed: {" ".join(command)}')


if __name__ == '__main__':
    main()
