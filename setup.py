
import sys
import distutils.core

if sys.version_info[0] < 3:
    print("Mannequin requires Python 3")
    sys.exit(1)

distutils.core.setup(
    name="mannequin",
    version="2.7.2",
    description="An ultra-compact reinforcement learning framework",
    author="SquirrelInHell",
    author_email="squirrelinhell@users.noreply.github.com",
    url="https://github.com/squirrelinhell/mannequin2/",
    packages=["mannequin"],
    install_requires=[
        "gym",
        "numpy",
        "matplotlib",
        "autograd",
        "tensorflow",
        "ipython",
        "box2d-py",
        "baselines",
        ],
)
