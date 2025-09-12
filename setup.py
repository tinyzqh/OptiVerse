from setuptools import setup, find_packages

setup(
    name="OptiVerse",
    version="0.1",
    install_requires=["gymnasium", "numpy"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"videostreaming": ["trace/**/*", "videosize/**/*"]},
    description="A Optimization Universe Gym environment.",
    author="Zhiqiang He",
    license="MIT",
)
