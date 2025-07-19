from setuptools import setup, find_packages

setup(
    name="videostreaming",
    version="0.1",
    install_requires=["gymnasium", "numpy"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"videostreaming": ["trace/**/*", "videosize/**/*"]},
    description="A VideoStreaming Gym environment.",
    author="zhiqiang he",
    license="MIT",
)
