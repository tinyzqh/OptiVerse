from setuptools import setup, find_packages

setup(
    name="videostreaming",
    version="0.1",
    install_requires=["gym", "numpy"],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'videostreaming': ['trace/**/*'],  # ✅ 包含所有 trace 文件夹下内容
    },
    description="A VideoStreaming Gym environment.",
    author="zhiqiang he",
    license="MIT",
)
