# coding=utf-8

#python setup.py sdist build
#twine upload dist/*



from setuptools import setup
from setuptools import find_packages
from os.path import join, dirname
# We need io.open() (Python 3's default open) to specify file encodings
import io



install_requires = [
    'numpy',
    'scipy',
    'setuptools',
    'requests',
    'GitPython'
]

README="AdvBox是一款由百度安全实验室研发，在百度大范围使用的AI模型安全工具箱，目前原生支持PaddlePaddle、PyTorch、Caffe2、MxNet、" \
       "Keras以及TensorFlow平台，方便广大开发者和安全工程师可以使用自己熟悉的框架。  AdvBox同时支持GraphPipe,屏蔽了底层使用的深度学习" \
       "平台，用户可以通过几个命令就可以对PaddlePaddle、PyTorch、Caffe2、MxNet、CNTK、ScikitLearn以及TensorFlow平台生成的模型文件进行黑盒攻击。"


version="0.4.1"

tests_require = [
    'pytest',
    'pytest-cov',
]

setup(
    name="advbox",
    version=version,
    description="Python toolbox to create adversarial examples that fool neural networks",  # noqa: E501
    long_description=README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    author="baidu xlab",
    author_email="liu.yan@baidu.com",
    url="https://github.com/baidu/advbox",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires
)
