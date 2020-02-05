##!/usr/bin/env python3
# Copyright 2019 Travis Matthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

# This reads the __version__ variable from tensornetwork/version.py
with open('quantn/version.py') as f:
  exec(f.read(), globals())

description = ('A high level API for constructing quantum circuits using tensor networks.')

# Reading long Description from README.md file.
with open("README.md", "r") as fh:
  long_description = fh.read()

# Read in requirements
requirements = [
  requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
  name='quantn-travmatth',
  version=__version__,
  url='http://github.com/travmatth/quantn',
  author='Travis Matthews',
  author_email='trav.matth+qtn@gmail.com',
  python_requires=('>=3.6.0'),
  install_requires=requirements,
  license='Apache 2.0',
  description=description,
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
)
