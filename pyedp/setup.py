#  Copyright 2022 Intel Corporation
#  This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
#  express license under which they were provided to you (License). Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
#  prior written permission.
#
#  This software and the related documents are provided as is, with no express or implied warranties, other than those
#  that are expressly stated in the License.
#
#
from distutils.core import setup

from setuptools import find_packages

from cli.version import version_info

setup(name='pedp',
      version=version_info.get_version(),
      url='',
      description='EMON Data Processing Tool',
      packages=find_packages(
          where='.',
          exclude=['dev*', 'tests*']
      ),
      scripts=['edp.py'],
      install_requires=[
          'numpy',
          'pandas',
          'defusedxml',
          'pytz',
          'tdigest',
          'xlsxwriter',
          'jsonschema'
      ]
      )
