# Copyright 2022 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you (License). Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
# prior written permission.
#
# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.
import json
from pathlib import Path
from typing import Dict, Union

VERSION_FILE_PATH = Path(__file__).parent / 'version.json'


KEY_MAJOR = "major_version"
KEY_MINOR = "minor_version"
KEY_PRE_RELEASE = "pre_release"
KEY_PRE_RELEASE_NUMBER = "pre_release_number"
KEY_BUILD_ID = "build_id"
KEY_BUILD_SHA = "build_sha"


def load_version_info() -> Dict[str, Union[int, str]]:
    with open(VERSION_FILE_PATH, 'r') as f:
        version_info = json.load(f)
    return version_info


class VersionInfo:
    """
    Loads EDP version information (major, minor, optionally pre-release and build-id)
    Gets version number and build_id
    """

    def __init__(self):
        self.__version_info = load_version_info()

    def __str__(self):
        version_and_build_id = f'{self.get_version()}'
        build_id = self.__get_build_id()
        if build_id:
            version_and_build_id += f' (build {build_id})'
        return version_and_build_id

    def get_version(self) -> str:
        """
        Takes version info from version.json and extracts a string of EDP version number (e.g. '5.0' or '5.0a2')
        """
        version_str = f'{self.__version_info[KEY_MAJOR]}.{self.__version_info[KEY_MINOR]}'
        if self.__version_info[KEY_PRE_RELEASE]:
            version_str += f'{self.__version_info[KEY_PRE_RELEASE]}'
            if self.__version_info[KEY_PRE_RELEASE_NUMBER]:
                version_str += f'{self.__version_info[KEY_PRE_RELEASE_NUMBER]}'
        return version_str

    def __get_build_id(self) -> int:
        return self.__version_info[KEY_BUILD_ID]

    def __get_build_sha(self) -> int:
        return self.__version_info[KEY_BUILD_SHA]

version_info = VersionInfo()
