#  Copyright 2021 Intel Corporation
#  This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
#  express license under which they were provided to you (License). Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
#  prior written permission.
#
#  This software and the related documents are provided as is, with no express or implied warranties, other than those
#  that are expressly stated in the License.
#
#
import glob
import os
from pathlib import Path

MAX_FILE_NAME_SIZE = 2000
MAX_FILE_SIZE_GB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 1024 * 1024 * 1024


class FileValidator:
    """
    Validate general file attributes (existence, size)
    """

    def __init__(self, file_must_exist=True, file_must_not_exist=False, max_file_size=0, allow_symlinks=True,
                 check_dir_only=False):
        """
        Configure the file validator

        :param file_must_exist: when True, verifies that the specified file exists
        :param file_must_not_exist: when True, verifies that the specified file does not exist
        :param max_file_size: maximum file size allowed (0 means MAX_FILE_SIZE_BYTES)
        :param allow_symlinks: when True, allow using symlinks for existing files, otherwise reject symlinks
        """
        if file_must_exist and file_must_not_exist:
            raise ValueError('file_must_exist and file_must_not_exist cannot both be True')
        if max_file_size and max_file_size < 0:
            raise ValueError('max_file_size must be greater or equal to 0')
        self.__must_exist = file_must_exist
        self.__must_not_exist = file_must_not_exist
        self.__max_file_size = max_file_size if max_file_size else MAX_FILE_SIZE_BYTES
        self.__allow_symlinks = allow_symlinks
        self.__check_dir_only = check_dir_only

    def __call__(self, file_spec):
        """
        Verify the file

        :param file_spec: file to verify. Wildcards (glob expressions) are supported
        :return: upon successful verification, returns the absolute path of file_spec
        :raise ValueError: file_spec verification failed
        """
        self._validate_path_length(file_spec)
        self._validate_path_does_not_contain_null_characters(file_spec)

        if file_spec.startswith('"') and file_spec.endswith('"'):
            file_spec = file_spec.strip('"')
        absolute_path = os.path.abspath(file_spec)
        if self.__check_dir_only:
            self._validate_xlsx_or_dir(absolute_path)
        self._validate_parent_path(absolute_path)
        if not self.__allow_symlinks and Path(absolute_path).is_symlink():
            raise ValueError(f'symbolic links not allowed: {absolute_path}')
        if self.__must_not_exist:
            self._validate_file_does_not_exist(absolute_path)
        if self.__must_exist:
            list_of_files = glob.glob(str(absolute_path))
            if len(list_of_files) == 0:
                raise ValueError(f'cannot find file(s): {absolute_path}')
            for file_path in list_of_files:
                self._validate_file_path(file_path)
        return absolute_path

    @staticmethod
    def _validate_file_does_not_exist(file_path):
        if os.path.isfile(file_path):
            raise ValueError(f'file already exists: {file_path}')
        # Verify file can be opened for writing
        try:
            with open(file_path, 'w'):
                pass
            os.remove(file_path)
        except Exception:
            raise ValueError(f'unable to write to file. Check file name and permissions: "{file_path}"')

    @staticmethod
    def _validate_xlsx_or_dir(file_path):
        file_path = '..' if file_path == '' else file_path
        if os.path.isdir(file_path) or (not os.path.isdir(file_path) and not Path(file_path).suffix):
            return
        elif '.xlsx' != Path(file_path).suffix:
            raise ValueError(f'File referenced is not an Excel file type: {file_path}')

    def _validate_parent_path(self, file_path):
        # case if path is known to be to an existing/future file with file extension
        if '.xlsx' in Path(file_path).name or not self.__check_dir_only:
            parent_dir = os.path.dirname(file_path)
        # case if path could be a path to an existing directory
        else:
            parent_dir = file_path
        if parent_dir == '':
            parent_dir = '..'
        if not os.path.isdir(parent_dir):
            raise ValueError(f'directory does not exist: {parent_dir}')

    @staticmethod
    def _validate_path_length(file_path):
        if len(file_path) > MAX_FILE_NAME_SIZE:
            raise ValueError(f'file names cannot be longer than {MAX_FILE_NAME_SIZE} characters')

    def _validate_file_size(self, file_path):
        if os.path.getsize(file_path) > self.__max_file_size:
            raise ValueError(f'input file size cannot be larger than '
                             f'{self.__max_file_size} bytes: {file_path}')

    def _validate_file_path(self, file_path):
        self._validate_path_length(file_path)
        if not os.path.isfile(file_path):
            raise ValueError(f'cannot find file(s): {file_path}')
        self._validate_file_size(file_path)
        return file_path

    @staticmethod
    def _validate_path_does_not_contain_null_characters(file_spec):
        file_spec_bytes = [b for b in file_spec.encode()]
        if file_spec_bytes.count(0) > 0:
            raise ValueError(f'illegal path name: {file_spec}')
