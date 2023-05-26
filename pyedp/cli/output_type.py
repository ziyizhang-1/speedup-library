from pathlib import Path


class OutputInfo:
    """
    The class variable self.__output_excel stores a boolean set to True indicating if the -o/--output argument
    as given in the file_path parameter is a path to an Excel file (file may or may not already exist)
    or set to False, indicating the argument is a full, exisiting path, with no Excel file endpoint.

    The class variable self.__output_dir, is set as the path leading to the Excel file endpoint, with the Excel
    file endpoint removed from the path contained in the file_path parameter, when self.__output_excel == True;
    otherwise, self.__output_dir is simply set as the given file_path parameter.
    
    At this point, the -o argument placed in args.output_file_path will
    have already been validated to be an existing path or a path to an Excel file
    and nothing else.

    :param file_path: args.output_file_path
    """

    def __init__(self, file_path: str):
        file_path = '..' if file_path == '' else file_path
        self.__output_excel = self._set_output_excel(file_path)
        self.__output_dir = self._set_output_dir(file_path)

    @staticmethod
    def _set_output_excel(file_path) -> bool:
        if ".xlsx" in Path(file_path).suffix:
            return True
        else:
            return False

    def _set_output_dir(self, file_path) -> str:
        if self.__output_excel:
            return Path(file_path).parent
        else:
            return Path(file_path)

    @property
    def output_excel(self) -> bool:
        return self.__output_excel

    @property
    def output_path(self) -> str:
        return str(self.__output_dir)
