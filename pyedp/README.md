# EDP - EMON Data Processing Tool
Python-based EMON Data Processing (EDP) tool

# Contents
- [Prerequisites](#prerequisites)
  - [Linux](#linux)
  - [Windows](#windows)
- [Install](#install)
- [Run](#run)
- [Known Issues and Limitations](#known-issues)

# Prerequisites <a name="prerequisites"></a>

## Linux <a name="linux"></a>

1. Install Python 3.7 (or later), pip and venv

    For example (Ubuntu):
    ```
    sudo -E apt-get update; sudo -E apt-get upgrade
    sudo -E apt-get install python3
    sudo -E apt-get install python3-pip
    sudo -E apt-get install python3-venv
    sudo -E apt-get install dos2unix
    ```

## Windows <a name="windows"></a>

1. Install python 3.7 (or later) and pip from [https://www.python.org/downloads/](https://www.python.org/downloads/).

# Install <a name="install"></a>
1. Create a virtual environment:

   ```
   python -m venv /path/to/venv
   ```
   For example:
   ```
   python -m venv ./edp-venv
   ```
   
2. Activate the virtual environment:

   Windows:
   ```
   c:\path\to\venv\scripts\activate
   ```
   
   Linux:
   ```
   source /path/to/venv/bin/activate
   ```
   
   Example (Windows):
   ```
   .\edp-venv\scripts\activate
   ```

3. Install EDP:
   ```
   python -m pip install .
   ```

# Run <a name="run"></a>

This version of EDP supports a subset of the Ruby EDP command line options. Please refer to the Known Issues and
Limitations section below for additional information.

To get usage information:
```
python edp.py --help
```

Basic usage - generate only the System views (summary and details):
```
python edp.py -i /path/to/emon.dat -m /path/to/metrics.xml -o /path/to/excel/output/summary.xlsx
```

Generate System and Socket views (summary and details):
```
python edp.py -i /path/to/emon.dat -m /path/to/metrics.xml -o /path/to/excel/output/summary.xlsx --socket-view
```

Generate Uncore Unit views (summary and details):
```
python edp.py -i /path/to/emon.dat -m /path/to/metrics.xml -o /path/to/excel/output/summary.xlsx --uncore-view
```

Generate all summary views _without_ the detail views (speeds-up processing by 10X or more):
```
python edp.py -i /path/to/emon.dat -m /path/to/metrics.xml -o /path/to/excel/output/summary.xlsx --socket-view --core-view --thread-view --no-detail-views
```

Generate all CSVs _without_ the excel file output (speeds-up processing by 2X or more):
```
python edp.py -i /path/to/emon.dat -m /path/to/metrics.xml -o /path/to/file/output --socket-view --core-view --thread-view --no-detail-views
```

Generate all summary views _without_ the detail views with multiprocessing (speeds-up processing by 15X or more):
```
python edp.py -i /path/to/emon.dat -m /path/to/metrics.xml -o /path/to/excel/output/summary.xlsx --socket-view --core-view --thread-view --no-detail-views --worker 8
# the number of workers may affect the efficiency depends on types of machines
```

# Known Issues and Limitations <a name="known-issues"></a>

1. Some EDP options (e.g., --normalize, --interval, --qpi) are not supported. 
   Use `python edp.py --help` to determine which features are currently supported.

2. (THORS-34) Installation may fail on Windows with an error message 
   similar to the following:

   > building 'accumulation_tree.accumulation_tree' extension
   >
   > error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": ...
   
   To work around this issue, Install the latest version of Visual Studio and make sure to select the C++ Build Tools
   during installation, or contact us for support.


3. To conserve memory, we're using streaming algorithms to approximate the "95th percentile" and "variation" values
   in the System Summary view. This value may not match the exact value computed by the Ruby implementation. 
   The value of the 95th percentile skews toward the maximum in some cases, but does not fall below the 
   exact 95th percentile.

4. (THORS-60) Some columns may not be generated for the detail views 
   if their corresponding events are absent in the first loop of the EMON collection (even when they appear on 
   subsequent loops). Workaround: run the EMON collection using the `--keep-all-data` flag.


-------------------------------------------------------------------------------
Intel and the Intel logo are trademarks of Intel Corporation in the U.S. and/or
other countries.

(*) Other names and brands may be claimed as the property of others.
Microsoft, Windows, and the Windows logo are trademarks, or registered 
trademarks of Microsoft Corporation in the United States and/or other countries.

Copyright 2022 Intel Corporation

This software and the related documents are Intel copyrighted materials, and
your use of them is governed by the express license under which they were
provided to you (License). Unless the License provides otherwise, you may not
use, modify, copy, publish, distribute, disclose or transmit this software or
the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.
-------------------------------------------------------------------------------
