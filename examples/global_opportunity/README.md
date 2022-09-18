# Global Investment Opportunity with Intelligent Asset Allocator (with Qlib)

## Install Qlib

- Run below command to install `qlib from local src`

  ```
  pip install ../../
  ```
#####  or
- Run below command to install `qlib from repository`

  ```
  pip install pyqlib
  ```

## Data Preparation

- Run below command to `download the data from Yahoo Finance`

  ```
  python ../../scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us --delete_old False --exists_skip True
  ```