Churn prediction
============

The project predicts churn of customers. 
Configurable variables can be defined in the constants.py .
This includes e.g.:
* Which categorical variables to encode.
* Paths to result files.
* Directories necessary for execution.

Improvements would be to refactor this code into separate classes 
which contains context objects e.g.

* Model (evaluation, prediction, fit, save)
* Datapreparer (preprocessing, feature engineering)
* Dataset (reading and writing any data)

Installation
-----------------------

Install dependencies from the requirements.txt file.


```bash
conda create --name mlops python=3.9
conda activate mlops
pip3 install -r requirements.txt
```

Running the code will be enabled with, which will be 
using the configuration defined in constants.py and runs the main.

The churn library stores logs into: ./logs/churn_library_main.log


```bash
ipython churn_library.py
```

Tests can be executed using:
One needs to disable the builtin logging for pytest beforehand.

```bash
pytest -p no:logging churn_script_logging_and_tests.py
```

The tests are logging the errors into the file stored at: ./logs/churn_librarytests.log



