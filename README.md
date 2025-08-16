# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/ThermoPhase-FCSRG/perphil/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                       |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/perphil/\_\_init\_\_.py                                |        2 |        0 |    100% |           |
| src/perphil/experiments/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| src/perphil/experiments/\_tests/conftest.py                |        2 |        0 |    100% |           |
| src/perphil/experiments/\_tests/test\_iterative\_bench.py  |       15 |        0 |    100% |           |
| src/perphil/experiments/\_tests/test\_petsc\_profiling.py  |       32 |        0 |    100% |           |
| src/perphil/experiments/convergence\_2d.py                 |       76 |       76 |      0% |    16-183 |
| src/perphil/experiments/iterative\_bench.py                |      122 |       60 |     51% |144-154, 171-188, 216, 227-235, 240, 243-247, 273-287, 315-337, 357-362 |
| src/perphil/experiments/petsc\_profiling.py                |      403 |      220 |     45% |65-66, 77-78, 119-123, 153-181, 203-230, 237, 240-241, 244, 257-272, 295-296, 357-380, 392-417, 429-447, 475-486, 493, 495, 500-521, 595-608, 618, 621-623, 633, 685-692, 698-699, 715, 719, 726-737, 743-755 |
| src/perphil/experiments/petsc\_profiling\_3d.py            |      120 |      120 |      0% |     8-241 |
| src/perphil/forms/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/perphil/forms/\_tests/test\_dpp.py                     |       21 |        0 |    100% |           |
| src/perphil/forms/\_tests/test\_dpp\_regressions.py        |       15 |        0 |    100% |           |
| src/perphil/forms/\_tests/test\_spaces.py                  |       12 |        0 |    100% |           |
| src/perphil/forms/dpp.py                                   |       72 |       21 |     71% |175-205, 229 |
| src/perphil/forms/spaces.py                                |        6 |        0 |    100% |           |
| src/perphil/mesh/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| src/perphil/mesh/\_tests/test\_mesh.py                     |       12 |        0 |    100% |           |
| src/perphil/mesh/builtin.py                                |        3 |        0 |    100% |           |
| src/perphil/models/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| src/perphil/models/dpp/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| src/perphil/models/dpp/\_tests/test\_parameters.py         |       14 |        0 |    100% |           |
| src/perphil/models/dpp/parameters.py                       |       26 |        0 |    100% |           |
| src/perphil/solvers/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| src/perphil/solvers/\_tests/test\_conditioning.py          |       37 |        0 |    100% |           |
| src/perphil/solvers/\_tests/test\_solver.py                |       32 |        0 |    100% |           |
| src/perphil/solvers/\_tests/test\_solver\_parameters.py    |       18 |        0 |    100% |           |
| src/perphil/solvers/conditioning.py                        |       80 |       37 |     54% |139, 153, 156-218 |
| src/perphil/solvers/parameters.py                          |       14 |        0 |    100% |           |
| src/perphil/solvers/solver.py                              |       37 |        1 |     97% |       113 |
| src/perphil/utils/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| src/perphil/utils/\_tests/test\_manufactured\_solutions.py |       22 |        0 |    100% |           |
| src/perphil/utils/manufactured\_solutions.py               |       44 |       15 |     66% |     72-94 |
| src/perphil/utils/plotting.py                              |       22 |       22 |      0% |      1-75 |
| src/perphil/utils/postprocessing.py                        |       36 |       36 |      0% |     1-124 |
|                                                  **TOTAL** | **1295** |  **608** | **53%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/ThermoPhase-FCSRG/perphil/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/ThermoPhase-FCSRG/perphil/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ThermoPhase-FCSRG/perphil/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/ThermoPhase-FCSRG/perphil/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FThermoPhase-FCSRG%2Fperphil%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/ThermoPhase-FCSRG/perphil/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.