# MatrixFactorization
The implementation of matrix factorization model

### Description
Matrix factorization (MF) model is now a very popular learning model for recommendation.
This is my own implementation of matrix factorization with cross validation only.

### Compilation & Running
Please refer to Makefile.

### Difference between C and Python Code

- Speed: C version is much faster.
- Readability: It is clearer to understand what the algorithm does in Python.
- Convergence condition: C code judges the termination of SGD with a split validation data; Python code with the original training data

### Input Rating File
- Column 1: user id (variable type "int")
- Column 2: item id (variable type "int")
- Column 3: rating of user to item (variable type "double")

### Output
- Performance 1: RMSE
- Performance 2: MAE

### Notes
- User id range should be in [0, ..., N - 1]
- Item id range should be in [0, ..., M - 1]
- N is the number of users, M is the number of items.
- N, M can be automatically detected.
- The code may crash or output unpredicted results if some id is out of range.
