To compile and run the code:
Please refer to Makefile.

Sorry for my code implementing cross validation only.

Input rating file:
Column 1: user id (variable type "int")
Column 2: item id (variable type "int")
Column 3: rating of user to item (variable type "double")

Output:
Performance 1: RMSE
Performance 2: MAE

Notes:
User id range should be in [0, ..., N - 1]
Item id range should be in [0, ..., M - 1]
N is the number of users, M is the number of items.
N, M can be automatically detected.
The code may crash or output unpredicted results if some id is out of range.
