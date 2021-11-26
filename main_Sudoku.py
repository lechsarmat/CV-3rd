import pandas as pd
import func_Sudoku as su

S1_df = pd.read_csv( "data/h3_S1.csv" )
S1 = su.DataToMatrix(S1_df).astype(int)

print('\nResult:\n')
print(su.SudokuPolymorphSolver(S1))