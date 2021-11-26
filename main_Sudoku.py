import pandas as pd
import func_Sudoku as su

S1_df = pd.read_csv( "data/h3_S1.csv" )
S1 = su.DataToMatrix(S1_df).astype(int)
S2_df = pd.read_csv( "data/h3_S2.csv" )
S2 = su.DataToMatrix(S2_df).astype(int)

print('\nFirst result:\n')
su.AnswerFilter(su.SudokuPolymorphSolver(S1))

print('\nSecond result:\n')
su.AnswerFilter(su.SudokuPolymorphSolver(S2))