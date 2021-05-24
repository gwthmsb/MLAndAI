# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:17:27 2021

Dataset: https://www.kaggle.com/datasnaek/chess

@author: gowthas
"""

import pandas as pd
import numpy
from math import nan
"""
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html

1) There are two core objects in Pandas
    DataFrame: It is a table. Constructor takes dictionary. Keys are columns
    Series: It is a list(sequence of data values)
    
    We can think: DataFrame is set of series glued together
2) Reading from file
3) Indexing, selecting and assigning
4) Summary functions and maps
5) Grouping and sorting
6) Multi indexing
"""

d1 = pd.DataFrame({'column1': [10, 11], 'column2': [100, 101]})
print(d1)
print(d1['column1'])

# We can provide row names by using index 
d2 = pd.DataFrame({'column1': [10, 11], 'column2': [100, 101]}, index=['row1', 'row2'])
print(d2)
print(d2['column1']['row1'])


s1 = pd.Series([1, 2, 3, 4])
print(s1)
print(s1[0])

# Assigning row names using index, name by name
s2 = pd.Series([1, 2, 3, 4], index=[1, 2, 3, 4], name='Some series')
print(s2)


"""
Reading data files. This create DataFrame object

"""
chess_data = pd.read_csv('D:/docs/GITLab/ML & AI/datasets/chess/games.csv')
# chess_data = pd.read_csv('D:/docs/GITLab/ML & AI/datasets/chess/games.csv', index_col=0)
chess_data.shape
chess_data.size
chess_data.head(1)
chess_data.reset_index(inplace=True)
chess_data.to_json('D:/docs/GITLab/ML & AI/datasets/chess/games.json')


"""
3) Indexing, selecting and assigning

Columns from dataframe objects can be accessed by name
Eg: d.column1

"""
print(d2)
print(d2.column1)
print(d2.column2)

"""
Indexing: Pandas have their own way accessor operator- loc, iloc
    Index based selection: Selecting data based on its numerical position in the data - iloc
    loc and iloc are row-first, column-second(Opposite of what we do in python)
    
    Label based selection: Selecting data based on its data index value not the position
    
    iloc treats the data as big matrix(list of list)
    loc uses information in the indices
    
    iloc uses the Python stdlib indexing scheme, where the first element of the range 
    is included and the last one excluded. 
    So 0:10 will select entries 0,...,9. 
    loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10

"""
print(chess_data.iloc[0])
print(chess_data.iloc[:, 0])
print(len(numpy.unique(chess_data.iloc[:, 0])))
print(chess_data.iloc[0, 0], chess_data.iloc[0, 1])
print(chess_data.iloc[[0, 1, 2, 3], [5, 6, 14]])
print(chess_data.iloc[:, [5, 6, 14]])
print(chess_data.iloc[:])

print(chess_data.loc[:5, ['winner', 'opening_name']])
x = chess_data.loc[:5, ['winner', 'opening_name']]
y = chess_data.iloc[:, [2]]
z = chess_data['winner']
print(z)
row = chess_data[:][1:2]
print(row)
print(y)
print(x)
print(x['winner'])

print(chess_data.loc[(chess_data.winner == 'white')])
print(chess_data.loc[(chess_data.winner == 'white') | (chess_data.white_rating > 2000)])
print(chess_data.loc[chess_data.opening_name.isin(["Queen's Pawn Game"])])
print(chess_data.boxplot(column=['white_rating', 'black_rating']))
print(chess_data.max('white_rating'))

chess_data[0][:]


"""
4) Summary functions and maps

Data does not always come out of memory in the format we want it in right out of the bat.
Sometimes we have to do some more work ourselves to reformat it for the task at hand. 


"""

print(chess_data.winner.describe())
print(chess_data.turns.describe())

print(chess_data.turns.mean())

print(numpy.unique(chess_data.winner))
print(chess_data.winner.unique())

print(chess_data.winner.value_counts())

mean_value = chess_data.turns.mean()
print(chess_data.turns)
chess_data.turns.map(lambda p: p - mean_value)
a_c = chess_data.turns.map(lambda p: p - mean_value)
print(chess_data.turns)
print(a_c)

print(chess_data.loc[:, ['black_rating', 'white_rating']])

div_series = chess_data.black_rating.divide(chess_data.white_rating)
max_s = div_series.idxmax()
print(chess_data.loc[5373, ['id']])

chess_data['id'][5373]


def opening(row):
    if row.opening_name == 'Philidor Defense':
        row.turns = 3
    else:
        if row.turns >= 95:
            row.turns = 3
        elif row.turns <95 and row.turns >=85:
            row.turns = 2
        else:
            row.turns = 1
    return row

rr = chess_data.apply(opening, axis='columns')

star_ratings = rr['turns']
print(star_ratings)



"""
5) Grouping and sorting
"""

print(chess_data.groupby('turns').turns.count())
print(chess_data.winner.value_counts())
print(chess_data.groupby('winner').winner.count())
print(chess_data.groupby('opening_name').apply(lambda row: row.winner.iloc[0]))
print(chess_data.groupby(['opening_name']).white_rating.agg([len, min, max]))


"""
    A multi-index differs from a regular index in that it has multiple levels
    Multi-indices have several methods for dealing with their tiered structure 
    which are absent for single-level indices. 
    They also require two levels of labels to retrieve a value
    
    GroupBy values and aggregate value should be different for applying reset_index
"""

index = chess_data.groupby(['opening_name', 'turns']).black_rating.count()
reset_index = index.reset_index()
print(reset_index)
print(reset_index.sort_values(by='turns'))



"""
6) Data types and missing values
    
    Dtype: Data type for a column in series
    
    One peculiarity to keep in mind (and on display very clearly here) is that
    columns consisting entirely of strings do not get their own type; 
    they are instead given the object type
    
    NaN values are always of the float64 dtype.
    
    Replacing NaN values using fillna("value")

"""
print(chess_data.turns.dtype)
print(chess_data.opening_name.dtype)
print(chess_data.dtypes)

print(chess_data.turns.astype('float64'))
chess_data.opening_name = nan
print(chess_data.opening_name)
print(pd.isnull(chess_data.opening_name))
print(pd.isna(chess_data.opening_name))
print(chess_data.opening_name.fillna("some random opening"))
print(chess_data.opening_name.replace(nan, "some value"))



"""
7) Renaming and combining
    
    Rename using "column" or "index"(rows)
    set_index()
    rename_axis - changing names of row and column headers
    
Pandas has following combine methods

    concat: Given a list of elements, this function will smush those elements together along an axis.
    join: combine different DataFrame objects which have an index in common
    merge: 
"""

chess_data.rename(columns = {'turns': 'renamed_column_turns'}).dtypes
chess_data.rename(index= {1:'index1', 2: 'index2'})
chess_data.rename_axis('rows', axis='rows').rename_axis('columns', axis='columns')


