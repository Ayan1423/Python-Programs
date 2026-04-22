import pandas as pd

# Create Series from a list
data_list = [10, 20, 30, 40, 50]
series = pd.Series(data_list)
print("Series:\n", series)

# Create DataFrame from a dictionary
data_dict = {
    'Name': ['Ayan', 'Ali', 'Sara'],
    'Marks': [85, 90, 88],
    'Age': [20, 21, 19]
}
df = pd.DataFrame(data_dict)
print("\nDataFrame:\n", df)