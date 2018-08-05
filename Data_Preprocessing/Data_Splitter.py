import pandas as pd
from datetime import datetime
import csv

df = pd.read_csv("15to17data.csv")

columns = df.columns.values.tolist()

with open('0to8.csv', 'w') as outcsv0to8:
    writer0to8 = csv.writer(outcsv0to8)
    with open('9to17.csv', 'w') as outcsv0to8:
        writer9to17 = csv.writer(outcsv0to8)
        with open('18to24.csv', 'w') as outcsv0to8:
            writer18to24 = csv.writer(outcsv0to8)
            for index, row in df.iterrows():
                datetime = row['Datetime']
                split = datetime.split(" ",2)
                hour = split[1]
                hour = hour.split(":",2)
                hour_int = int(hour[0])
                if(hour_int in range(0,8)):
                    writer0to8.writerow(row)
                elif(hour_int in range(9,17)):
                    writer9to17.writerow(row)
                else:
                    writer18to24.writerow(row)

			

            			
