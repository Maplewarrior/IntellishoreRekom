import numpy as np
import pandas as pd
import matplotlib
import os
import sys
from datetime import datetime as dt

#######################################################
# LOAD
#######################################################

 

df_calender = pd.read_csv("data\master_calendar.csv")
df_id = pd.read_csv("data\master_id_density_encoded.csv")
df_planday = pd.read_csv("data\master_planday_shifts_encoded.csv")
df_transactions = pd.read_csv("data\master_transactions.csv")
df_venues = pd.read_csv("data\master_venues_encoded.csv")
df_weather = pd.read_csv("data\master_weather_data.csv")

 


#######################################################
# FORMAT TO PREPARE MERGE
#######################################################
df_calender["Date"] = df_calender["Date"].str.replace(" ", "-")
df_calender["Date"] = df_calender["Date"].str.replace("Jan", "2022-01")
df_calender["Date"] = df_calender["Date"].str.replace("Feb", "2022-02")
df_calender["Date"] = df_calender["Date"].str.replace("Mar", "2022-03")
df_calender["Date"] = df_calender["Date"].str.replace("Apr", "2022-04")
df_calender["Date"] = df_calender["Date"].str.replace("May", "2022-05")
df_calender["Date"] = df_calender["Date"].str.replace("Jun", "2022-06")
df_calender["Date"] = df_calender["Date"].str.replace("Jul", "2022-07")
df_calender["Date"] = df_calender["Date"].str.replace("Aug", "2022-08")
df_calender["Date"] = df_calender["Date"].str.replace("Sep", "2022-09")
df_calender["Date"] = df_calender["Date"].str.replace("Oct", "2022-10")
df_calender["Date"] = df_calender["Date"].str.replace("Nov", "2022-11")
df_calender["Date"] = df_calender["Date"].str.replace("Dec", "2022-12")

 

 


df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
df_weather["datetime"] = df_weather["datetime"].drop_duplicates()
df_transactions["transaction_hour"] = pd.to_datetime(df_transactions["transaction_hour"] )
df_transactions["date_rekom"] = pd.to_datetime(df_transactions["date_rekom"] )
df_calender["Date"] = pd.to_datetime(df_calender["Date"])

 

#######################################################
# MERGE
#######################################################

 

df = df_transactions.copy()
df = df.merge(df_venues, how = "left", left_on = "id_onlinepos", right_on = "id_onlinepos")
df = df.merge(df_weather, how = "left", left_on = "transaction_hour", right_on = "datetime")

print(df.shape)