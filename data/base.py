"""STANDARD PACKAGES"""

import numpy as np
import pandas as pd
import matplotlib
import os
import sys
from datetime import datetime as dt


def get_data(only_test = True):
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
    # FIX DATA
    #######################################################
    df_calender["Date"] = df_calender["Date"].str.replace(" ", "-")
    df_calender["Date"] = df_calender["Date"].str.replace("Jan", "01")
    df_calender["Date"] = df_calender["Date"].str.replace("Feb", "02")
    df_calender["Date"] = df_calender["Date"].str.replace("Mar", "03")
    df_calender["Date"] = df_calender["Date"].str.replace("Apr", "04")
    df_calender["Date"] = df_calender["Date"].str.replace("May", "05")
    df_calender["Date"] = df_calender["Date"].str.replace("Jun", "06")
    df_calender["Date"] = df_calender["Date"].str.replace("Jul", "07")
    df_calender["Date"] = df_calender["Date"].str.replace("Aug", "08")
    df_calender["Date"] = df_calender["Date"].str.replace("Sep", "09")
    df_calender["Date"] = df_calender["Date"].str.replace("Oct", "10")
    df_calender["Date"] = df_calender["Date"].str.replace("Nov", "11")
    df_calender["Date"] = df_calender["Date"].str.replace("Dec", "12")

    rekom_date = []
    for j in range(df_calender.shape[0]):
        rekom_date.append(str(df_calender["year"].values[j]) + "-" + df_calender["Date"].values[j])

    df_calender["date_rekom"] = rekom_date

    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
    df_weather["datetime"] = df_weather["datetime"].drop_duplicates()
    df_transactions["transaction_hour"] = pd.to_datetime(df_transactions["transaction_hour"])
    df_transactions["date_rekom"] = pd.to_datetime(df_transactions["date_rekom"])
    df_calender["date_rekom"] = pd.to_datetime(df_calender["date_rekom"])

    df_calender[df_calender["date_rekom"].duplicated() == True]
    df_calender = df_calender[["date_rekom", "Holiday Name"]]
    df_calender_dummies = pd.get_dummies(df_calender)
    df_calender_dummies = df_calender_dummies.groupby("date_rekom").sum()

    # df_calender_dummies["date_rekom"] = df_calender_dummies.index.astype(str)

    #######################################################
    # PLAN DAY
    #######################################################
    df_planday["ShiftApprovedStatus"] = "Approved"
    df_planday.drop_duplicates(inplace=True)

    df_plan = df_planday.groupby(["starthour_rounded", "venueName"]).count()[["employeeID"]]

    df_plan.reset_index(inplace=True)
    df_plan["starthour_rounded"] = pd.to_datetime(df_plan["starthour_rounded"])

    df_planday["ShiftApprovedStatus"] = "Approved"
    df_planday.drop_duplicates(inplace=True)

    df_planday_selected = df_planday[['employeeID', 'starthour_rounded', 'venueName']]
    df_planday_selected[df_planday[['employeeID', 'starthour_rounded', 'venueName']].duplicated() == False]
    df_planday_selected.groupby(['starthour_rounded', 'venueName']).count()

    #######################################################
    # MERGE
    #######################################################

    df = df_transactions.copy()
    df = df.merge(df_venues, how="left", left_on="id_onlinepos", right_on="id_onlinepos")
    df = df.merge(df_weather, how="left", left_on="transaction_hour", right_on="datetime")
    df = df.merge(df_calender_dummies, how="left", left_on="date_rekom", right_on="date_rekom")
    df = df.merge(df_id, how="left", left_on="global_venueName", right_on="venueName")
    df = df.merge(df_plan, how="left", left_on=["global_venueName", "transaction_hour"],
                  right_on=["venueName", "starthour_rounded"])
    df['weekday'] = pd.to_datetime(df["date_rekom"]).dt.dayofweek
    df['month'] = pd.to_datetime(df["date_rekom"]).dt.month
    df['year'] = pd.to_datetime(df["date_rekom"]).dt.year
    df['week'] = pd.to_datetime(df["date_rekom"]).dt.isocalendar().week
    features = ["transactionLocal_VAT_beforeDiscount", "hour", "global_venueName", "zip_code", "clusterCategoryJoined", "m2_salesArea", "m2_nonSalesArea", "temp", "feelslike", "precip", "precipprob", "cloudcover", "solarradiation","conditions", 'Holiday Name_Ascension Day', 'Holiday Name_Bank Holiday',
       'Holiday Name_Christmas Day', 'Holiday Name_Christmas Eve Day',
       'Holiday Name_Constitution Day', 'Holiday Name_Easter Monday',
       "Holiday Name_Father's Day", 'Holiday Name_General Prayer Day',
       'Holiday Name_Good Friday', 'Holiday Name_Labour Day',
       'Holiday Name_Maundy Thursday', "Holiday Name_Mother's Day",
       "Holiday Name_New Year's Day", "Holiday Name_New Year's Eve",
       'Holiday Name_Pentecost Sunday', 'Holiday Name_Second Day of Christmas',
       'Holiday Name_Whit Monday', "capacity", "weekday", "month", "week", "year", "transaction_hour"]

    df2 = df[features]
    df2["weekday"] = df2["weekday"].astype(str)
    df2["month"] = df2["month"].astype(str)
    df2["week"] = df2["week"].astype(str)
    df2["year"] = df2["year"].astype(str)
    df2.set_index("transaction_hour", inplace = True, drop = True)
    
    holidays = [x for x in df2.columns if 'Holiday' in x]
    df2[holidays] = df2[holidays].fillna(0).copy()
    df2.dropna(inplace = True)
    df2 = pd.get_dummies(df2)
    df2 = df2.sort_index()
    df2.index = pd.to_datetime(df2.index)
    part1 = df2[df2.index < dt(2020, 2, 15)]
    part2 = df2[(df2.index > dt(2021, 5, 1)) & (df2.index < dt(2021, 12, 20))]
    part3 = df2[(df2.index > dt(2022, 2, 10))]

    final_data = pd.concat([part1, part2, part3])

    return final_data

def get_test_data(only_test = True):
    #######################################################
    # LOAD
    #######################################################

    df_calender = pd.read_csv("data\master_calendar.csv")
    df_id = pd.read_csv("data\master_id_density_encoded.csv")
    df_planday = pd.read_csv("data\master_planday_shifts_encoded.csv")
    df_transactions = pd.read_csv("data\master_transactions2.csv")
    df_venues = pd.read_csv("data\master_venues_encoded.csv")
    df_weather = pd.read_csv("data\master_weather_data.csv")

    #######################################################
    # FIX DATA
    #######################################################
    df_calender["Date"] = df_calender["Date"].str.replace(" ", "-")
    df_calender["Date"] = df_calender["Date"].str.replace("Jan", "01")
    df_calender["Date"] = df_calender["Date"].str.replace("Feb", "02")
    df_calender["Date"] = df_calender["Date"].str.replace("Mar", "03")
    df_calender["Date"] = df_calender["Date"].str.replace("Apr", "04")
    df_calender["Date"] = df_calender["Date"].str.replace("May", "05")
    df_calender["Date"] = df_calender["Date"].str.replace("Jun", "06")
    df_calender["Date"] = df_calender["Date"].str.replace("Jul", "07")
    df_calender["Date"] = df_calender["Date"].str.replace("Aug", "08")
    df_calender["Date"] = df_calender["Date"].str.replace("Sep", "09")
    df_calender["Date"] = df_calender["Date"].str.replace("Oct", "10")
    df_calender["Date"] = df_calender["Date"].str.replace("Nov", "11")
    df_calender["Date"] = df_calender["Date"].str.replace("Dec", "12")

    rekom_date = []
    for j in range(df_calender.shape[0]):
        rekom_date.append(str(df_calender["year"].values[j]) + "-" + df_calender["Date"].values[j])

    df_calender["date_rekom"] = rekom_date

    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
    df_weather["datetime"] = df_weather["datetime"].drop_duplicates()
    df_transactions["transaction_hour"] = pd.to_datetime(df_transactions["transaction_hour"])
    df_transactions["date_rekom"] = pd.to_datetime(df_transactions["date_rekom"])
    df_calender["date_rekom"] = pd.to_datetime(df_calender["date_rekom"])

    df_calender[df_calender["date_rekom"].duplicated() == True]
    df_calender = df_calender[["date_rekom", "Holiday Name"]]
    df_calender_dummies = pd.get_dummies(df_calender)
    df_calender_dummies = df_calender_dummies.groupby("date_rekom").sum()

    # df_calender_dummies["date_rekom"] = df_calender_dummies.index.astype(str)

    #######################################################
    # PLAN DAY
    #######################################################
    df_planday["ShiftApprovedStatus"] = "Approved"
    df_planday.drop_duplicates(inplace=True)

    df_plan = df_planday.groupby(["starthour_rounded", "venueName"]).count()[["employeeID"]]

    df_plan.reset_index(inplace=True)
    df_plan["starthour_rounded"] = pd.to_datetime(df_plan["starthour_rounded"])

    df_planday["ShiftApprovedStatus"] = "Approved"
    df_planday.drop_duplicates(inplace=True)

    df_planday_selected = df_planday[['employeeID', 'starthour_rounded', 'venueName']]
    df_planday_selected[df_planday[['employeeID', 'starthour_rounded', 'venueName']].duplicated() == False]
    df_planday_selected.groupby(['starthour_rounded', 'venueName']).count()

    #######################################################
    # MERGE
    #######################################################

    df = df_transactions.copy()
    df = df.merge(df_venues, how="left", left_on="id_onlinepos", right_on="id_onlinepos")
    df = df.merge(df_weather, how="left", left_on="transaction_hour", right_on="datetime")
    df = df.merge(df_calender_dummies, how="left", left_on="date_rekom", right_on="date_rekom")
    df = df.merge(df_id, how="left", left_on="global_venueName", right_on="venueName")
    df = df.merge(df_plan, how="left", left_on=["global_venueName", "transaction_hour"],
                  right_on=["venueName", "starthour_rounded"])
    df['weekday'] = pd.to_datetime(df["date_rekom"]).dt.dayofweek
    df['month'] = pd.to_datetime(df["date_rekom"]).dt.month
    df['year'] = pd.to_datetime(df["date_rekom"]).dt.year
    df['week'] = pd.to_datetime(df["date_rekom"]).dt.isocalendar().week
    features = ["transactionLocal_VAT_beforeDiscount", "hour", "global_venueName", "zip_code", "clusterCategoryJoined", "m2_salesArea", "m2_nonSalesArea", "temp", "feelslike", "precip", "precipprob", "cloudcover", "solarradiation","conditions", 'Holiday Name_Ascension Day', 'Holiday Name_Bank Holiday',
       'Holiday Name_Christmas Day', 'Holiday Name_Christmas Eve Day',
       'Holiday Name_Constitution Day', 'Holiday Name_Easter Monday',
       "Holiday Name_Father's Day", 'Holiday Name_General Prayer Day',
       'Holiday Name_Good Friday', 'Holiday Name_Labour Day',
       'Holiday Name_Maundy Thursday', "Holiday Name_Mother's Day",
       "Holiday Name_New Year's Day", "Holiday Name_New Year's Eve",
       'Holiday Name_Pentecost Sunday', 'Holiday Name_Second Day of Christmas',
       'Holiday Name_Whit Monday', "capacity", "weekday", "month", "week", "year", "transaction_hour"]

    df2 = df[features]
    df2["weekday"] = df2["weekday"].astype(str)
    df2["month"] = df2["month"].astype(str)
    df2["week"] = df2["week"].astype(str)
    df2["year"] = df2["year"].astype(str)
    df2.set_index("transaction_hour", inplace = True, drop = True)
    
    holidays = [x for x in df2.columns if 'Holiday' in x]
    df2[holidays] = df2[holidays].fillna(0).copy()
    df2.dropna(inplace = True)
    df2 = pd.get_dummies(df2)
    df2 = df2.sort_index()
    df2.index = pd.to_datetime(df2.index)
    part1 = df2[df2.index < dt(2020, 2, 15)]
    part2 = df2[(df2.index > dt(2021, 5, 1)) & (df2.index < dt(2021, 12, 20))]
    part3 = df2[(df2.index > dt(2022, 5, 20, 10, 0, 0))]

    final_data = part3

    return final_data