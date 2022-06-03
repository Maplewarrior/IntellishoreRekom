"""STANDARD PACKAGES"""

import numpy as np
import pandas as pd
import matplotlib
import os
import sys
from datetime import datetime as dt


def get_data():
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

    return df