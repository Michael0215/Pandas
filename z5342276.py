import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())



def question_1(routes, suburbs):
    """
    :param routes: the path for the routes dataset
    :param suburbs: the path for the routes suburbs
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    routesDF = pd.read_csv(routes)
    df1 = routesDF.copy(True)
    # start = df1['service_direction_name'].str.extract('(^[A-Z0-9]+[a-zA-Z0-9.]*( +[A-Z0-9]+[a-zA-Z0-9]*)*)(?:,)?')[0]
    # end = df1['service_direction_name'].str.extract('(?:,|-|–|then|to|and|or|, and) *([A-Z0-9]+[A-Za-z]*( *[A-Z&0-9]+[A-Za-z0-9]*)*) *(?:via.*|\([A-Za-z ]*\))?$')[0]
    start = df1['service_direction_name'].str.extract('(^[A-Z\d]+[a-z\d]*(\s+[A-Z\d]+[a-z\d]*)*)(?:,|to)?')[0]
    end = df1['service_direction_name'].str.extract('(?:then |to |and |or |,+\s+and |, | - | –+\s)+([A-Z\d]+[a-z\d]*(\s+[A-Z\d]+[a-z\d]*)*)(?:via.*|\([a-zA-Z]*\)+\s)?$')[0]
    df1['start'] = start
    df1['end'] = end
    #################################################
    log("QUESTION 1", output_df=df1[["service_direction_name", "start", "end"]], other=df1.shape)
    return df1

def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: dataframe df2
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df2 = pd.concat([df1['start'], df1['end']]).value_counts()[0:5]
    df2 = df2.to_frame().rename_axis().reset_index()
    df2 = df2.rename(columns={'index': 'sevice_location', 0: 'frequence'})
    #################################################

    log("QUESTION 2", output_df=df2, other=df2.shape)
    return df2


def question_3(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    df3 = df1.copy(True)
    df3.loc[(df3['mot_for_interchange'] == 'Other'), 'transport_name'] = 'Bus'
    df3.loc[(df3['mot_for_interchange'] == 'Bus'), 'transport_name'] = 'Bus'
    df3.loc[(df3['mot_for_interchange'] == 'Coach'), 'transport_name'] = 'Bus'
    df3.loc[(df3['mot_for_interchange'] == 'Commuter railway'), 'transport_name'] = 'Light Rail'
    df3.loc[(df3['mot_for_interchange'] == 'Ferry'), 'transport_name'] = 'Ferry'
    df3.loc[(df3['mot_for_interchange'] == 'Tram'), 'transport_name'] = 'Train'
    df3.loc[(df3['mot_for_interchange'] == 'Subway'), 'transport_name'] = 'Metro'
    #################################################

    log("QUESTION 3", output_df=df3[['transport_name']], other=df3.shape)
    return df3


def question_4(df3):
    """
    :param df3: the dataframe created in question 3
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df4 = df3.copy(True)
    df4 = df4['transport_name'].value_counts()
    df4 = df4.to_frame().rename_axis().reset_index()
    df4 = df4.rename(columns={'index': 'transport_name', 'transport_name': 'frequency'})
    df4 = df4.sort_values(by='frequency', ascending=True)
    df4 = df4.set_index([pd.Index([0, 1, 2, 3, 4])])
    #################################################

    log("QUESTION 4", output_df=df4[["transport_name", "frequency"]], other=df4.shape)
    return df4


def question_5(df3, suburbs):
    """
    :param df3: the dataframe created in question 2
    :param suburbs : the path to dataset
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    df5 = df3[df3['depot_name'].notnull()]
    suburbsDF = pd.read_csv(suburbs)
    df_5 = suburbsDF.copy(True)
    df_5 = df_5[['suburb', 'population']]
    df_5 = df_5.groupby(['suburb']).sum().reset_index()

    df5 = df5['depot_name'].value_counts()
    df5 = df5.to_frame().rename_axis().reset_index()
    df5 = df5.rename(columns={'index': 'suburb', 'depot_name': 'frequency'})
    df5 = pd.merge(df5, df_5, on='suburb').rename(columns={'suburb': 'depot'})
    df5['ratio'] = df5['frequency'] / df5['population']
    df5 = df5[['depot', 'ratio']].sort_values(by='ratio', ascending=False)[0:5]
    df5 = df5.set_index('depot')
    #################################################

    log("QUESTION 5", output_df=df5[["ratio"]], other=df5.shape)
    return df5


def question_6(df3):
    """
    :param df3: the dataframe created in question 3
    :return: pandas pivot table
    """

    #################################################
    # Your code goes here ...
    df6 = df3.copy(True)
    df6 = df6[['operator_name', 'transport_name']]
    df6 = df6.groupby(['operator_name', 'transport_name']).size().unstack(fill_value=0)
    df_6 = df6.copy(True)
    df6.loc['sum of transport'] = df_6.sum(axis=0)
    df6['total transports for each operator'] = df_6.sum(axis=1)
    df6[['total transports for each operator']] = df6[['total transports for each operator']].astype('Int64')
    df6['# of transport types for each operator'] = df_6.astype(bool).sum(axis=1)
    df6[['# of transport types for each operator']] = df6[['# of transport types for each operator']].astype('Int64')
    df6 = df6.T
    df6['# of operators for each transport'] = df_6.astype(bool).sum(axis=0)
    df6['# of operators for each transport'] = df6['# of operators for each transport'].astype('Int64')
    df6 = df6.T
    df6['the most transport for each operator'] = df_6.idxmax(axis=1)
    df6.loc['the largest operator for each transport'] = df_6.idxmax(axis=0)
    table = df6
    #################################################

    log("QUESTION 6", output_df=None, other=table)
    return table


def question_7(df3,suburbs):
    """
    :param df3: the dataframe created in question 3
    :param suburbs : the path to dataset
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    suburbsDF = pd.read_csv(suburbs)
    df7 = suburbsDF.copy(True)
    df7 = df7[df7['state'] == 'NSW']
    df7 = df7[df7['statistic_area'] == 'Greater Sydney']
    df7 = df7[['local_goverment_area', 'population', 'median_income', 'sqkm']]
    df7 = df7.groupby(['local_goverment_area']).mean().reset_index()
    LGA = df7['local_goverment_area'].tolist()
    median_income = df7['median_income'].tolist()
    population = df7['population'].tolist()
    sqkm = df7['sqkm'].tolist()
    np.seterr(divide='ignore', invalid='ignore')
    area_per_population = np.divide(np.array(sqkm), np.array(population)).tolist()
    plt.style.use('seaborn')
    plt.scatter(area_per_population, median_income, s=150, c=np.arange(len(LGA)), cmap='flag')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('area per population')
    plt.ylabel('median income')
    plt.title('z5342276_q7.png')
    plt.tight_layout()
    # plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig("{}-Q7.png".format(studentid))
    plt.show()
    #################################################




def question_8(df3,suburbs):
    """
    :param df3: the dataframe created in question 3
    :param suburbs : the path to dataset
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    suburbsDF = pd.read_csv(suburbs)
    df8 = suburbsDF.copy(True)
    df8 = df8[df8['state'] == 'NSW']
    lng = df8['lng']
    lat = df8['lat']
    plt.hist2d(lng, lat, weights=df8['sqkm'], bins=380)
    plt.style.use('seaborn')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.clim(0, 1)
    df8 = df8[['suburb', 'lng', 'lat']]
    df_8 = df3[df3['start'].isin(df8['suburb']) & df3['end'].isin(df8['suburb'])]
    df_8 = df_8[['transport_name', 'start', 'end']]
    df_8 = df_8.rename(columns={'start': 'suburb'})
    df_8 = pd.merge(df_8, df8, on='suburb').rename(columns={'suburb': 'start'})
    df_8 = df_8.rename(columns={'end': 'suburb'})
    df_8 = pd.merge(df_8, df8, on='suburb').rename(columns={'suburb': 'end'})
    df8 = df_8[df_8['transport_name'] != 'Bus'].reset_index()
    df8 = df8[['transport_name', 'lng_x', 'lng_y', 'lat_x', 'lat_y']]
    z = df8.values.tolist()
    for i in range(len(z)):
        z[i] = [z[i][0], [z[i][1], z[i][2]], [z[i][3], z[i][4]]]
    c = {'Ferry': 'g', 'Light Rail': 'r', 'Train': 'y', 'Metro': 'b'}
    for i in range(len(z)):
        plt.plot(z[i][1], z[i][2], color=c[z[i][0]])
    for legend, c in c.items():
        plt.scatter([], [], label=legend, c=c, s=150)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('z5342276_q8.png')
    plt.savefig("{}-Q8.png".format(studentid))
    plt.show()
    #################################################


if __name__ == "__main__":
    df1 = question_1("routes.csv", "suburbs.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df1.copy(True))
    df4 = question_4(df3.copy(True))
    df5 = question_5(df3.copy(True), "suburbs.csv")
    table = question_6(df3.copy(True))
    question_7(df3.copy(True), "suburbs.csv")
    question_8(df3.copy(True), "suburbs.csv")
