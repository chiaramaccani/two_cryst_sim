"""
Simple functions to access NXCALS data.

F.F. Van der Veken October 2021.
Based on Arek Gorzawski.
"""

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.sql.session

from nxcals.api.extraction.data.builders import DataQuery


_NXCALS_timestamp_unit = 'ns'


# Quick function to test that the spark connection is a valid SparkSession
def _test_connection(spark=None):
    if not type(spark) is pyspark.sql.session.SparkSession:
        raise ValueError('Cannot proceed without spark connection, connect to the cluster and try again!')

        
#  Returns basic data set build for the spark extraction
def _build_spark_dataSet(tStart, tEnd, variable, spark=None):
    _test_connection(spark)
    t1 = tStart.astimezone('UTC').to_datetime64()
    t2 = tEnd.astimezone('UTC').to_datetime64()
    return DataQuery.builder(spark).byVariables().system("CMW").startTime(t1).endTime(t2).variableLike(variable).buildDataset()

#  Returns basic data set build for the spark extraction
def _build_spark_dataSets(tStart, tEnd, variables, spark=None):
    _test_connection(spark)
    t1 = tStart.astimezone('UTC').to_datetime64()
    t2 = tEnd.astimezone('UTC').to_datetime64()
    return DataQuery.builder(spark).byVariables().system("CMW").startTime(t1).endTime(t2).variablesLike(variables).buildDataset()

def _build_spark_dataSetsNew(tStart, tEnd, variables, spark=None):
    _test_connection(spark)
    t1 = tStart.astimezone('UTC').to_datetime64()
    t2 = tEnd.astimezone('UTC').to_datetime64()    
    return DataQuery.builder(spark).variables().system("CMW").nameIn(variables).timeWindow(t1b,t1e).build()

# Returns basic data set build for the spark extraction but takes in epoch timestamp in ns, UTC timezone
def _build_spark_dataSetEpoch(tStart, tEnd, variable, spark=None):
    _test_connection(spark)
    t1 = np.datetime64(tStart,'ns')
    t2 = np.datetime64(tEnd,'ns')
    return DataQuery.builder(spark).byVariables().system("CMW").startTime(t1).endTime(t2).variableLike(variable).buildDataset()



def get_unique_data(tStart, tEnd, variable, spark=None):
    """
    Returns 'variable' from NXCALS, assuming it does not change
    over the specified time interval.
    """
    # TODO: do uniqueness check on Spark
    data = _build_spark_dataSet(tStart, tEnd, variable, spark=spark)\
                .select('nxcals_value.elements').toPandas()['elements']
    data = pd.Series([set(val) for val in zip(*data)])
    # sanity check
    if not all([ len(val)==1 for val in data ]):
        raise ValueError('NXCALS data for ' + variable
                         + ' is not constant between ' + str(tStart) + ' and ' + str(tEnd) + '!')
    return pd.Series([ next(iter(val)) for val in data ])


def get_aggregate_data(tStart, tEnd, variable, spark=None):
    """
    Queries 'variable' from NXCALS, returning the average and
    standard deviation over the specified time interval.
    """
    # To check: calculation of std. This was done with np.std before,
    # which does the same as F.stddev_pop. However, in Spark the default
    # is F.stddev which gives a slightly larger result. Maybe better to use that one?
    df = _build_spark_dataSet(tStart, tEnd, variable, spark=spark)\
                .select('nxcals_timestamp','nxcals_value.elements')\
                .select(F.posexplode('elements'))\
                .groupBy('pos').agg(\
                    F.avg('col').alias('avg'),
                    F.stddev_pop('col').alias('std')
                ).sort('pos').select('avg','std')
    return df.toPandas()


def get_timestamped_data(tStart, tEnd, variable, spark=None):
    """
    Queries 'variable' from NXCALS, returning all values found with the respective timestamps.
    """
    # TODO: convert to pd.Timestamp (with correct timezone) on Spark (but without UDFs).
    df = _build_spark_dataSet(tStart, tEnd, variable, spark=spark)\
                .select(
                    F.col('nxcals_timestamp').alias('timestamp'),
                    F.col('nxcals_value.elements').alias('vals')
                )
    df = df.toPandas()
    # If it is needed to round the timestamps on the second, this can be done with pd.Timestamp.round(freq='S')
    df['timestamp'] = [ pd.Timestamp(time, unit=_NXCALS_timestamp_unit, tz='CET') for time in df['timestamp'] ]
    return df


