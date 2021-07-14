import os
import time

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

__SOURCE_PATH = ""
__DEST_PATH = ""
if __name__ == '__main__':
    conf = SparkConf()
    conf.setMaster("local[6]")
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.executor.cores", "1")
    conf.set("spark.hadoop.parquet.enable.summary-metadata", "false")
    conf.set("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    spark = SparkSession.builder.appName("EnrichProcessor").config(
        conf=conf).getOrCreate()

    transaction_df = spark.read.csv(os.path.join(__SOURCE_PATH, "train_transaction.csv"),
                                    inferSchema=True, header=True)

    identity_df = spark.read.csv(os.path.join(__SOURCE_PATH, "train_identity.csv"),
                                 inferSchema=True, header=True)

    spark.conf.set("spark.sql.shuffle.partitions", 10)
    union_df = transaction_df.join(identity_df, on="transactionID", how="left") \
        .orderBy("transactionDT") \
        .limit(1000)

    union_df.write.parquet(__DEST_PATH, mode="overwrite", compression="snappy")
