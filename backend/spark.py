from pyspark.sql import SparkSession
import os 
from pyspark.sql.functions import lit,col

spark = SparkSession.builder.appName("Readapt").config("spark.master", "local").getOrCreate()

books = spark.read.csv(os.path.join(os.path.dirname(__file__), 'dataset/books_dataset.csv'), header=True,  sep=";", inferSchema=True)


def filterByClassification(classification):
    return books.filter(col("classification").contains(lit(classification))).select("title", "description", "genre","pages","cover","reference")
