from pyspark.sql import SparkSession    
import os 
from pyspark.sql.functions import lit,col

spark = SparkSession.builder.appName("Readapt").config("spark.master", "local").getOrCreate()   # cria o spark

books = spark.read.csv(os.path.join(os.path.dirname(__file__), 'dataset/books_dataset.csv'), header=True,  sep=";", inferSchema=True)   # carrega o dataset


def filterByClassification(classification): # função para filtrar o dataset
    return books.filter(col("classification").contains(lit(classification))).select("title", "description", "genre","pages","cover","reference")    # filtra o dataset
