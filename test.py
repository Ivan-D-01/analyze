from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def main():
    spark = SparkSession.builder \
        .appName("SparkMasterTestJob") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    try:
        data = [
            ("Alice", 34, "Engineering"),
            ("Bob", 45, "Sales"),
            ("Charlie", 28, "Marketing")
        ]
        columns = ["name", "age", "department"]

        df = spark.createDataFrame(data, columns)

        print("=== Исходные данные ===")
        df.show()

        filtered_df = df.filter(col("age") > 30)
        print("=== Люди старше 30 ===")
        filtered_df.show()

        dept_stats = df.groupBy("department").count()
        print("=== Количество людей по отделам ===")
        dept_stats.show()

    finally:
        spark.stop()

if __name__ == "__main__":
    main()