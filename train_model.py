import os  
from pyspark.sql import SparkSession 
from pyspark.sql.functions import col 
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType , FloatType

def main():
    spark = SparkSession.builder \
    .appName("ModelEgitimi") \
    .getOrCreate() #fonksiyon tanımladım. 

    data_path = "ml-latest-small/"

    ratings_df = spark.read.csv(data_path + "ratings.csv" , header=True, inferSchema=True ) #headerla basladıgı icin true dondurduk, infersvhema spark her sutunun veri tipini otomatik algılasın diye
    movies_df = spark.read.csv(data_path + "movies.csv" , header=True, inferSchema=True ) 

    ratings_df = ratings_df.withColumn("userId" , col("userId").cast(IntegerType()))
    ratings_df = ratings_df.withColumn("movieId" , col("movieId").cast(IntegerType()))
    ratings_df = ratings_df.withColumn("rating" , col("rating").cast(FloatType())) #ratingler 3.8 vs olaibilir

    #model secimi collabrating filtering kullanıcaz. benzer zevklere sahip insanlarin sevdiğii sen de seversin genelde
    #ALS kullancaz bir filmin ne kadar komedi aksiyon oldugunu kendi kesfetiyo puanlardan

    als = ALS(
        maxIter=10,  #maximum iterasyon sayısı verinin uzerinden ne kadar gecicekse ona ıterasyon denir bunu artırmak modeli daha isabetli yapar ama egitim suresini uzatır
        regParam=0.1,   #overfittine ugramasını engellemek icin Overfıttıng = modelin egitim verisini ezberlemesi yeni veriye basarısız
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop" #modelin egitimi sırasında hic gormedigi bir kullanıcıya da film icin tavsiye uretmeye calistiginda karsilastigi problem hskkınds hic bi sey bilmediğin kullanıcıyla karsılasırsn BIRAK
    )

    model = als.fit(ratings_df)
    model_path = "als_model"
    model.write().overwrite().save(model_path)
    movies_parquet_path = "movies_parquet"
    movies_df.write.mode("overwrite").parquet(movies_parquet_path)


    print(f"Model başarıyla eğitildi ve {model_path} klasorune kaydedildi.")
    print(f"Filmler verisi '{movies_parquet_path}' klasorune kaydedildi.")

    spark.stop()

if __name__ == "__main__" :
    main()

 
