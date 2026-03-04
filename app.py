#modelle dıs dunya arasındaki kopru web sunucusuFor
#modeli canlandırmak ,arayuz saglamak, url vericek
#html sayfasına donusturup 

import os
import json
from flask import Flask, render_template #python verilerini htmle gondermemizi saglayacak
from pyspark.sql import SparkSession  #spark ile iletisim kurucak
from pyspark.ml.recommendation import ALSModel

app = Flask(__name__)    #web satırını hayata gecirdik uygulamaya kimlşk verdik

spark = SparkSession.builder \
    .appName("MovieRecommender") \
    .getOrCreate() #spark evrenine giris anahtarı vererek kapıyı actık
model_path = "als_model"
movies_parquet_path = "movies_parquet"

model = ALSModel.load(model_path) #disk uzerinde pasif duran dijital beynl yani alsmodel klasoruyu ram'e yukluyor model emir bekliyor hazır ve canlı 
movies_df = spark.read.parquet(movies_parquet_path)  #modeli sadece film idleri degerler vercek 589 nolu filmi oneriyo diyemez film isimleriyle eslestiricek

@app.route("/")  #adres tabelasu bu tabelanın altındakı fonskyonu ozel bir gorevle calıstır 
def index():
    return "Film tavsiyesi almak için adres çubuğuna /recommend/KULLANICI_ID yazin."

@app.route("/recommend/<int:user_id>")  # << dinamik adres tabelası recommendden sonra gelen herhangi bi tamsayıyı al user id degiskene ata spnra da bunu asagıdakı fonksıyona hediye et
def get_recommendations(user_id):
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recommendations = model.recommendForUserSubset(user_df, 10)
    # Karmaşık yapıdan sadece movieId ve rating alıyoruz, tek kullanıcı var
    recommendations_df = recommendations.select("recommendations.movieId","recommendations.rating").first().asDict() #karmasık yapının icinden sadece movie idlerini ve ratişngleri seciyo tek bir kullanıcı oldugu icin  first asdict'le python sozlugune ceviriyo
    recs_list = []

    if recommendations_df and recommendations_df.get('movieId'):
        recommended_movies_df = spark.createDataFrame(zip(recommendations_df['movieId'] , recommendations_df ['rating']), ["movieId", "rating"])
        final_recommendations = recommended_movies_df.join(movies_df,"movieId") 

        recs_list = final_recommendations.toJSON().map(lambda j:json.loads(j)).collect()
        #sparktan pythona gecisi sagladık.
    return render_template('recommendations.html' , user_id= user_id , recommendations=recs_list)
    
if __name__ == "__main__": 
    app.run(debug=True , port=5001) #debug hata mesajlarını detaylı goster port url uzerınde hangı porttan ulasacagımızı soylicek
