# Spark Movie Recommendation System

Film öneri sistemi, Apache Spark ALS modeli ile kullanıcı-film bazlı öneri yapar ve Flask arayüzü üzerinden gösterir.

## Kurulum
1. `git clone https://github.com/larahelva/spark_movie_recommendation.git`
2. `conda create -n sparktm-env python=3.11`
3. `conda activate sparktm-env`
4. `pip install -r requirements.txt`
5. `python train_model.py`
6. `python app.py`

## Kullanım
- Ana sayfa: `/`
- Kullanıcı önerisi: `/recommend/<user_id>`  
  Örnek: `/recommend/50`

## Lisans
MIT

