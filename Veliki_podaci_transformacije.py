# Databricks notebook source
# MAGIC %md
# MAGIC # 1 - Data transformation

# COMMAND ----------



# COMMAND ----------

import pandas as pd
from pyspark.sql import Row, Column
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

# MAGIC %md
# MAGIC ## File 1 - OI Results 1896-2016
# MAGIC ID - id of athlete
# MAGIC <br>
# MAGIC Name - name of athlete
# MAGIC <br>
# MAGIC Gender - gender of athlete
# MAGIC <br>
# MAGIC Age - age of athlete, can be NULL
# MAGIC <br>
# MAGIC Height - height of athlete, can be NULL
# MAGIC <br>
# MAGIC Weight - weight of athlete, can be NULL
# MAGIC <br>
# MAGIC Team - Country team the athlete represents
# MAGIC <br>
# MAGIC NOC - National Olympic Committee
# MAGIC <br>
# MAGIC Games - Summer or Winter season and year held
# MAGIC <br>
# MAGIC Year - when Olympics took place
# MAGIC <br>
# MAGIC Season - Summer or Winter
# MAGIC <br>
# MAGIC City - where Olympics took place
# MAGIC <br>
# MAGIC Sport - Athlete's sport
# MAGIC <br>
# MAGIC Event - Athlete's discipline
# MAGIC <br>
# MAGIC Medal - Medal athlete got on event
# MAGIC <br><br>
# MAGIC Link: https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results

# COMMAND ----------

file_location = "dbfs:/FileStore/tables/athletes-1.csv"

#schema 
schema_def = StructType([StructField('ID', IntegerType(), True),
                     StructField('Name', StringType(), True),
                     StructField('Gender', StringType(), True),
                     StructField('Age', IntegerType(), True),
                     StructField('Height', IntegerType(), True),
                     StructField('Weight', IntegerType(), True),
                     StructField('Team', StringType(), True),
                     StructField('NOC', StringType(), True),
                     StructField('Games', StringType(), True),
                     StructField('Year', IntegerType(), True),
                     StructField('Season', StringType(), True),
                     StructField('City', StringType(), True),
                     StructField('Sport', StringType(), True),
                     StructField('Event', StringType(), True),
                     StructField('Medal', StringType(), True),
])

# COMMAND ----------

#load data from csv file
games = spark.read.csv(file_location, encoding="UTF-8", header=True, schema=schema_def) 
games.cache()

#replace null with -1 for age, height and weight
games = games.fillna(-1, 'Age')
games = games.withColumn('Medal', regexp_replace('Medal','NA', 'NO'))

#select columns
games_df = games.select(
    "ID",
    "Name",
    "Age",
    "NOC",
    "Games",
    "Sport",
    "Event",
    "Medal",
    "Team",
    "Gender"
)

games_df.cache()
# dropping ALL duplicate values
games_no_duplicates_df = games_df.drop_duplicates(subset=['Name','Games'])
# sorting by id and name
games_no_duplicates_df = games_no_duplicates_df.sort(['ID','Name'])

athl_df = games_no_duplicates_df.select("Age", "Medal").where((col("Medal") == "Gold") | (col("Medal") == "NO") & (col("Age") != -1))
# save in DBFS
athl_df.write.mode("overwrite").csv(path='dbfs:/FileStore/tables/athletes_results.csv', header=True)
athl_df.show()
display(games_no_duplicates_df)


games_no_duplicates = games_no_duplicates_df.rdd
games_list = games_no_duplicates.map(list).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ### RDD map and reduce transformations

# COMMAND ----------

# athlete who participated in games
games_list_filter = [el for el in games_list if len(el[4]) > 0]
athletes_rdd = sc.parallelize(games_list)
athletes = athletes_rdd.map(lambda row: row[1]).cache()

athletes_count = athletes.map(lambda athlete: (athlete, 1)).reduceByKey(lambda c1,c2: c1+c2).sortBy(lambda gc: gc[1], ascending=False).cache()
athletes_map = athletes_count.take(5)

#Top 5 athletes who participated in most of the Olimpycs
for athlete in athletes_map:
    print(athlete)


# COMMAND ----------

# teams who had most participants in game
games_list_filter = [el for el in games_list if len(el[4]) > 0]
teams_rdd = sc.parallelize(games_list)
teams = teams_rdd.map(lambda row: row[3]+"_"+row[4]).cache()

teams_count = teams.map(lambda team: (team, 1)).reduceByKey(lambda c1,c2: c1+c2).sortBy(lambda tc: tc[1], ascending=False).cache()
teams_map = teams_count.take(5)

#Top 5 teams who had most participants in Olimpycs in one year
for team in teams_map:
    print(team)

# COMMAND ----------

teams_rdd = sc.parallelize(games_list)
teams = teams_rdd.map(lambda row: row[3]+"_"+row[4]).cache()

teams_count = teams.map(lambda team: (team, 1)).reduceByKey(lambda c1,c2: c1+c2).sortBy(lambda tc: tc[1], ascending=False).cache()
teams_map = teams_count.take(5)

#Top 5 teams who had most participants in Olimpycs in one year
for team in teams_map:
    print(team)

# COMMAND ----------

def delete_team_duplicates(list):
    list_filter = []
    for elem in list:
        new_elem = elem[3]+"_"+elem[4]+"_"+elem[6]
        if (elem[8] == 'Team') and new_elem not in list_filter:
            list.remove(elem)

    return list

# COMMAND ----------

#country with most won medals in games
games_list_filter = games_df.drop_duplicates(subset=['NOC', 'Games', 'Event', 'Medal']).collect()

#filter medals only
games_list_filter = [el for el in games_list_filter if el[7] != 'NO']

medals_rdd = sc.parallelize(games_list_filter)

medals = medals_rdd.map(lambda row: row[3]+"_"+row[4]).cache()

medals_count = medals.map(lambda medal: (medal, 1)).reduceByKey(lambda c1,c2: c1+c2).sortBy(lambda mc: mc[1], ascending=False).cache()
medals_map = medals_count.take(5)

#Top 5 teams who had most medals in Olimpycs in one year
for team in medals_map:
    print(team)
    

# COMMAND ----------

games_no_duplicates_athlete = games_df.drop_duplicates(subset=['Name','Games','Event'])
games_no_duplicates_athlete = games_no_duplicates_athlete.rdd
games_list_athlete = games_no_duplicates_athlete.map(list).collect()
games_list_filter = [el for el in games_list_athlete if el[7] != 'NO']

athletes_medals_rdd = sc.parallelize(games_list_filter)

athletes_medals = athletes_medals_rdd.map(lambda row: row[1]).cache()

athletes_medals_count = athletes_medals.map(lambda athlete: (athlete, 1)).reduceByKey(lambda c1,c2: c1+c2).sortBy(lambda amc: amc[1], ascending=False).cache()
athletes_medals_map = athletes_medals_count.take(5)

for athlete in athletes_medals_map:
    print(athlete)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transformation for OI data 1986-2016
# MAGIC We want to get number of gold, silver and bronze medals for each Game and each NOC who won medals on OI Games each year

# COMMAND ----------

games_df_new = games_df.drop_duplicates(subset=['NOC', 'Games', 'Event', 'Medal'])

# GOLD MEDALS FOR NOC AND GAME
gold_df = games_df_new.select("*").where((col("Medal") == "Gold")).groupBy(["NOC", "Games"]).agg(count("*").alias("Gold"))
# SILVER MEDALS FOR NOC AND GAME
silver_df = games_df_new.select("*").where(col("Medal") == "Silver").groupBy(["NOC", "Games"]).agg(count("*").alias("Silver"))
#rename columns
silver_df = silver_df.withColumnRenamed("NOC","NOC 1").withColumnRenamed("Games","Games 1")

# BRONZE MEDALS FOR NOC AND GAME
bronze_df = games_df_new.select("*").where(col("Medal") == "Bronze").groupBy(["NOC", "Games"]).agg(count("*").alias("Bronze"))
#rename columns
bronze_df = bronze_df.withColumnRenamed("NOC","NOC 2").withColumnRenamed("Games","Games 2")

# JOIN ALL TABLES 
first_join = gold_df.join(silver_df, (gold_df["NOC"] == silver_df["NOC 1"]) & (gold_df["Games"] == silver_df["Games 1"]), "outer")
first_join = first_join.withColumn('NOC',when(col('NOC').isNotNull(),col('NOC')).otherwise(col('NOC 1')))
first_join = first_join.withColumn('Games',when(col('Games').isNotNull(),col('Games')).otherwise(col('Games 1')))

second_join = first_join.join(bronze_df, (first_join["NOC"] == bronze_df["NOC 2"]) & (first_join["Games"] == bronze_df["Games 2"]), "outer")
second_join = second_join.withColumn('NOC',when(col('NOC').isNotNull(),col('NOC')).otherwise(col('NOC 2')))
second_join = second_join.withColumn('Games',when(col('Games').isNotNull(),col('Games')).otherwise(col('Games 2')))

# Remove columns
final_df = second_join.drop("NOC 1", "NOC 2", "Games 1", "Games 2")

final_df = final_df.fillna(0, ['Gold', 'Silver', 'Bronze'])

# Total
final_df = final_df.withColumn("Total",  expr("Gold + Silver + Bronze"))
# Sort
final_df = final_df.sort(desc("Total"))
#final_df = final_df.sort(asc("Games"), asc("NOC"))

display(final_df)

# COMMAND ----------

gm = games.filter(col("Games").contains("Summer"))
gm = gm.select("Event").distinct().orderBy("Event")
display(gm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## File 2 - OI Results 2020
# MAGIC ID - id of athlete
# MAGIC <br>
# MAGIC Team - Olympic team name
# MAGIC <br>
# MAGIC Gold - number of team gold medals 
# MAGIC <br>
# MAGIC Silver - number of team silver medals 
# MAGIC <br>
# MAGIC Bronze - number of team bronze medals 
# MAGIC <br>
# MAGIC Total - sum of number of team medals 
# MAGIC <br>
# MAGIC Rank - overall team placement
# MAGIC <br><br>
# MAGIC Link: https://www.kaggle.com/datasets/berkayalan/2021-olympics-medals-in-tokyo

# COMMAND ----------

# File 2 -> Results from 2020 Olympics
file_location_2 = "dbfs:/FileStore/tables/oi_summer_2020-2.csv"

#schema 
schema_def_2 = StructType([StructField('ID', IntegerType(), True),
                     StructField('Team', StringType(), True),
                     StructField('Gold', IntegerType(), True),
                     StructField('Silver', IntegerType(), True),
                     StructField('Bronze', IntegerType(), True),
                     StructField('Total', IntegerType(), True),
                     StructField('Rank', IntegerType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transformation for OI data 2020
# MAGIC We want to make column NOC by joining with existing dataframe we made with transforming oi data 1986-2016

# COMMAND ----------

oi_2020 = spark.read.csv(file_location_2, encoding="UTF-8", header=True, schema=schema_def_2)
display(oi_2020)

# Add column NOC
oi_2020 = oi_2020.withColumn("Games 2", lit("2020 Summer"))
oi_2020 = oi_2020.join(games_df, (oi_2020["Team"] == games_df["Team"]), "outer").cache()
oi_2020 = oi_2020.select("NOC", "Games 2", "Gold", "Silver", "Bronze", "Total").where(col("Games 2") == "2020 Summer").drop_duplicates(subset=["NOC", "Games 2"])
oi_2020 = oi_2020.fillna('RUS', 'NOC')
display(oi_2020)

# COMMAND ----------

# MAGIC %md
# MAGIC Union dataframe we got with transforming oi data 1986-2016 and results from oi 2020 and adding dense ranking for NOC for each OI Games

# COMMAND ----------

# Add Olympics results from 2020 to all
final_result = final_df.union(oi_2020).cache()
final_result = final_result.sort(desc("Total"))

display(final_result)

# COMMAND ----------

from pyspark.sql.window import Window

windowSpec = Window.partitionBy(final_result["Games"]).orderBy(desc("Total"))
final_result = final_result.withColumn("Rank", dense_rank().over(windowSpec))
display(final_result)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving transformed data in DBFS for later analysis

# COMMAND ----------

final_result.write.mode("overwrite").csv(path='dbfs:/FileStore/tables/oi_results-2.csv', header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## File 3 - RENT
# MAGIC Date - when advertisment was posted
# MAGIC <br>
# MAGIC Price - anual price for house
# MAGIC <br>
# MAGIC Bedrooms - number of bedrooms
# MAGIC <br>
# MAGIC Bathrooms - nummber of bathrooms
# MAGIC <br>
# MAGIC Sqft living - living area in square foot
# MAGIC <br>
# MAGIC Sqft LOT - square footage or area 
# MAGIC <br>
# MAGIC Floors - number of floors
# MAGIC <br>
# MAGIC Waterfront - if house has waterfront or not
# MAGIC <br>
# MAGIC View - number of house views
# MAGIC <br>
# MAGIC Condition - number of conditions
# MAGIC <br>
# MAGIC Sqft above - all living square feet in a home that is above the ground
# MAGIC <br>
# MAGIC Sqft basement - living square feet in a home that is bellothe ground
# MAGIC <br>
# MAGIC Year built - when house is built
# MAGIC <br>
# MAGIC Year renovated - when house is renovated, or 0 if not
# MAGIC <br>
# MAGIC Street - in which street house is located
# MAGIC <br>
# MAGIC City - where house is located
# MAGIC <br><br>
# MAGIC Link: https://www.kaggle.com/datasets/shree1992/housedata

# COMMAND ----------

# File 3 -> House renting
file_location_3 = "dbfs:/FileStore/tables/data.csv"

#schema 
schema_def_3 = StructType([StructField('Date', DateType(), True),
                     StructField('Price', DoubleType(), True),
                     StructField('Bedrooms', DoubleType(), True),
                     StructField('Bathrooms', DoubleType(), True),
                     StructField('Sqft Living', IntegerType(), True),
                     StructField('Sqft LOT', IntegerType(), True),
                     StructField('Floors', DoubleType(), True),
                     StructField('Waterfront', IntegerType(), True),
                     StructField('View', IntegerType(), True),
                     StructField('Condition', IntegerType(), True),
                     StructField('Sqft above', IntegerType(), True),
                     StructField('Sqft basement', DoubleType(), True),
                     StructField('Year built', IntegerType(), True),
                     StructField('Year renovated', IntegerType(), True),
                     StructField('Street', StringType(), True),
                     StructField('City', StringType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC Transformation for RENT data
# MAGIC We want to make column Old depending on year when house was built

# COMMAND ----------

rent_df = spark.read.csv(file_location_3, encoding="UTF-8", header=True, schema=schema_def_3).cache()
rent_df = rent_df.select("Price", "Bedrooms", "Bathrooms", "Sqft Living", "Floors", "Year built").where(col("Price") > 0).cache()
rent_df = rent_df.withColumn("Old", when(((col("Year built") >= 1900) & (col("Year built") <= 1950)), lit("20th c - first half")).when((col("Year built") >= 1951) & (col("Year built") <= 2000), lit("20th c - second half")).when(col("Year built") > 2000, lit("21th")))

steps = rent_df.select("Bedrooms").distinct().collect()
for step in steps[:]:
    print(step[0])
          #step[0])
#    _df = rent_df.select("*").where(col("Bedrooms") == step[0])
#    _df.coalesce(1).write.mode("append").option("header", "true").csv('dbfs:/FileStore/stream')

# COMMAND ----------

# save in DBFS
rent_df.write.mode("overwrite").csv(path='dbfs:/FileStore/tables/rent_results-2.csv', header=True)

# COMMAND ----------

rent_df.groupBy("Old").agg(avg("Price"), count("*")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## File 4 - Diet
# MAGIC <br>
# MAGIC Gender - male or female
# MAGIC <br>
# MAGIC Diet - type of diet
# MAGIC <br>
# MAGIC Weigth - weight before diet
# MAGIC <br>
# MAGIC Weigth After - number of bathrooms
# MAGIC <br><br>
# MAGIC Link: https://www.kaggle.com/datasets/zaranadoshi/anova-diet

# COMMAND ----------

# File 4 -> Diet plan
file_location_4 = "dbfs:/FileStore/tables/Diet.csv"

#schema 
schema_def_4 = StructType([StructField('Gender', StringType(), True),
                     StructField('Diet', StringType(), True),
                     StructField('Weight', DoubleType(), True),
                     StructField('Weight After', DoubleType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC Transformation for Diet data
# MAGIC We want to make column Difference that is made in weight after six weeks diet

# COMMAND ----------

diet_df = spark.read.csv(file_location_4, encoding="UTF-8", header=True, schema=schema_def_4)
diet_df = diet_df.withColumn("Difference", round(diet_df["Weight"]-diet_df["Weight After"], 2)).cache()
display(diet_df)
# save in DBFS
diet_df.write.mode("overwrite").csv(path='dbfs:/FileStore/tables/diet_results.csv', header=True)
