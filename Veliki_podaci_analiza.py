# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 2 - Data Vizualization and Analysis 
# MAGIC ## 2.1 Hypotesis Testing 
# MAGIC For Analysis, I used different statistical test:
# MAGIC - Pearson's correlation coefficient - the test statistics that measures the statistical relationship, or association, between two continuous variables. It is known as the best method of measuring the association between variables of interest because it is based on the method of covariance.  
# MAGIC - T-test - an inferential statistic used to determine if there is a significant difference between the means of two groups and how they are related. 
# MAGIC - The one-way analysis of variance (ANOVA) - used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data

# COMMAND ----------

import pandas as pd
from pyspark.sql import Row, Column
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# define schemas

# OI Games
file_location_1 = "dbfs:/FileStore/tables/oi_results-2.csv"

#schema 
schema_def_1 = StructType([StructField('NOC', StringType(), True),
                     StructField('Games', StringType(), True),
                     StructField('Gold', IntegerType(), True),
                     StructField('Silver', IntegerType(), True),
                     StructField('Bronze', IntegerType(), True),
                     StructField('Total', IntegerType(), True),
                     StructField('Rank', IntegerType(), True)
])

# Athletes
file_location_1_1 = "dbfs:/FileStore/tables/athletes_results.csv"

# schema
schema_def_1_1 = StructType([StructField('Age', IntegerType(), True),
                     StructField('Medal', StringType(), True),
])

# Diet
file_location_2 = "dbfs:/FileStore/tables/diet_results.csv"

#schema 
schema_def_2 = StructType([StructField('Gender', StringType(), True),
                     StructField('Diet', StringType(), True),
                     StructField('Weight', DoubleType(), True),
                     StructField('Weight After', DoubleType(), True),
                     StructField('Difference', DoubleType(), True),
])

# RENT
file_location_3 = "dbfs:/FileStore/tables/rent_results-2.csv"

#schema 
schema_def_3 = StructType([StructField('Price', DoubleType(), True),
                     StructField('Bedrooms', DoubleType(), True),
                     StructField('Bathrooms', DoubleType(), True),
                     StructField('Sqft Living', IntegerType(), True),
                     StructField('Floors', DoubleType(), True),
                     StructField('Year built', IntegerType(), True),
])

# COMMAND ----------

# import data from DBFS
# data for OI Games
oi_df = spark.read.csv(file_location_1, encoding="UTF-8", header=True, schema=schema_def_1)
# data for Athletes medals
athl_df = spark.read.csv(file_location_1_1, encoding="UTF-8", header=True, schema=schema_def_1_1)
# data for Diet
diet_df = spark.read.csv(file_location_2, encoding="UTF-8", header=True, schema=schema_def_2)
# data for RENT
rent_df = spark.read.csv(file_location_3, encoding="UTF-8", header=True, schema=schema_def_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data vizualization

# COMMAND ----------

# 1 Show top 5 OI teams with most gold, silver and bronze medals
display(oi_df.select("NOC","Gold", "Silver", "Bronze").groupBy("NOC").agg(sum("Gold").alias("Gold"), sum("Silver").alias("Silver"), sum("Bronze").alias("Bronze")).orderBy(desc("Gold"), desc("Silver"), desc("Bronze")).head(5))

# 2 Show male/female ratio when it comes to type of diet
male_diet = diet_df.select("Gender", "Diet").where(col("Gender") == "M").groupBy("Diet").agg(count("Gender").alias("Num")).withColumn("Gender", lit("M"))
female_diet = diet_df.select("Gender", "Diet").where(col("Gender") == "F").groupBy("Diet").agg(count("Gender").alias("Num")).withColumn("Gender", lit("F"))
display(male_diet.union(female_diet))

# 3 Show how many badrooms and bathrooms can you get for top 5 highest price
display(rent_df.select("Bedrooms", "Bathrooms", "Year built", "Price").orderBy(col("Price")).head(5))

# 4 Show the average difference made by men and women with different diets
male_diet_diff = diet_df.select("Gender", "Diet", "Difference").where(col("Gender") == "M").groupBy("Diet").agg(round(avg("Difference"), 2).alias("Difference")).withColumn("Gender", lit("M"))
female_diet_diff = diet_df.select("Gender", "Diet", "Difference").where(col("Gender") == "F").groupBy("Diet").agg(round(avg("Difference"), 2).alias("Difference")).withColumn("Gender", lit("F"))
display(male_diet_diff.union(female_diet_diff))

# 5 Serbia OI medals and place over years
display(oi_df.select("Games", "Gold", "Silver", "Bronze", "Rank").where(col("NOC")=="SRB"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data analysis

# COMMAND ----------

from pyspark.mllib.stat import Statistics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pearson's correlation test
# MAGIC The Product Moment Correlation Coefficient (PMCC), or r, is a measure of how strongly related 2 variables are
# MAGIC <br>
# MAGIC Sample size 4600, significant level 0.05
# MAGIC <br>
# MAGIC The hypothesis is one-tailed (right) since we are only testing for positive correlation
# MAGIC <br>
# MAGIC The corresponding critical correlation value r_c for a significance level of α=0.05, for a right-tailed test is: r_c = 0.024
# MAGIC <br>
# MAGIC Observe that in this case, the null hypothesis is rejected if |r| > r_c = 0.024 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 
# MAGIC Null Hypotesis: Number of bathrooms and floors do not correlate with year when house is built 
# MAGIC <br>
# MAGIC Alternate Hypotesis: Number of bathrooms and floors correlate with the year when house is built 
# MAGIC <br>
# MAGIC Output: confusion matrix with calculated PMCCs (r)
# MAGIC <br>
# MAGIC The absolute value of the PMCCs are 0.489, 0.464, 0.467, which is larger than 0.024. Since the PMCCs are larger than the critical value at the 5% level of significance, we can reach a conclusion.
# MAGIC <br>
# MAGIC Conclusion: Since the PMCCs are larger than the critical value, we choose to reject the null hypothesis. We can conclude that there is significant evidence to support the claim that number of bathrooms and floors
# MAGIC are in correlation with the year when house is built

# COMMAND ----------

corr_analysis_df = rent_df.select("Bathrooms", "Floors", "Year built")
vector_col = "col-features"
assembler = VectorAssembler(inputCols=corr_analysis_df.columns, outputCol=vector_col)
df_vector = assembler.transform(corr_analysis_df).select(vector_col)
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corr_matrix = matrix.toArray().tolist()

colums = ['Bathrooms', 'Floors', 'Year built']
df_corr = spark.createDataFrame(corr_matrix, colums)
df_corr.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2
# MAGIC Null Hypotesis: It is not important for the price when house is built
# MAGIC <br>
# MAGIC Alternate Hypotesis: It is important for the price when house is built
# MAGIC <br>
# MAGIC Output: confusion matrix with calculated PMCC (r)
# MAGIC <br>
# MAGIC The absolute value of the PMCC is 0.022 which is NOT larger than 0.024. Since the PMCC is NOT larger than the critical value at the 5% level of significance, we can reach a conclusion.
# MAGIC <br>
# MAGIC Conclusion: Since the PMCC is NOT larger than the critical value, we choose to accept the null hypothesis. We can conclude that there is no significant evidence to support the claim that it is important when house is built for pricing 

# COMMAND ----------

corr_analysis_df = rent_df.select("Year built", "Price")
vector_col = "col-features"
assembler = VectorAssembler(inputCols=corr_analysis_df.columns, outputCol=vector_col)
df_vector = assembler.transform(corr_analysis_df).select(vector_col)
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corr_matrix = matrix.toArray().tolist()

colums = ['Year built', 'Price']
df_corr = spark.createDataFrame(corr_matrix, colums)
df_corr.show()

# COMMAND ----------

from scipy import stats

# COMMAND ----------

# MAGIC %md
# MAGIC ### T-test
# MAGIC #### 3
# MAGIC Test the hypotesis - Is the average age of Gold medal winners and No medal winners same
# MAGIC <br>
# MAGIC First use groupby to check the mean difference
# MAGIC <br>
# MAGIC Null hypotesis: there is no significant difference in the average age of gold medal winners and no medal winners
# MAGIC <br>
# MAGIC Alternate hypotesis: there is significant difference in the average age of gold medal winners and no medal winners
# MAGIC <br>
# MAGIC Output: pvalue=0.12
# MAGIC <br>
# MAGIC Conclusion: Since pvalue is grater than 0.05 fail to reject the null hypotesis and reject Alternate

# COMMAND ----------

mean_df = athl_df.select("Age", "Medal").groupBy("Medal").agg(avg("Age").alias("Average Age"))

medal = athl_df[athl_df.Medal=='Gold'] 
medal = medal.toPandas()
no_medal = athl_df[athl_df.Medal=='NO']
no_medal = no_medal.toPandas()
st = stats.ttest_ind(no_medal.Age,medal.Age,equal_var=False)
print(st)

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-way ANOVA 
# MAGIC #### 4 
# MAGIC Null Hypotesis: There is no significant difference between the means of the three diet groups when come to results after 6 weeks
# MAGIC <br>
# MAGIC Alternate Hypotesis: There is significant difference between the means of the three diet groups when come to results after 6 weeks
# MAGIC <br>
# MAGIC Output: pvalue = 0.003
# MAGIC <br>
# MAGIC Conclusion: pvalue is less than α = 0.05, we reject the null hypothesis of the ANOVA and conclude that there is a statistically significant difference between the means of three diet groups.

# COMMAND ----------

diet_p = diet_df.select("Diet", "Difference").toPandas()
diet_groups = diet_p.groupby('Diet')
a_diet = diet_groups.get_group('A')["Difference"]
b_diet = diet_groups.get_group('B')["Difference"]
c_diet = diet_groups.get_group('C')["Difference"]

anova = stats.f_oneway(a_diet, b_diet, c_diet)
print(anova)

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-way ANOVA
# MAGIC #### 5
# MAGIC Null Hypotesis: There is no significant difference between the means of the best three OI teams based on total won medals
# MAGIC <br>
# MAGIC Alternate Hypotesis: There is significant difference between the means of the best three OI teams based on won medals
# MAGIC <br>
# MAGIC Output: pvalue = 0.216
# MAGIC <br>
# MAGIC Conclusion: pvalue is greater than α = 0.05, we do not reject the null hypothesis of the ANOVA and conclude that there is NO statistically significant difference between the means of the best three OI teams.

# COMMAND ----------

oi_p = oi_df.select("NOC", "Total", "Games").where((col("NOC") == "USA") | (col("NOC") == "CHN") | (col("NOC") == "RUS")).toPandas()
oi_groups = oi_p.groupby('NOC')
usa = oi_groups.get_group('USA')["Total"]
chn = oi_groups.get_group('CHN')["Total"]
rus = oi_groups.get_group('RUS')["Total"]

anova_2 = stats.f_oneway(usa, chn, rus)
print(anova_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-way ANOVA
# MAGIC #### 6
# MAGIC Null Hypotesis: There is no significant difference between the means of number of gold, bronze and silver medals of USA OI team
# MAGIC <br>
# MAGIC Alternate Hypotesis: There is significant difference between the means of gold, bronze and silver medals of USA OI team
# MAGIC <br>
# MAGIC Output: pvalue = 0.135
# MAGIC <br>
# MAGIC Conclusion: pvalue is greater than α = 0.05, we do not reject the null hypothesis of the ANOVA and conclude that there is NO statistically significant difference between the means of number of gold, bronze and silver medals of USA OI team

# COMMAND ----------

oi_usa = oi_df.select("Bronze", "Gold", "Silver").where(col("NOC") == "USA").toPandas()

anova_3 = stats.f_oneway(oi_usa.Bronze, oi_usa.Gold, oi_usa.Silver)
print(anova_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Structured streaming
# MAGIC Since we just have a static set of files, we are going to emulate a stream from them by reading one file at a time, in the chronological order they were created.
# MAGIC We are using rent data. We want to query count od houses and average rent price for different time when house was built. 

# COMMAND ----------

# MAGIC %fs ls "dbfs:/FileStore/stream/"

# COMMAND ----------

input_path = "dbfs:/FileStore/stream/"

# COMMAND ----------

df = spark.read.csv("dbfs:/FileStore/stream/part-00000-tid-1489551210369461886-d1a27393-c3ef-4215-be93-489aea772a25-14134-1-c000.csv", header=True)
display(df)
schema = df.schema

# COMMAND ----------

streamingInputDF = (
  spark
    .readStream                       
    .schema(schema)               
    .option("maxFilesPerTrigger", 1)  
    .csv(input_path)
)

# COMMAND ----------

# MAGIC %md
# MAGIC As you we see, streamingCountsDF is a streaming Dataframe (streamingCountsDF.isStreaming was true). We can start streaming computation, by defining the sink and starting it. In our case, we want to interactively query the counts and average price

# COMMAND ----------

streamingCountsDF = streamingInputDF.select("Price", "Old").groupBy(streamingInputDF.Old).agg(sum(col("Price")).alias("Price"), count("*").alias("count")).orderBy(desc("count"))

streamingCountsDF.isStreaming

# COMMAND ----------

query = (
  streamingCountsDF
    .writeStream
    .format("memory")        # memory = store in-memory table 
    .queryName("counts")     # counts = name of the in-memory table
    .outputMode("complete")  # complete = all the counts should be in the table
    .start()
)

# COMMAND ----------

from time import sleep
sleep(5)

# COMMAND ----------

# MAGIC %sql SELECT Old, count, round((Price/count), 2) as AvgPrice FROM counts

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we can stop the query running in the background, either by clicking on the 'Cancel' link in the cell of the query, or by executing query.stop()

# COMMAND ----------

query.stop()
