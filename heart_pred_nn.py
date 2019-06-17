from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def cargar_data():
	conf = SparkConf().setAppName("HeartPredNN").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)
	return sqlContext.read.csv("heart.csv", header=True).rdd

def preprocesar_data(rdd):
	rdd = rdd.map(lambda x: ( int(x[0]), int(x[1]), 
					int(x[2]), int(x[3]), int(x[4]), 
					int(x[5]) , int(x[6]), int(x[7]), 
					int(x[8]), float(x[9]),
					int(x[10]), int(x[11]), int(x[12]) , 
					int(x[13]) ))
	df = rdd.toDF(["age","sex","cp",
					"trestbps","chol",
					"fbs","restecg",
					"thalach","exang","oldpeak","slope",
					"ca","thal","target"])
	return df

def entrenar(df):
	vectorAssembler = VectorAssembler(
		inputCols=["age","sex","cp",
					"trestbps","chol",
					"fbs","restecg",
					"thalach","exang","oldpeak","slope",
					"ca","thal"],
		outputCol="features"
	)
	stringIndexer = StringIndexer(inputCol="target", 
		outputCol="indexedLabel")
	vectorIndexer = VectorIndexer(inputCol="features", 
		outputCol="indexedFeatures")

	# Division en data de entrenamiento y data de test
	(training_df, test_df) = df.randomSplit([0.7, 0.3])

	# Configurar Red Neuronal
	capas = [13, 13, 13, 2]
	entrenador = MultilayerPerceptronClassifier(
		layers=capas, 
		featuresCol="indexedFeatures",
		labelCol="indexedLabel",
		maxIter=10000
	)

	# Entrenar mi RN
	pipeline = Pipeline(
		stages=[vectorAssembler,
				stringIndexer, 
				vectorIndexer, 
				entrenador]
	)
	return pipeline.fit(training_df), test_df

def validar(modelo, test_df):
	predictions_df = modelo.transform(test_df)
	predictions_df.select("indexedLabel", 
		"probability", "prediction").show()
	evaluador = MulticlassClassificationEvaluator(
		labelCol="indexedLabel", predictionCol="prediction",
		metricName="accuracy"
	)
	exactitud = evaluador.evaluate(predictions_df)
	print("Exactitud: {}".format(exactitud))
	print("PARAMS:{}".format(modelo.explainParams()))


if __name__ == "__main__":
	rdd = cargar_data()
	df = preprocesar_data(rdd)
	modelo, test_df = entrenar(df)
	validar(modelo, test_df)







	



