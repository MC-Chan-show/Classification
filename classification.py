# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report    
from sklearn.metrics import confusion_matrix

from pyspark.ml.stat import Correlation

from pyspark.ml.linalg import Vectors,DenseMatrix
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator

from pyspark.sql.column import Column, _to_java_column, _to_seq
import numpy as np



def rand(seed=None):
        sc = SparkContext._active_spark_context
        if seed is not None:
            jc = sc._jvm.functions.rand(seed)
        else:
            jc = sc._jvm.functions.rand()
        return Column(jc)



def bestmodels(crossval , dataset, numFolds ):
    est=crossval.getEstimator()
    epm=crossval.getEstimatorParamMaps()
    numModels = len(epm)
    nFolds =numFolds
    eva=crossval.getEvaluator()
    seed=crossval.getSeed()
    h = 1.0 / nFolds 
    randCol = crossval.uid + "_rand"
    df = dataset.select("*", rand(seed).alias(randCol))
    metrics = [0.0] * numModels
    for i in range(nFolds):
            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition)
            train = df.filter(~condition)
            models = est.fit(train, epm)
            for j in range(numModels):
                model = models[j]
                # TODO: duplicate evaluator to take extra params from input
                metric = eva.evaluate(model.transform(validation, epm[j]))
                metrics[j] += metric/nFolds
    if eva.isLargerBetter():
        bestIndex = np.argmax(metrics)
    else:
        bestIndex = np.argmin(metrics)    
    
    vals=[]
    for n in epm[bestIndex].values():
        vals.append(n)
    val_name=[]
    for x in epm[bestIndex].keys():
        val_name.append(x.name)
    res=''
    for j in range(len(vals)):
        res+=val_name[j]+': '+str(vals[j])+' , '
    
    print(res)
    

def getPredictionsLabels(model, test_data):
    predictions = model.predict(test_data.map(lambda r: r.features))
    return predictions.zip(test_data.map(lambda r: r.label))


#def printMetrics(predictions_and_labels):
#    metrics = MulticlassMetrics(predictions_and_labels)
#    print('Precision of True ', metrics.precision(1))
#    print('Precision of False', metrics.precision(0))
#    print('Recall of True    ', metrics.recall(1))
#    print('Recall of False   ', metrics.recall(0))
#    print('F-1 Score         ', metrics.fMeasure())
#    print('Confusion Matrix\n', metrics.confusionMatrix().toArray())
    
def vectorizeData(data):
    return data.rdd.map(lambda r: [r[-1], Vectors.dense(r[:-1])]).toDF(['label','features'])
    
def labelData(data):
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))


if __name__ == "__main__":

    sc = SparkSession.builder.master("local").appName("Demo").getOrCreate()

    CV_data = sc.read.load('F:\\zengc_code\\TrainData\\churn\\churn-bigml-80.csv', 
                              format='com.databricks.spark.csv', 
                              header='true', 
                              inferSchema='true')
    
    final_test_data = sc.read.load('F:\\zengc_code\\TrainData\\churn\\churn-bigml-20.csv', 
                              format='com.databricks.spark.csv', 
                              header='true', 
                              inferSchema='true')
    
    CV_data = CV_data.drop('State').drop('Area code') \
        .drop('Total day charge').drop('Total eve charge') \
        .drop('Total night charge').drop('Total intl charge').drop('Voice mail plan').cache()
    
    final_test_data = final_test_data.drop('State').drop('Area code') \
        .drop('Total day charge').drop('Total eve charge') \
        .drop('Total night charge').drop('Total intl charge').drop('Voice mail plan').cache()
    
    stratified_CV_data = CV_data.sampleBy('Churn', fractions={0: 388./2278, 1: 1.0}).cache()
 
    vectorized_CV_data = vectorizeData(stratified_CV_data)
	
    labelIndexer = StringIndexer(inputCol='label',
                                 outputCol='indexedLabel').fit(vectorized_CV_data)

    featureIndexer = VectorIndexer(inputCol='features',
                                   outputCol='indexedFeatures',
                                   maxCategories=2).fit(vectorized_CV_data)
    
    f1_evaluator = MulticlassClassificationEvaluator(labelCol='indexedLabel',predictionCol='prediction', metricName='f1')  
    ROC_evaluator = BinaryClassificationEvaluator(labelCol='indexedLabel',rawPredictionCol='prediction', metricName='areaUnderROC')  
  
    ######################决策树###############
    dTree = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures') 
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dTree])   
    paramGrid = ParamGridBuilder() \
         .addGrid(dTree.maxBins, [10, 17]) \
         .build()  

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=ROC_evaluator,
                              numFolds=2)
    bestmodels(crossval,vectorized_CV_data,2)
    CV_model = crossval.fit(vectorized_CV_data)   
    tree_model = CV_model.bestModel.stages[2]
#    tree_model.explainParams()
#    tree_model.explainParam(tree_model.maxBins)
#    tree_model.explainParam(tree_model.maxDepth)
    
    ###################GBDT##############################
    
    GBTree = GBTClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures')    
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, GBTree])     
    GBparamGrid = ParamGridBuilder() \
    .addGrid(GBTree.maxIter, [20, 25]) \
    .build()

    GBT_crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=GBparamGrid,
                              evaluator=ROC_evaluator,
                              numFolds=2)    
    GBT_model = GBT_crossval.fit(vectorized_CV_data) 

    GBT_tree_model = GBT_model.bestModel.stages[2]

    ###################随机森林##############################
    
    RFTree = RandomForestClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures')
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, RFTree])
    RFparamGrid = ParamGridBuilder() \
     .addGrid(RFTree.maxBins, [10, 17, 29]) \
     .addGrid(RFTree.maxDepth, [7,12]) \
     .build()

    RFT_crossval = CrossValidator(estimator=pipeline,
                               estimatorParamMaps=RFparamGrid,
                               evaluator=f1_evaluator,
                               numFolds=2)
    
    bestmodels(RFT_crossval,vectorized_CV_data,2 )

    RFT_model = RFT_crossval.fit(vectorized_CV_data)
    RFT_tree_model = RFT_model.bestModel.stages[2]

    
    #####预测和模型评估#######
    vectorized_test_data = vectorizeData(final_test_data)
    tf_data = CV_model.transform(vectorized_test_data)
    GBT_tf_data = GBT_model.transform(vectorized_test_data)
    RFT_tf_data = RFT_model.transform(vectorized_test_data)
    
    
    ################决策树评估###################################################   
    print( f1_evaluator.getMetricName(), 'score:', f1_evaluator.evaluate(tf_data,{f1_evaluator.metricName: "f1"}))
    print( ROC_evaluator.getMetricName(), 'AUC:', ROC_evaluator.evaluate(tf_data)) 

    
    predictions = tf_data.select('indexedLabel', 'prediction', 'probability')
    resultdf=predictions.toPandas()
    print(accuracy_score(resultdf.indexedLabel, resultdf.prediction))
    print(confusion_matrix(resultdf.indexedLabel, resultdf.prediction))
    print(classification_report(resultdf.indexedLabel, resultdf.prediction))

    ################GBDT评估###################################################
    print( f1_evaluator.getMetricName(), 'score:', f1_evaluator.evaluate(GBT_tf_data))
    print( ROC_evaluator.getMetricName(), 'AUC:', ROC_evaluator.evaluate(GBT_tf_data))  

    
    predictions = GBT_tf_data.select('indexedLabel', 'prediction', 'probability' )
    resultdf=predictions.toPandas()
    print(resultdf.head())
    print(accuracy_score(resultdf.indexedLabel, resultdf.prediction))
    print(confusion_matrix(resultdf.indexedLabel, resultdf.prediction))
    print(classification_report(resultdf.indexedLabel, resultdf.prediction))

    ################随机森林评估###################################################
    print( f1_evaluator.getMetricName(), 'score:', f1_evaluator.evaluate(RFT_tf_data))
    print( ROC_evaluator.getMetricName(), 'AUC:', ROC_evaluator.evaluate(RFT_tf_data)) 

      
    predictions = RFT_tf_data.select('indexedLabel', 'prediction', 'probability' )
    resultdf=predictions.toPandas()
    print(accuracy_score(resultdf.indexedLabel, resultdf.prediction))
    print(confusion_matrix(resultdf.indexedLabel, resultdf.prediction))
    print(classification_report(resultdf.indexedLabel, resultdf.prediction)) 
    
    
    #############################模型导出#################################
    GBT_model.bestModel.save('GBT_model')
    sc.stop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
