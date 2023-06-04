## Business_Analytics_Transaction_Clustering_Virtual_Training

### table of contents
   * [Overview](#overview)
      * [Notebook](#r1)
      * [Requirements](#r2)
      * [Trained models](#r3)
      * [Test script](#r4)
      * [Dataset results](#r5)
   * [How to run and test locally](#configure)
   * [Sample Predicted Output](#predict)
   * [Project Flow](#notebook)



## Overview<a name="overview"></a>
   This project was about to categorize customer narrations at the time of transactions and analysis the behaviour of customers based on the categorization. It was done using unsupervised learning algorithms and NLP.

### Notebook<a name="r1"></a>
   The whole work is located at [Training_Task.ipynb](https://github.com/hishamcse/Business_Analytics_Transaction_Clustering/blob/main/Training_Task.ipynb)

### Requirements<a name="r2"></a>
   The requirements to run this project locally and test accordingly at [requirements.txt](https://github.com/hishamcse/Business_Analytics_Transaction_Clustering/blob/main/requirements.txt)

### Trained Models<a name="r3"></a>
   I have trained three models for this project.

   * KMeans Clustering Model
   * MiniBatch KMeans Clustering Model
   * Bisecting KMeans Clustering Model

  These trained models added at [models](https://github.com/hishamcse/Business_Analytics_Transaction_Clustering/tree/main/models) folder. Vectorization model also added there

### Test Script<a name="r4"></a>
   The test script to test the models [test_script.py](https://github.com/hishamcse/Business_Analytics_Transaction_Clustering/blob/main/test_script.py). This will predict the clustering using the trained model. More on how to run this in the later section

### Dataset Results<a name="r5"></a>
   The results on the given labeled small dataset added in the [dataset_results](https://github.com/hishamcse/Business_Analytics_Transaction_Clustering/tree/main/dataset_results) folder. The dataset results done on 25 lakhs data can be found [here](https://drive.google.com/file/d/1YHIdg6M8olq4xPPsnxCGGzPFpJeaWU83/view?usp=sharing)



## How to run and test locally<a name="configure"></a>

 * clone or download the zip of the repo and unzip it
 * inside the project folder, open terminal and run -

       pip install -r requirements.txt

 * now we will be able to run the test script to predict new data. To run the test script, we need to provide a <b>filePath (.csv)</b> indicating the test data, <b>column name(narrations or something like that)</b> for prediction  and <b>the model name</b> of any of the three trained models.
 So, the command will be -

       python3 test_script.py "filepath" "columnName" "modelName"


   * for <b>kmeans</b> model on the "test.csv" dataset for the "narrations" column -

         python3 test_script.py test.csv narrations kmeans

   * for <b>minibatch</b> model on the "test.csv" dataset for the "narrations" column -

         python3 test_script.py test.csv narrations minibatch

    * for <b>bisect</b> model on the "test.csv" dataset for the "narrations" column -

          python3 test_script.py test.csv narrations bisect



## Sample Predicted Output<a name="predict"></a>
   If the command runs successfully, it will create a new csv file like <b>"test_kmeans_result.csv"</b> which will have the output something like this-
   <p align="center">
    <img
      style="border-radius:50%" src="https://github.com/hishamcse/Business_Analytics_Transaction_Clustering/blob/main/images/sample_output.png"  alt="sample"/>
   </p>


## Project Flow<a name="notebook"></a>

   * Load and read Dataset
   * Drop Duplicates Narrations
   * Text Cleaning

     * lowercasing
     * special character and punctuation removal
     * stop words removal
     * tokenization
     * stemming (doesnot give good result, so ignored)
     * lemmatization
     * non-english word and named entity removal (spacy takes a healthy amount of time, that's why manual calculation added)
     * empty narrations removal

   * Text Exploratory Analysis (frequency barchart)
   * Featured Engineering (Gensim Word2Vec vectorization model)
   * Build clustering models

      * KMeans Clustering
      * Minibatch KMeans Clustering
      * Bisecting KMeans Clustering
   * Frequency Plot Per Cluster and Finding Keywords
   * Prediction on new data

