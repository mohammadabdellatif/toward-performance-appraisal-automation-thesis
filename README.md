# Toward Performance Appraisal Automation - A Harmonized Machine Learning Model Utilizing Dialog Act Classification
This is the source code used in the Ms thesis "Toward Performance Appraisal Automation - A Harmonized Machine 
Learning Model Utilizing Dialog Act Classification" by Mohammad S. Abdellatif. The study was based on a production
database of tickets of a helpdesk team, but to maintain privacy and confidentiality, the original database is not 
provided in this source or anywhere else, but a masked version of the data is provided in this repository, allowing 
anyone to run this code to execute same experiment or contribute to this study. 

## Thesis Abstract
Data mining and Machine Learning become essential to extract valuable information instantly and
efficiently from vast and various data formats to assist a business’s progress. Humans are one of the
base assets in businesses, demanding an assessment process to highlight an individual’s strengths and
weaknesses and suggest improvement in work performance and well-being. Still, many obstacles are
faced through the process, and biased human input is the most influential. Performance Appraisals
are managed and executed throughout the year, consuming time and effort from many participants
who could lose belief in the value of this process after a while. Assessment tools and techniques
vary, but the issues raised are almost the same, and the fairness of an individual’s score is the focus,
whether it reflects his level or not. This study is an experiment conducted on a help desk team
operating in an international Jordanian software company to introduce an automated performance
appraisal model to appraise them based on reported issues data and valuable features extracted by
classifying the exchanged messages according to Dialog Acts. Datasets of the reported issues and the
related communication comments were extracted. The comments were processed, broken into sentences/
utterances, and then classified according to Dialog Acts. An annotator appraised a stratified
sample of the issues to produce a labeled dataset to train the Decision Tree, Naive-Bayes, k-nearest
Neighbors, Logistic Regression, and Ordinal Logistic Regression classifiers to predict the performance
scores by following multiple training approaches on different dataset formats. The trained models’
accuracy ranged from 14% to 86%, noting that the performance generally improved when Dialog Act
features were included, producing automated performance appraisal models interpreted to find performance
benchmarks that justify the scores and guide improvement.

**Keywords**: Performance Management, Performance Appraisal, Employees Assessment, Machine
Learning, Data Mining, Dialog Act Classification, Decision Tree, KNN, Logistic Regression,
Ordinal Logistic Regression, Naive-Bayes, Clustering, K-Means, Automation.

## Python modules
The code is organized in the following modules:
* [classifiers](classifiers): the common code to train and test the classifiers
* [clustering](clustering): the code to cluster the issues
* [ds_extractor](ds_extractor): the code that extracts the data from the database and do the required data 
engineering practices.
* [exploration](exploration): the code to generate the plots and statistical summary tables for issues 
and comments datasets.
* [preprocessing](preprocessing): the code that performs data engineering functions.
* [run](run): the main scripts to run the experiment.
* [sampling](sampling): the code that is used to generate a representative sample from the issues.
* [sys_utils](sys_utils): common utilities shared a cross the code.

## Main Python Scripts
The scripts under [run](run) module are the entry points to run the scripts that
extracted the dataset from the Postgres database, do the required preprocessing, and build the final dataset that is used to train ML models.

- [issues_extractor.py](run/issues_extractor.py): extract the issues data from the database and builds two datasets, issues.csv and issues_snapshots.csv.
It reads the issues information from the database tables and walks through the history of the issues (history reply) to calculate
the spent time and counts the processing steps, and do the required pre-processing and data cleaning.
The script takes around 2 hours to generate both datasets.
- [comments_extractor.py](run/comments_extractor.py): extract issues comments, do the required pre-processing, splits the comments into utternaces, then builds the utterances.csv dataset
- [sample_issues.py](run/sample_issues.py): extract a representative sample from issues.csv that are sent to the annotator for scoring
- [build_w2v_clustered_ml_dataset.py](run/build_w2v_clustered_ml_dataset.py): takes the utterances.csv dataset, classify the text by DA classes by clustering the utterances using Word2Vec vectors, then combines the summary of the classification with utterances snapshots to produce the final dataset for the ML models training.

## Notebooks 
The [docker-compose.yml](docker-compose.yml) file builds and runs the jupyter notebook docker container:

```shell
docker-compose up 
```

and after running the file, you can open the notebook link [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab)
to browse and open the scripts.

**Important Note:** The notebooks can't be run independently of this project source, as the common scripts
and code are in Python scripts and modules, and this project folder is mounted to the jupyter notebook container

| Notebook name                                                                                    | Usage                                                                                                        |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [comments_exploration.ipynb](comments_exploration.ipynb)                                         | Explore and visualize the pp_utterances.csv dataset                                                          |
 | [issue_features_correlation.ipynb](issue_features_correlation.ipynb)                             | Explore possible correlations between issues features                                                        |
 | [issues_categorizing.ipynb](issues_categorizing.ipynb)                                           | An experiment to cluster issues by projects nature, but this was never used in the study                     |
 | [issues_clustering_experimentor.ipynb](issues_clustering_experimentor.ipynb)                     | A notebook to experiment and visualize issues clustering                                                     |
 | [issues_exploration.ipynb](issues_exploration.ipynb)                                             | For exploring the issues.csv through statistical summary and visual plots                                    |
 | [merged_dataset_exploration.ipynb](merged_dataset_exploration.ipynb)                             | For exploring the dataset that is the result of merging issues_snapshots.csv and utterances.csv              |
 | [score_classification-knn.ipynb](score_classification-knn.ipynb)                                 | Train and test a KNN classifier to predict the scores                                                        |
 | [score_classification-logistic-regression.ipynb](score_classification-logistic-regression.ipynb) | Train and test a Logistic Regressor to predict the scores                                                    |
 | [score_classification-NB.ipynb](score_classification-NB.ipynb)                                   | Train and test a Naive Bayesian classifier to predict the scores                                             |
 | [score_classification-OLR.ipynb](score_classification-OLR.ipynb)                                 | Train and test an Ordered Logistic Regressor (OLR) to predict the scores                                     |
 | [score_classification-OLR-stepwise.ipynb](score_classification-OLR-stepwise.ipynb)               | The notebook uses a stepwise AIC to select the best features to fit OLR                                      |
 | [score_classification-DT.ipynb](score_classification-DT.ipynb)                                   | Train and test a Decision Tree classifier to predict scores                                                  |
 | [scored_issue_features_correlation.ipynb](scored_issue_features_correlation.ipynb)               | Explore the correlation between features in the final dataset used for ML model training                     |
 | [utterances_clustering.ipynb](utterances_clustering.ipynb)                                       | Experiment clustering for utterances to detect the correct number of clusters to choose for each author role |
 | [utterances_exploration.ipynb](utterances_exploration.ipynb)                                     | Explore and visualize the pre-processed comments in pp_utterances.csv                                        |

## Datasets
The following datasets are provided:

| Dataset                      | Purpose & Usage                                                                                                            |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| issues.csv                   | Contains all issues exported from the production database                                                                  |
| issues_snapshots.csv         | Contains all issues exported from the production database, but contains a separate record for each issue per assignee turn |
| pp_utterances.csv            | Contains all pre-processed issues' comments.                                                                               |
| issues_snapshot_sample.xlsx  | Contains the issues samples and the scores set by the annotator                                                            |


## Execution sequence
The scripts in this project could be utilized to be executed inside a data pipeline, such like Apache Airflow or Nifi,
but it could be run locally through Python shell or an IDE as below:

**Steps 1 and 2 are not required to run the experiment, as those datasets are already provided in [datasets](datasets)
folder.**

1. Run [issues_extractor.py](run/issues_extractor.py) to produce two datasets, the issues.csv and issues_snapshots.csv as
a result of extracting issues/tickets information from the database.
2. [comments_extractor](run/comments_extractor.py) will run the required scripts to connect to the database 
and then extract the issues/tickets comments from the database tables to produce datasets utterances.csv, 
and the pp_utterances.csv.
3. Run [sample_issues.py](run/sample_issues.py) to extract a representative sample for the annotator.
4. Run [build_w2v_clustered_ml_dataset.py](run/build_w2v_clustered_ml_dataset.py) to build the final dataset to train the
ML models to predict the scores.
5. Open and then run any notebook **score_classification-###.ipynb** to train and test ML model to predict the score.
6. Notebooks require having the datasets under ./temp folder to run correctly, as the previous steps generate the
required datasets under that folder. Any notebook not mentioned in the [notebooks](#notebooks) section are for running
experiments and practices that could help in the study.

