# Disaster Response Pipeline Project
Udacity Data Scientist Project

## Table of Contents
1. About
2. Installation
3. Poject Motivation
4. File Descriptions
5. Results

## About
This is a program for Udacity's nanodegree of Data Scientist.

## Installaion
You need Python3 and these libralies.
* matplotlib
* nltk
* numpy
* pandas
* scipy
* sklearn
* sqlalchemy
* tqdm

## Project Motivation
This project will classify categories of disaster messages and display them on a web app.

The web app look like below.
<img src="img/web_app.png" width="80%" alt="disaster response project web app">

## File Descriptions
* notebook/ETL Pipeline Preparation.ipynb
  * ETL pipeline working at notebook
* notebook/ML Pipeline Preparation.ipynb
  * Machine learning working at notebook
* app/run.py
  * main script of web app
* app/templates/*.html
  * html templates of web app
* data/DisasterResponse.db
  * database of disaster response
* data/disaster_categories.csv
  * raw data of disaster categories
* data/disaster_messages.csv
  * raw data of disaster messages
* data/process_data.py
  * ETL pipeline
* models/classifier.pkl
  * trained model
* models/train_classifier.py
  * train script

## Acknowledgements
The disaster message data was from [Figure Eight](https://www.figure-eight.com/).
