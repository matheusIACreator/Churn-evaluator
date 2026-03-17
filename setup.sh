#!/bin/bash
mkdir -p data
curl -L -o data/telco_churn.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

mkdir -p models
/home/adminuser/venv/bin/python src/train.py