#!/bin/bash

curl --request POST \
  --url http://localhost:5000/predict \
  --header 'Content-Type: application/json' \
  --data '{"PassengerId": 892, "Pclass": 3, "Name": "Kelly, Mr. James", "Sex": "male", "Age": 35, "SibSp": 1, "Parch": 0, "Ticket": "330911", "Fare": 7.8292, "Cabin": "", "Embarked": "Q"}'
