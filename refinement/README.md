# Overview

As a team's Jira history matures metrics are gathered which can be used to aid in the sizing of issues.
This project trains a neural network from exported issue query results and attempts to categorize new issues to determine an sizing estimate.  

## Training data

Training data is exported from Jira and provided as a kubernetes secret. Data must be provided as CSV as `training.csv` to the secret. The fields Description, Issue Type, Summary, and Sizing are required.

## Building the model

The model is rebuilt from training data each time the container starts. The container expects to find training data at `/training/training.csv`.

