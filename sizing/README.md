# Overview

As a team's Jira history matures metrics are gathered which can be used to aid in determining the refinement status of issues. The intention is to help in the preparation of backlog refinement.This project trains a neural network from exported issue query results and attempts to determine if the issue is ready for refinement by the team.  

## Training data

Training data is exported from Jira and provided as a kubernetes secret. Data must be provided as CSV as `training.csv` to the secret. The fields Description, Summary, and a column name REFIND which contains a simple Y/N which tells the model if an issue is refined or not.

## Building the model

The model is rebuilt from training data each time the container starts. The container expects to find training data at `/training/training.csv`.

