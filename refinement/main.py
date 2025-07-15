import string
import os
import time
import numpy as np
import pandas as pd
from typing import cast
from jira import JIRA
from jira.client import ResultList
from jira.resources import Issue
import logging
import sys
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
import argparse
# packages: pip install tf-keras spacy
# python -m spacy download en_core_web_md 

# Configure logging to log to a file and stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stderr)
                    ])

nlp = spacy.load('en_core_web_md')

def preprocessing(text):
  text = text.lower()
    
  tokens = [token.text for token in nlp(text)]
  
  tokens = [t for t in tokens if 
              t not in STOP_WORDS and 
              t not in string.punctuation and 
              len(t) > 3]
  
  tokens = [t for t in tokens if not t.isdigit()]
    
  return " ".join(tokens)

def prepareTrainingSet(filenames, output_train_file=None, output_test_file=None, output_file=None, test_size=0.2, random_state=42):
    """
    Reads multiple CSV files, processes 'Summary' and 'Description' into a 'text' column,
    concatenates all dataframes, randomly shuffles and splits the common DataFrame
    into training and testing sets, and then writes the 'REFINED' and 'text' columns
    of both sets to separate specified output text files (only if file paths are provided).

    Args:
        filenames (list): A list of paths to the CSV files to be processed.
        output_train_file (str, optional): The path to the output text file for the training set.
                                          If None, training file will not be written.
        output_test_file (str, optional): The path to the output text file for the testing set.
                                         If None, testing file will not be written.
        output_file (str, optional): The path to the output CSV file containing combined data.
                                    If None, combined CSV file will not be written.
        test_size (float): The proportion of the dataset to include in the test split.
                           Defaults to 0.2 (20% for testing).
        random_state (int): Controls the shuffling applied to the data before applying the split.
                            Pass an int for reproducible output across multiple function calls.
                            Defaults to 42.

    Returns:
        tuple: A tuple containing two pandas.DataFrames: (train_df, test_df).
               Returns (pd.DataFrame(), pd.DataFrame()) if no dataframes were successfully loaded.
    """
    all_dfs = [] # List to hold individual DataFrames

    for filename in filenames:
        print(f"Reading CSV training set: {filename}")
        try:
            df = pd.read_csv(filename)
            print(f"Rows in {filename}: {len(df)}")

            # Ensure 'Summary' and 'Description' columns exist
            if 'text' in df.columns and 'REFINED' in df.columns:
                print(f"Info: 'text' and 'REFINED' columns already exist in {filename}. Using file as is.")
                all_dfs.append(df)
                continue

            if 'Summary' not in df.columns or 'Description' not in df.columns:
                print(f"Warning: 'Summary' or 'Description' column missing in {filename}. Skipping file.")
                continue
            if 'REFINED' not in df.columns:
                print(f"Warning: 'REFINED' column missing in {filename}. Skipping file.")
                continue

            # Create 'text' column by concatenating 'Summary' and 'Description'
            # Using .fillna('') to handle potential NaN values gracefully before concatenation
            df["text"] = df["Summary"].astype(str).fillna('') + " " + df["Description"].astype(str).fillna('')

            all_dfs.append(df) # Add the processed DataFrame to our list

        except FileNotFoundError:
            print(f"Error: File not found - {filename}. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"Warning: {filename} is empty. Skipping.")
        except Exception as e:
            print(f"An error occurred while reading {filename}: {e}. Skipping.")

    if not all_dfs:
        print("No dataframes were successfully loaded. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames if no files were loaded

    # Concatenate all DataFrames in the list into one common DataFrame
    # ignore_index=True ensures a continuous new index for the combined DataFrame
    common_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n--- All files concatenated. Total rows in common_df: {len(common_df)} ---")

    # Randomly shuffle the common DataFrame
    # frac=1 samples all rows, random_state ensures reproducibility
    common_df = common_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("DataFrame rows randomized.")

    # Calculate the split point for training and testing sets
    split_index = int(len(common_df) * (1 - test_size))
    train_df = common_df.iloc[:split_index]
    test_df = common_df.iloc[split_index:]

    print(f"DataFrame split into training ({len(train_df)} rows) and testing ({len(test_df)} rows) sets.")

    # Helper function to write DataFrame to file
    def _write_df_to_file(df_to_write, file_path, df_name):
        print(f"Writing 'REFINED' and 'text' columns for {df_name} to {file_path}...")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if 'REFINED' in df_to_write.columns and 'text' in df_to_write.columns:
                    for index, row in df_to_write.iterrows():
                        refined_val = str(row['REFINED']).strip() if not pd.isna(row['REFINED']) else ""
                        text_val = str(row['text']).strip() if not pd.isna(row['text']) else ""
                        f.write(f"__label__{refined_val} {text_val}\n")
                    print(f"Successfully wrote {len(df_to_write)} lines to {file_path}.")
                else:
                    print(f"Error: 'REFINED' or 'text' column not found in {df_name}. Cannot write to file.")
        except Exception as e:
            print(f"An error occurred while writing {df_name} to {file_path}: {e}")

    # Write the training DataFrame to its output file (only if specified)
    if output_train_file is not None:
        _write_df_to_file(train_df, output_train_file, "training set")

    # Write the testing DataFrame to its output file (only if specified)
    if output_test_file is not None:
        _write_df_to_file(test_df, output_test_file, "testing set")

    # Write combined dataframe to a csv file (only if specified)
    if output_file is not None:
        # combine train and test dataframes into a single dataframe
        combined_df = pd.concat([train_df, test_df])

        # drop all columns except for 'REFINED' and 'text'
        combined_df = combined_df[['REFINED', 'text']]

        # Write dataframe to a csv file
        combined_df.to_csv(output_file, index=False)
        logging.info(f"{output_file} written")
    
    return train_df, test_df

def dataframe_to_label_dict(df):
    """
    Converts a DataFrame with 'REFINED' and 'text' columns into a dictionary
    where each key is a unique value from 'REFINED' and the value is a list
    of corresponding 'text' entries.

    Args:
        df (pd.DataFrame): DataFrame with 'REFINED' and 'text' columns.

    Returns:
        dict: Dictionary mapping REFINED values to lists of text.
    """
    label_dict = {}
    for _, row in df.iterrows():
        key = row['REFINED']
        value = row['text']
        if pd.isna(key) or pd.isna(value):
            continue
        if key not in label_dict:
            label_dict[key] = []
        label_dict[key].append(value)
    return label_dict


def create_training_data_from_jira(jql_query, output_filename="jira_training_data.csv"):
    """
    Performs a Jira query, processes the issues, and writes the output to a CSV file.

    The output CSV file will have two columns: 'text' and 'REFINED'.
    - 'text' is a combination of the issue's Summary and Description.
    - 'REFINED' is the predicted assessment of whether the issue is ready for prioritization ('Y' or 'N').

    Args:
        jql_query (str): The JQL query to execute for finding issues.
        output_filename (str): The name of the output CSV file.
    """
    logging.info(f"Starting to create training data from Jira. Output file: {output_filename}")
    
    issues_data = []
    
    try:
        # Use maxResults=False to get all issues. For very large result sets, consider pagination.
        issues = cast(ResultList[Issue], jira.search_issues(jql_str=jql_query, maxResults=False))
        logging.info(f"Found {len(issues)} issues from Jira query.")
        
        for issue in issues:
            summary = issue.fields.summary or ""
            description = issue.fields.description or ""
            
            text = f"{summary} {description}".strip()
            
            if not text:
                logging.warning(f"Skipping issue {issue.key} due to empty summary and description.")
                continue

            ready, _ = isReadyForRefinement(issue)
            refined_status = "Y" if ready else "N"
            
            issues_data.append({"REFINED": refined_status, "text": text})
            
    except Exception as e:
        logging.error(f"An error occurred while fetching or processing Jira issues: {e}")
        return

    if not issues_data:
        logging.warning("No data was collected from Jira. The CSV file will not be created.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(issues_data)
    
    # Save the DataFrame to a CSV file
    try:
        df.to_csv(output_filename, index=False, encoding='utf-8')
        logging.info(f"Successfully wrote {len(df)} rows to {output_filename}")
    except Exception as e:
        logging.error(f"Failed to write data to CSV file {output_filename}: {e}")


def isTypeOfInterest(issue):
    """
    Checks if the issue is of a type that should be considered for refinement.
    """
    # This is a placeholder. The original JQL filters out Epics.
    # You might want to add more logic here based on issue types.
    if issue.fields.issuetype.name == 'Epic':
        return False
    return True

def isReadyForRefinement(issue):
    if not isTypeOfInterest(issue):
        return False, 100.0
        
    labels = issue.raw["fields"]["labels"]
    
    if len(labels) == 0:
        return False, 100.0
        
    for label in labels:
        if label == "ready-for-prioritization" or \
            label == "splat-bot-disable-automation":
            return False, 100.0

    if type(issue.fields.description) == type(None) or \
        type(issue.fields.summary) == type(None):
            return False, 100.0
    
    context = issue.fields.summary + " " + issue.fields.description
    result = nlp(context)
    categories = result._.cats

    if "Y" in categories and categories["Y"] > 0.5:
        return True, categories["Y"]
    else:
        return False, categories["N"]

jira_token = os.environ["JIRA_PERSONAL_ACCESS_TOKEN"]
jira_url = os.environ["JIRA_URL"]

jira = JIRA(server=jira_url,
        token_auth=jira_token)

poll_rate_minutes = 60
if "POLL_RATE" in os.environ:
    poll_rate_minutes = int(os.environ["POLL_RATE"])

def setup_training(train_files=["training-org.csv", "UPDATES_TRAINING_SET_JIRA_REFINEMENT.csv"], output_file=None, output_train_file=None, output_test_file=None):
    train, test = prepareTrainingSet(train_files, output_file=output_file, output_train_file=output_train_file, output_test_file=output_test_file)

    label_dict = dataframe_to_label_dict(train)

    nlp.add_pipe("classy_classification", 
        config={
            "data": label_dict, 
            "model": "spacy"
        }
    )

def assess_prioritization(dry_run=True, train_files=["training-org.csv", "UPDATES_TRAINING_SET_JIRA_REFINEMENT.csv"], output_file=None, output_train_file=None, output_test_file=None):
    setup_training(train_files, output_file=output_file, output_train_file=output_train_file, output_test_file=output_test_file)
    while True:
        logging.info("Querying Jira for issues to assess...")
        issues = cast(ResultList[Issue], jira.search_issues(jql_str='project in (SPLAT, OPCT) AND created > -26w AND status not in (Closed) AND creator not in (cucushift-bot, sgaoshang, "yanhli@redhat.com") AND issuetype not in (Epic) AND (labels in (needs-refinement) OR priority = Undefined or "Story Points" IS EMPTY ) AND status  != Closed ORDER BY key DESC, priority ASC, created ASC, Rank ASC'))
        for issuekey in issues:
            issue = jira.issue(issuekey)
            ready, prob = isReadyForRefinement(issue)
            logging.info("issue %s refinement result: %s, confidence %f", issuekey, "ready" if ready else "not ready", prob)
            if ready:
                if dry_run:
                    logging.info("DRY-RUN: Would mark issue %s as ready-for-prioritization", issuekey)
                else:
                    logging.info("marking issue %s as ready-for-prioritization", issuekey)
                    jira.add_comment(issue,
                                    "splat-jira-bot believes this issue is ready for refinement with a probability of " + str(prob) + ". \
                                     If the issue is not ready for refinement, please set the labels `needs-refinement` and `splat-bot-disable-automation` and this issue will be reviewed to understand the " \
                                    "descrepancy.",
                                    visibility={"type": "group", "value": "jira-users"})
                    new_labels = []
                    labels = issue.get_field("labels")
                    for label in labels:
                        if label == "needs-refinement":
                            continue
                        new_labels.append(label)
                    new_labels.append("ready-for-prioritization")
                    labels.clear()
                    for new_label in new_labels:
                        labels.append(new_label)

                    if len(labels) > 0:                        
                        issue.update(fields={"labels": labels})
                        logging.info("updated labels for issue %s: %s", issuekey, labels)
                        logging.info("backing off for 2 seconds")
                        time.sleep(2)
                    else:
                        logging.info("no labels to update for issue %s", issuekey)
        logging.info("sleeping for %d minutes", poll_rate_minutes)
        time.sleep(60 * poll_rate_minutes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Classifier for Jira Issue Prioritization")
    parser.add_argument("command", choices=["assess-prioritization", "create-training-data"], help="Command to run")
    parser.add_argument("--jql", help="JQL query for creating training data. Required for 'create-training-data'.")
    parser.add_argument("--output-file", help="Output file for training data.")
    parser.add_argument("--output-train-file", help="Output file for training data in fastText format.")
    parser.add_argument("--output-test-file", help="Output file for test data in fastText format.")
    parser.add_argument('--live-run', action='store_true', help='Perform a live run that updates Jira issues. Default is dry-run.')
    parser.add_argument("--train-files", nargs='+', help="A list of training files to use for assessment.", default=["train_test.csv"])
    args = parser.parse_args()

    if args.command == "assess-prioritization":
        if args.train_files:
            assess_prioritization(dry_run=not args.live_run, train_files=args.train_files, 
                                output_file=args.output_file, output_train_file=args.output_train_file, 
                                output_test_file=args.output_test_file)
        else:
            assess_prioritization(dry_run=not args.live_run, output_file=args.output_file, 
                                output_train_file=args.output_train_file, output_test_file=args.output_test_file)
    elif args.command == "create-training-data":
        if not args.jql:
            logging.error("Error: --jql argument is required for the 'create-training-data' command.")
            sys.exit(1)
        create_training_data_from_jira(args.jql, args.output_file)
    
