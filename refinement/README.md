# Jira Issue Refinement Assistant

This utility uses a machine learning model to assess Jira issues and determine if they are ready for refinement and prioritization. It can be used to generate training data from Jira or to continuously poll a Jira instance and tag issues that are predicted to be ready.

## How it Works

The script processes Jira issues by combining their `Summary` and `Description` fields into a single text block. This text is then fed into a pre-trained NLP model which classifies the issue as either ready for refinement ('Y') or not ('N').

The assistant operates in two main modes:
1.  **Training Data Creation**: Fetches issue data from Jira based on a JQL query and saves it to a CSV file. This data can then be used to train the model.
2.  **Assessment**: Polls Jira for new or updated issues, uses the NLP model to predict their refinement readiness, and can automatically add a `ready-for-prioritization` label and a comment to the issue.

## Prerequisites

The following Python libraries are required:
- `pandas`
- `jira-python`
- `spacy`
- `scikit-learn`
- `tensorflow`
- `keras`
- `nltk`

Additionally, the `en_core_web_md` model for `spacy` is needed. It can be downloaded via:
```bash
python -m spacy download en_core_web_md
```

## Configuration

The application is configured through environment variables:

| Variable                      | Description                                                  | Required |
| ----------------------------- | ------------------------------------------------------------ | -------- |
| `JIRA_URL`                    | The base URL of your Jira instance (e.g., `https://my-jira.com`). | Yes      |
| `JIRA_PERSONAL_ACCESS_TOKEN`  | A Personal Access Token for authenticating with the Jira API.  | Yes      |
| `POLL_RATE`                   | The interval in minutes for polling Jira in assessment mode.   | No (defaults to 60) |

## Training Data

The model is trained using CSV files. The training data must contain the following columns:
- `Summary`: The issue summary.
- `Description`: The issue description.
- `REFINED`: The label, which must be `Y` for issues considered ready for refinement, and `N` otherwise.

Alternatively, you can provide a CSV with `text` and `REFINED` columns, where `text` is the pre-combined summary and description.

## Usage

The script is controlled via command-line arguments.

### 1. Create Training Data (`create-training-data`)

This command queries Jira and creates a CSV file that can be used for training the model.

**Arguments:**
- `--jql "QUERY"`: (Required) The JQL query to select issues for the training set.
- `--output-file FILENAME`: The name for the output CSV file. Defaults to `train_test_out.csv`.

**Example:**
```bash
export JIRA_URL="https://your-jira.example.com"
export JIRA_PERSONAL_ACCESS_TOKEN="your_pat"

python main.py create-training-data \
  --jql 'project = "MyProject" AND status = "Done"' \
  --output-file "my_project_training_data.csv"
```

### 2. Assess Issue Prioritization (`assess-prioritization`)

This command starts a process that continuously polls Jira, assesses issues, and updates them.

**Arguments:**
- `--train-files FILE [FILE ...]`: (Required) A list of one or more training CSV files to train the model.
- `--live-run`: If included, the script will modify Jira issues (add labels and comments). If omitted, it runs in "dry-run" mode, only logging the actions it would have taken.

**Example (Dry Run):**
```bash
export JIRA_URL="https://your-jira.example.com"
export JIRA_PERSONAL_ACCESS_TOKEN="your_pat"

python main.py assess-prioritization \
  --train-files "training_data.csv"
```

**Example (Live Run):**
```bash
export JIRA_URL="https://your-jira.example.com"
export JIRA_PERSONAL_ACCESS_TOKEN="your_pat"

python main.py assess-prioritization \
  --train-files "training_data.csv" "more_data.csv" \
  --live-run
```

## Containerized Execution

A `Containerfile` is provided to build a container image for the application.

**1. Build the Image:**
```bash
docker build -t jira-refinement-assistant -f Containerfile .
```

**2. Run the Container:**

You must pass the required environment variables and the command-line arguments to the `docker run` command.

**Example (Assessment Dry Run):**
```bash
docker run --rm \
  -e JIRA_URL="https://your-jira.example.com" \
  -e JIRA_PERSONAL_ACCESS_TOKEN="your_pat" \
  -v $(pwd)/my_training_data.csv:/usr/app/src/my_training_data.csv \
  jira-refinement-assistant \
  python main.py assess-prioritization --train-files "my_training_data.csv"
```

Note: In the example above, a local training file is mounted into the container at `/usr/app/src/my_training_data.csv`. 