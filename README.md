# DialDoc21: Shared Task on Doc2Dial Dataset

[DialDoc21 Shared Task](https://doc2dial.github.io/workshop2021/shared.html) at [ACL 2021](https://2021.aclweb.org) includes two subtasks for building goal-oriented document-grounded dialogue systems. The first subtask is to predict the grounding in the given document for next agent response; the second subtask is to generate agent response in natural language given document-based and dialogue-based contexts.

## Data

This shared task is based on Doc2Dial v1.0.1 in folder [`data/doc2dial`](data/doc2dial). For more information about the dataset, please refer to [README](data/doc2dial/DATA_README.md), [paper](https://arxiv.org/abs/2011.06623) and [Doc2Dial Project Page](https://doc2dial.github.io/).

**Note**: you can choose to utilize other public datasets in addition to Doc2Dial data for training. Please examples [here](https://mrqa.github.io/2019/shared).

## Shared Task

### Subtask 1

The task is to predict the knowledge grounding in form of document span for the next agent response given dialogue history and the associated document.

- **Input**: the associated document and dialogue history.

- **Output**: the grounding text.

- **Evaluation**: `exact match` and `F1 scores`. Please refer to [script](scripts/sharedtask_utils.py) for more details.

### Subtask 2

The task is to generate the next agent response in natural language given dialogue-based and document-based contexts.

- **Input**: the associated document and dialogue history.

- **Output**: dialog utterance.

- **Evaluation**: `sacrebleu` and `human evaluation`. Please refer to [script](scripts/sharedtask_utils.py) for more details. Stay tuned for more details about human evaluations.


## Baselines

### **Environment Setup**

Create a virtual environment

```bash
conda create -n ENV_NAME python=3.7
conda activate ENV_NAME
````

Install PyTorch

```bash
conda install pytorch cudatoolkit=10.2 -c pytorch
```

Install Huggingface Transformers, Datasets and a few more dependencies

```bash
pip install -r requirements.txt
```

Install NVIDIA/apex

```bash
conda install -c conda-forge nvidia-apex 
```

### **Load Dataset**

You can use [Huggingface Dataset](https://huggingface.co/docs/datasets/loading_datasets.html) to load Doc2Dial datasets. The latest [source code](https://github.com/huggingface/datasets/tree/master/datasets/doc2dial) includes the code for loading Doc2Dial v1.0.1.  

The [script](scripts/sharedtask_utils.py) shows how to obtain the ground truth of the given IDs for evaluations of Subtask 1 and Subtask 2. IDs are `{dial_id}_{turn_id}`, where `turn_id` is of the turn right before the next agent turn for grounding prediction (Subtask 1) or generation (Subtask 2). For the withheld test set for the [challenge](https://eval.ai/web/challenges/challenge-page/793/overview), the data was collected in the same process as training and validation sets; the ground truth would be obtained the same way as in the [script](scripts/sharedtask_utils.py).

### **Run Baseline for Subtask 1**

> Run [HuggingFace QA](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering) on Doc2Dial

- For fine-tuning Bert on Doc2Dial,

    ```bash
    cd sharedtask-dialdoc2021/scripts/subtask1
    ./run_qa.sh
    ```

- Results on validation set:

    ```bash
    # bert-base-uncased
    f1 = 56.29 
    exact_match = 39.73
    # bert-large-uncased-whole-word-masking
    f1 = 62.98
    exact_match = 50.50
    ```

> Evaluating your model output

- Output format and sample file

    Please see the format in [sample file](scripts/sample_files/sample_prediction_subtask1.json).

- Evaluation script

    Please refer to [`script`](scripts/sharedtask_utils.py) for evaluating your model predictions.

    ```bash
    python sharedtask_utils.py --task subtask1 --prediction_json sample_files/sample_prediction_subtask1.json
    ```

### **Run Baseline for Subtask 2**

> Run [HuggingFace Seq2Seq](https://github.com/huggingface/transformers/tree/master/examples/seq2seq) on Doc2Dial

- For generating input files,

    We first create source and target files. Please see run [script](scripts/subtask2/seq2seq_utils.py) with required parameters along with other default values.

    ```bash
    cd scripts/subtask2
    python seq2seq_utils.py --split validation --output_dir seq2seq_files
    ```

- For fine-tuning bart on Doc2Dial,

    ```bash
    cd scripts/subtask2
    ./run_seq2seq.sh
    ```

- Results on validation set:

    ```bash
    # bart-large-cnn
    val_bleu = 17.72
    ```

> Evaluating your model output

- Output format and sample file

    Please see the format in [sample file](scripts/sample_files/sample_prediction_subtask2.json).

- Evaluation script

    Please refer to [script](scripts/sharedtask_utils.py) for evaluating your model predictions.

    ```bash
    python sharedtask_utils.py --task subtask2 --prediction_json sample_files/sample_prediction_subtask2.json
    ```

## About Participation

For more up-to-date information about participating DialDoc21 Shared Task, please refer to our [workshop page](https://doc2dial.github.io/workshop2021/shared.html).
