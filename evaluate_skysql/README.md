# SkyRL-SQL-7B Evaluation

## Evaluation Reproduction
You can easily reproduce our evaluation results as follows:

1. **Set Up Environment:**
   ```sh
   conda create -n omnisql_eval python=3.9.5
   conda activate omnisql_eval
   pip3 install vllm==0.6.3.post1 func_timeout tqdm matplotlib nltk==3.8.1 sqlparse
   python3 nltk_downloader.py
   ```

2. **Download Evaluation Materials:**
   - Download Spider's test-suite databases and evaluation scripts from [test_suite_sql_eval.zip](https://drive.google.com/file/d/1iNa1WgA9tN_OFna08nq_tHZdXx9Lz2vO/view) and unzip `test_suite_sql_eval.zip` in this folder.
   - Download the dataset files from [OmniSQL-datasets](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets/tree/main) and unzip them in the `data` folder.

3. **Download the evaluation datasets:**
   We use a custom format of the evaluation datasets for evaluation. You can download the dataset from [here](https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-eval/tree/main/).

```bash
huggingface-cli download NovaSky-AI/SkyRL-SQL-eval NovaSky-AI/SkyRL-SQL-eval skyrl_eval --repo-type dataset
```

Once done you should have the following folders: 

```bash
OmniSQL/evaluate_skysql/
├── data
├── test_suite_sql_eval
└── skyrl_eval
└── ...
```

4. **Run Evaluation:**
   ```bash
   bash infer_multiturn_all.sh data test_suite_sql_eval skyrl_eval
   ```
   Evaluation results are stored in the `evaluation_results` folder.