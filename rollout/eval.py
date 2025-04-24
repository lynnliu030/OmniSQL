import json
import sqlite3
import os
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import sys

def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-"*30)
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        res = (data_idx, db_file, sql, None, 0)

    return res

def load_json_file(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def execute_sql(data_idx, db_file, sql):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        execution_res = frozenset(execution_res) # make set hashable
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1
    except:
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, None, 0
    
def compare_sql(question_id, db_file, ground_truth, pred_sql) :
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    correctness = 0

    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(pred_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        print('Successfully executed')
        if set(predicted_res) == set(ground_truth_res):
            correctness = 1
        conn.rollback()
    except:
        conn.rollback()
    finally:
        conn.close()
    return question_id, db_file, ground_truth, pred_sql, correctness

def execute_callback_evaluate_sql(result):
    '''Store the execution result in the collection'''
    question_id, db_file, ground_truth, pred_sql, correctness = result
    # evaluation_res = dict()
    # evaluation_res['question_id'] = question_id
    # evaluation_res["db_file"] = db_file
    # evaluation_res["question"] = question
    # evaluation_res["ground_truth"] = ground_truth
    # evaluation_res["pred_sql"] = pred_sql
    # evaluation_res["correctness"] = correctness
    evaluation_results.append(
        {
            "question_id": question_id,
            "db_file": db_file,
            "ground_truth": ground_truth,
            "pred_sql": pred_sql,
            "correctness": correctness
        }
    )

    print('Done (question_id, correctness)', question_id, correctness) # Print the progress
    sys.stdout.flush()
    sys.stderr.flush()
    
def compare_sql_wrapper(args, timeout):
    '''Wrap execute_sql for timeout'''
    try:
        result = func_timeout(timeout, compare_sql, args=args)
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = (*args, 0)
    except Exception as e:
        result = (*args, 0)
    return result

def execute_callback_execute_sqls(result):
    data_idx, db_file, sql, query_result, valid = result
    print('Done:', data_idx) # Print the progress

    execution_results.append(
        {
            "data_idx": data_idx,
            "db_file": db_file,
            "sql": sql,
            "query_result": query_result,
            "valid": valid
        }
    )

def evaluate_sqls_parallel(db_files, pred_sqls, ground_truth_sqls, num_cpus=1, timeout=1):
    '''Execute the sqls in parallel'''
    pool = mp.Pool(processes=num_cpus)
    for question_id, db_file, pred_sql, ground_truth in zip([x for x in range(len(db_files))], db_files, pred_sqls, ground_truth_sqls):
        pool.apply_async(compare_sql_wrapper, args=((question_id, db_file, ground_truth, pred_sql), timeout), callback=execute_callback_evaluate_sql)
    pool.close()
    pool.join()

def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=1):
    pool = mp.Pool(processes=num_cpus)
    for data_idx, db_file, sql in zip(list(range(len(sqls))), db_files, sqls):
        pool.apply_async(execute_sql_wrapper, args=(data_idx, db_file, sql, timeout), callback=execute_callback_execute_sqls)
    pool.close()
    pool.join()

import argparse

if __name__ == "__main__":
    # get current file directory 
    current_dir = os.path.dirname(os.path.realpath(__file__))
    results = load_json_file(os.path.join(current_dir, "rollouts/Qwen2.5-Coder-32B-Instruct-5-turns-test.json"))
    # results = load_json_file(os.path.join(current_dir, "rollouts/gpt-4o-responses-turn10.json"))
    
    # NOTE: take input path 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_file", type=str, help="Path to the rollout JSON file")
    # args = parser.parse_args()
    # results = load_json_file(args.input_file)
    
    batch_db_files = []
    batch_sqls = []
    execution_results = []

    db_files = [os.path.join("/shared/dcli/lshu/BIRD/RLEF/bird-table/data/SynSQL-2.5M/databases", result["db_id"], result["db_id"] + ".sqlite") for result in results]
    ground_truth_sqls = [result["pred_sqls"][0] for result in results]
    pred_sqls = [result["sql"] for result in results]
    num_cpus = 32
    timeout = 10
    
    evaluation_results = []
    evaluate_sqls_parallel(db_files, pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)

    # sort evaluation_results by question_id
    evaluation_results = sorted(evaluation_results, key=lambda x:x['question_id'])
    evaluation_scores = [res["correctness"] for res in evaluation_results]
    for res in evaluation_results:
        if res["correctness"] == 0:
            print("GT:", res["ground_truth"])
            print("Pred:", res["pred_sql"])
            print("-"*30)
    print("EX Accuracy (greedy search):", sum(evaluation_scores)/len(evaluation_scores))
    
    
    