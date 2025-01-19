import argparse
import sys
import subprocess
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import os
import fnmatch
from itables import init_notebook_mode
import openpyxl
from time import process_time

parser = argparse.ArgumentParser()
parser.add_argument("src", help="Designates the path to the data with the parquet files")
parser.add_argument("lvl", help="Designates the level upon which the clustering is quantified (e.g. document, sentence, token) as a variable")
parser.add_argument("var", help="Designates the variable upon which the clustering is supposed to be run")
parser.add_argument("-p", "--parquets", help="Designates the parquet filename in which you would like to read")
parser.add_argument("-c", "--cluster", help="Designates the size of the cluster(s) of interest of interest. Default is [500,1000,2000,5000,10000]")
parser.add_argument("-b", "--batch_size", help="Designates the batch size to be used for the clustering workflow. Default is 20480")
args = parser.parse_args()

print("File path is " + args.src)
print("Analysis level is " + args.lvl)
print("Variable of interest is " + args.var)
print("Filenames will use the format " + args.parquets)

init_notebook_mode(all_interactive=True)
pd.set_option('display.max_rows', None)

def file_extractor(file_dir):
    if "*" in file_dir:
        matched_file_list = []
        fold_dir, file_pattern = file_dir.rsplit('/',1)
        for each_file in os.listdir(fold_dir):
            if fnmatch.fnmatch(each_file, file_pattern):
                matched_file_list.append(fold_dir+"/"+each_file)
        return matched_file_list
    else:
        return [file_dir+a for a in os.listdir(file_dir)]
    

# Load files and building KMeans cluster model, using MiniBatchKMeans to improve speed.
    
def parquet_emb_clustering(cluster_num, batch_size,  dir):
    print("Clustering embeddings fro parquet files with kmeans...")
    print("  Cluster num: %s, batch size: %s \n  Directory: %s"%(cluster_num, batch_size, dir))
    file_list = file_extractor(dir)
    file_count = len(file_list)
    print("         Extracted %s files from dir %s"%(file_count, dir))
    kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=0, batch_size=batch_size, n_init="auto")

    total_line = 0
    for the_file in tqdm(file_list):
        df = pd.read_parquet(the_file, columns=[args.var])
        df = df[pd.notnull(df[args.var])]
        emb_array = np.array(df[args.var].tolist())
        total_line += emb_array.shape[0]
        kmeans = kmeans.partial_fit(emb_array)
    print("Training complete on %s files with %s data points"%(file_count, total_line))
    return kmeans

# Use the trained KMeans model to generate a sample file, that provide examples for each cluster. 

def generate_sample_file_from_cluster_model(model, data_file, output_file, sample_num=20):
    cluster_num = model.cluster_centers_.shape[0]
    df = pd.read_parquet(data_file, columns=[args.lvl,args.var])
    df = df[pd.notnull(df[args.var])]
    emb_array = np.array(df[args.var].tolist())
    print("File shape:",emb_array.shape)
    last_cluster = model.predict(emb_array)
    df['cluster'] = last_cluster
    for idx in range(cluster_num):
        uniques = df[df['cluster']==idx][args.lvl].unique()
        min_size = min(sample_num, uniques.shape[0])
        samples = np.concatenate((uniques[:min_size], np.empty([sample_num-min_size,])))
        column_name = 'Cluster:'+str(idx)+":unique:"+str(uniques.shape[0])+":sent:"+str(df[df['cluster']==idx][args.lvl].shape[0])
        if idx == 0:
            cluster_sample_df = pd.DataFrame(columns =[column_name])
        cluster_sample_df[column_name] = samples
    cluster_sample_df.to_excel(output_file)
    
# Use the trained cluster model to assign cluster to each sentence.

# Aggregate the cluster result into patient + index date level, using <Participant_ID_Variable>. remove duplicates and count positive cluster number for each <Participant_ID_Variable>
    
def assign_cluster_labels(model, input_dir, output_file=None):
    print("Assigning cluster for parquet files with kmeans...")
    print("  Cluster num: %s \n  Directory: %s"%(model.cluster_centers_.shape[0],  input_dir))
    file_list = file_extractor(input_dir)
    file_count = len(file_list)
    dfs = []
    for each_file in tqdm(file_list):
        df = pd.read_parquet(each_file, columns=[args.var,'<Participant_ID_Variable>'])
        df = df[pd.notnull(df[args.var])]
        emb_array = np.array(df[args.var].tolist())
        last_cluster = model.predict(emb_array)
        df = df.drop([args.var], axis=1)
        df['cluster'] = last_cluster
        aggregated_df = df.groupby('<Participant_ID_Variable>').agg(lambda x: list(set(x))).reset_index()
        dfs.append(aggregated_df)
    cohort_df = pd.concat(dfs, ignore_index=True)
    cohort_df = cohort_df.groupby('<Participant_ID_Variable>').agg(lambda x: list(set(sum(x,[])))).reset_index() ## sum is to concatenate lists, set to remove duplicates
    cohort_df["postive_cluster_num"] = cohort_df['cluster'].apply(lambda x: len(x))
    if output_file:
        cohort_df.to_excel(output_file, index=False)
    return cohort_df

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
folder_dir = args.src
file_name = args.parquets
if args.batch_size:
    batch_size = args.batch_size
else:
    batch_size = 20480

if args.cluster is not None:
    print("Cluster size is set to " + args.cluster)
    args.cluster = int(args.cluster)
    if not isinstance(args.cluster, list):
        if isinstance(args.cluster, int):
            cluster_num_list = [args.cluster]
        else:
            print(args.cluster)
            print("Cluster number is not returning as an integer. Please correct this and re-run.")
else:
    cluster_num_list = [500,1000,2000,5000,10000]
    print("Cluster size is set to calculate cluster sizes of 500, 1000, 2000, 5000, and 10000")

t1_start = process_time()

for cluster_num in cluster_num_list:
    ## define file names to be saved
    sample_file_source = file_name.replace("*","0")
    sample_file = "sample_cluster_" + args.lvl + "_" + args.var + "_c%s_b%s.xlsx"%(cluster_num, batch_size)
    cluster_file = "result_cluster_" + args.lvl + "_" + args.var + "_c%s_b%s.xlsx"%(cluster_num, batch_size)
    model_file = "model_cluster_" + args.lvl + "_" + args.var + "_c%s_b%s.pkl"%(cluster_num, batch_size)
    ## training kmeans model
    kmeans = parquet_emb_clustering(cluster_num, batch_size, folder_dir+file_name)
    ## generate sample file to provide examples of each cluster
    generate_sample_file_from_cluster_model(kmeans, folder_dir+sample_file_source, folder_dir+sample_file)
    ## assign cluster and aggregate to <Participant_ID_Variable>, and save to file
    cohort_df = assign_cluster_labels(kmeans, folder_dir+file_name, folder_dir+cluster_file)
    ## save KMeans model
    with open(folder_dir+model_file, 'wb') as f:
        pickle.dump(kmeans, f)
        print("Model saved to:", folder_dir+model_file)

t1_stop = process_time()

print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
