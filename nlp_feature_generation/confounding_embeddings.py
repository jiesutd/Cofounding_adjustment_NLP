"""
//******************************************************************************

// FILENAME:           confounding_embeddings.py

// DESCRIPTION:        This Natural Language Processing Python code is the

//                     embeddings service for the Confounding Project.

// PROJECT:            HDPA Confounding Project (Joshua Lin PI, Li Zhou co-I)

// CREATION DATE:      June 1, 2020

// AUTHOR(s):          Jie Yang, Joseph M. Plasek, Kerry Ngan

// LAST MODIFIED DATE: 7/6/2022

// LAST MODIFIED BY:   Joseph M. Plasek

// Copyright (C) The Brigham and Women’s Hospital, Inc. 2020-2021 All Rights Reserved

//                     MTERMS Lab (Contact: bwhmterms@bwh.harvard.edu)

//                     Department of General Internal Medicine and Primary Care

//                     Brigham and Women's Hospital

//                     Mass General Brigham

// CONFIDENTIAL: DO NOT USE OR DISTRIBUTE IN WHOLE OR IN PART without prior 

// authorization from Li Zhou (lzhou@bwh.harvard.edu)

// To be used solely for the following permitted purpose: analysis related to the Confounding Project.

//******************************************************************************

"""

"""
First, Install SparkNLP and NLU (open source versions from John Snow Labs)
See latest install instructions at:
https://nlp.johnsnowlabs.com/docs/en/install
https://nlu.johnsnowlabs.com/docs/en/install
java -version
should be Java 8 (Oracle or OpenJDK)
To downgrade to the right verison, add the following to your .bashrc:
export JAVA_HOME="/usr/lib/jvm/java-1.8.0-openjdk-amd64"
export PATH=$JAVA_HOME/bin:$PATH
pip install spark-nlp==3.2.3 pyspark==3.1.2 nlu==3.2.1 sparknlp==1.0.0
"""




# Load necessary libraries
import pandas as pd

import numpy as np

import sys

import os

import time
start_time = time.time()

import datetime

import string 

translator = str.maketrans('', '', string.punctuation)


from tqdm import tqdm

import csv

import json 

import nlu

import sparknlp 

from sparknlp.pretrained import PretrainedPipeline

import ast 

import re

from pathlib import Path

from glob import glob

from pprint import pprint

from multiprocessing import Process, Pool



from pprint import pprint
from nltk import sent_tokenize
import asyncio


import asyncio
from glob import glob
import sparknlp
import nlu
#from johnsnowlabs import sparknlp
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from pprint import pprint
np.set_printoptions(threshold = np.inf, suppress=True)


def asynchronous(func):
    """
    This decorator converts a synchronous function to an asynchronous

    """
    
    async def wrapper(*args, **kwargs):
        await asyncio.to_thread(func, *args, **kwargs)

    return wrapper



#Load the .json file with the settings for the experiment you want to run
project_settings_json = r'/path_to_configuration_file/config.json'

#Adjust max size for csv files as our data is bigger than the default
#import sys, csv, ctypes as ct
#csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
csv.field_size_limit(sys.maxsize)

#Create a data structure to store data about each patient, including a list of notes

class Patient:

    def __init__(self, P_ID, expdate, EMPI, outcome):

        self.P_ID = P_ID

        self.expdate = expdate

        self.EMPI = EMPI

        self.outcome = outcome

        self.notelist = [] #list of notes from Note class

    def __repr__(self):

        return 'Test'




#Create a data structure to store each note
class Note:

    def __init__(self, notetype, empi, date, text):

        self.type = notetype

        self.EMPI = empi

        self.date = date 

        self.text = text



## Import Patient Dictionary from csv created from running RPDR_loader and joining with outcome table
# The goal is to only import data that you need in order to do the analysis of interest, 
# rather than all data or data in raw RPDR format
#This is just an example of the SQL query and the variable names are the same as the headings used in the original files:
"""
SELECT p_id,

date('1/1/2007') + oc30 as end_date,

pul.empi,

case when o = '1' then '1' else '0' end as o,

report_type, date(report_date_time) as note_date, notetxt

FROM hdpa_statin.lg436_050420114443901239_pul pul

inner JOIN hdpa_statin.statin_cohort_lily_csv statin

on pul.empi = statin.empi and date(report_date_time) between date('1/1/2007') + oc30 - 365 and date('1/1/2007') + oc30

;
"""
#Use the above query to extract .csv's for each RPDR note type separately (as they'll each be stored in their own files)
#RPDR files that contain free text notes of interest for the confounding project include:
"""
                        "Car": "Cardiology",
                        "Dis": "Discharge",
                        "End": "Endoscopy",
                        "Hnp": "HistoryPhysical",
                        "Lno": "LMRNotes",
                        "Mic": "Microbiology",
                        "Opn": "Operative",                       
                        "Pat": "Pathology",
                        "Phy": "History",
                        "Prg": "ProgressNotes",
                        "Pul": "Pulmonary",
                        "Rad": "RadiologyReports",
                        "Vis": "Visits"
"""
def import_patient_dict(note_path_prefix):
    
    ## update patient dict by extracting features from note folder

    start_time = time.time()

    print("Import patient dict...")
    patientdict = {}
    directory = note_path_prefix

    file_count = 0 

    for subdir, dirs, files in os.walk(directory): 

        valid_filename_list = []

        for filename in files:

            #Added starts with 2 to get timing on smaller files
            if filename.endswith(".csv"):

                valid_filename_list.append(filename)

        print("     valid file name number:", len(valid_filename_list))

        for filename in valid_filename_list:

            file_count += 1 
          
            textfile = os.path.join(subdir, filename)

            # print("     working on file:", textfile)

            ## for each note, read the freetext and split out structured fields from free-text 
            #i.e., extract EMPI, date, and free-text separately to build the Note object.

            with open(textfile, 'r', encoding="utf8") as csv_file:

                print("     Start file {}".format(textfile))
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                                                
                        empi = row[2]
                        
                        expdate = datetime.datetime.strptime(row[1], '%Y-%m-%d').date()
                        
                        if empi not in patientdict:
                            #Patient(P_ID, expdate, EMPI, outcome):
                            patientdict[empi] = Patient(row[0], expdate, empi, row[3])
                        
                        rdate = datetime.datetime.strptime(row[5],'%Y-%m-%d').date()
                        
                        #note = Note(filename, empi, rdate, rtext)
                        note = Note(row[4], empi, rdate, row[6])
                        
                        patientdict[empi].notelist.append(note)


                        line_count += 1
                print(f'Processed {line_count} lines.')
                
    print("Patient dict updated: %s s"%(time.time()-start_time))

    return patientdict

def SASdate(datein):
    origin = pd.to_datetime('1960-1-1')
    delta:pd.Timedelta = pd.to_datetime(datein) - (origin)
    return delta.days

def key_sort(x):
    num = Path(x).stem.split('_')[-1]
    sub = None
    if '-' in num:
        num = num.replace('-','.')
    return float(num)


def count_patient_number_with_notes( patientdict, span_start, span_end):

    start_time = time.time()

    print("Generating final output file...")

    start_time = time.time()

    unique_EMPI_dict = {}

    with_valid_note_count = 0

    for key in patientdict:

        if len(patientdict[key].notelist) > 0 :

            with_valid_note_count += 1

        for i in range(len(patientdict[key].notelist)):

            span = abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days)

            if (span < span_end and span >= span_start):

                if patientdict[key].EMPI in unique_EMPI_dict:

                    unique_EMPI_dict[patientdict[key].EMPI] += 1

                else:

                    unique_EMPI_dict[patientdict[key].EMPI] = 1

    print("     Number of unique EMPI:", len(unique_EMPI_dict), "; patient with valid note:", with_valid_note_count)

    print("     Feature updated: %s s"%( time.time()-start_time))

def filter_dict(patientdict, span_start, span_end, spark):
    patientembeddings = []
    for key in list(patientdict.keys()):

        patientdict[key].notelist.sort(key=lambda x: x.date, reverse=True)

        for i in range(len(patientdict[key].notelist)):

            span = (patientdict[key].expdate - patientdict[key].notelist[i].date).days

            if (span < span_end and span >= span_start):
                    predictions = {}

                    text = patientdict[key].notelist[i].text
                    #print(text)
                    
                    #append EMPI to predictions panda dataframe
                    predictions['EMPI'] = patientdict[key].EMPI
                    
                    #append exposure daet to predictions panda dataframe
                    predictions['expdate'] = patientdict[key].expdate
                                
                    #append notedate to predictions panda dataframe
                    predictions['notedate'] = patientdict[key].notelist[i].date
                    
                    #append difference in days between exposure date and note date to predictions panda dataframe
                    predictions['diffdays'] = abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days)
                    
                    #append note type to predictions panda dataframe
                    predictions['notetype'] = patientdict[key].notelist[i].type
                   
                    #append outcome to predictions pandas dataframe
                    predictions['Outcome'] = patientdict[key].outcome

                    predictions['text'] = text

                    #print(predictions)

                    patientembeddings.append(predictions)
                
    
    return patientembeddings


#This function uses the opensource nlu package from John Snow Labs, which outputs a pandas dataframe
#These are open source general language models only, as the clinical models are licensed
#NLU is a single (1) line of code  black box implemenation that does all of the preprocessing, etc behind the scenes
#Available currently: sentence_detector pos use glove bert elmo electra xlnet biobert covidbert embed_sentece.bert

def generate_standard_embeddings(df_patients:pd.DataFrame, use_multi, embeddings,pipe)->pd.DataFrame:
    
    start_time = time.time()
    
    #pd.set_option('display.width', None)
    #pd.set_option("display.max_colwidth", None)

    
    #Print list of available embeddings in NLU
    #nlu.print_all_model_kinds_for_action('embed')
    
    #just load the model once and use it for all patients/notes
    #pipe = nlu.load('sentence_detector pos use glove bert elmo electra xlnet biobert covidbert embed_sentece.bert')
    
    print("loaded embeddings: ",embeddings)

    
    #We are interested in the document level output
    level = None
    if 'sent' in embeddings:
        level = 'document' # token or document
    else:
        level = 'token'
    print(f"Size: {len(df_patients)}")
    print(f"level: {level}")
    print(f'columns: {df_patients.columns}')

    prediction = pipe.predict(df_patients, output_level = level, multithread = True, positions=True)

    print('Dropping irrelevant columns...')
    try:
        prediction.drop(['text'],axis=1,inplace=True)
        prediction.drop(['document_begin'],axis=1,inplace=True)
        prediction.drop(['document_end'],axis=1,inplace=True)
        prediction.drop(['sentence_begin'],axis=1,inplace=True)
        prediction.drop(['sentence_end'],axis=1,inplace=True)
        

    except:
        print(f"Could not drop columns")

    
    print("Generating Embeddings took: %s s"%(time.time()-start_time))
    
    del df_patients 

    return prediction



def write_toParquet(patientembeddings:pd.DataFrame, path,i,name, group_size = 1000):
    
    path = path+name +f'_{i}.parquet'
    print('Writing to: ', path)
    file_name = Path(path).stem
    parent = Path(path).parent.name
    path = Path('/netapp3/raw_data3/joshlin-team/embeddings/').joinpath(parent, file_name)

    patientembeddings.to_parquet(path, row_group_size=group_size)

    
    print(f'Write parquet success!', path)


def get_entire_row(pos, data):
    result = {}
    for k,v in data.items():
        col, i = k
        if pos == i:
            result[k] = v
    return result


def getHeader(embeddings, output_directory, task): 
    path = None
    header = None
    if "pos" in embeddings:
        print("pos")
        path = output_directory + task+'_embeddings_token_pos_jsl'
        header = ["EMPI","expdate","notedate","diffdays","notetype","pos"]
        
    if "use" in embeddings:
        print("USE embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","sentence_embedding_use"]
        path = output_directory + task+'_embeddings_sentence_use_jsl'

    if "glove" in embeddings:
        print("glove embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","word_embedding_glove"]
        path = output_directory + task+'_embeddings_word_glove_jsl'
                    
    if " bert" in embeddings:
        print("BERT embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","word_embedding_bert"]
        path = output_directory + task+'_embeddings_word_bert_jsl'
    
    if "elmo" in embeddings:
        print("ELMO embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","word_embedding_elmo"]
        path = output_directory + task+'_embeddings_word_elmo_jsl'
    
    if "electra" in embeddings:
        print("ELECTRA embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","word_embedding_electra"]
        path = output_directory + task+'_embeddings_word_electra_jsl'
    
    if "xlnet" in embeddings:
        print("XLnet embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","word_embedding_xlnet"]
        path = output_directory + task+'_embeddings_word_xlnet_jsl'
        
    if "biobert" in embeddings:
        print("BioBERT embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","word_embedding_biobert"]
        path = output_directory + task+'_embeddings_word_biobert_jsl'
    
    if "embed_sentence.bert" in embeddings:
        print("embed_sentence.bert embeddings")
        header = ["EMPI","expdate","notedate","diffdays","notetype","sentence_embedding_bert"]
        path = output_directory + task+'_embeddings_sentence_bert_jsl'
    return path, header

def get_sum(weights):
    result = 0
    for weight_list in weights:
        result += len(weight_list)
    return result






def main(project_settings_json_list, start = 0, end = -1):

    # import sparknlp_jsl
    # params = {
    #     "spark.driver.memory":"16G",
    #       "spark.kryoserializer.buffer.max":"2000M",
    #       "spark.driver.maxResultSize":"2000M", 
    #       "spark.driver.extraJavaOptions":"-Djava.io.tmpdir=/path_to_temp/temp",
    #       "spark.executer.extraJavaOptions":"-Djava.io.tmpdir=/path_to_temp/temp"

    #       }
    # # https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/20.SentenceDetectorDL_Healthcare.ipynb#scrollTo=0I_C8p8lrj02
    # # looks like we need a liscence to use this one
    # spark = sparknlp_jsl.start(params=params)



    # spark = sparknlp.SparkSession.builder \
    #         .appName("Spark NLP") \
    #         .master("local[*]") \
    #         .config("spark.driver.memory", "16G") \
    #         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    #         .config("spark.kryoserializer.buffer.max", "2000M") \
    #         .config("spark.driver.maxResultSize", "0") \
    #         .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.2.3") \
    #         .config('spark.driver.extraJavaOptions','-Djava.io.tmpdir=/path_to_temp/temp')\
    #         .config('spark.executer.extraJavaOptions','-Djava.io.tmpdir=/path_to_temp/temp')\
    #         .getOrCreate()
    # # change to .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:5.2.3") \ #uncomment to use gpu
    

    
    use_gpu = True
    # spark = sparknlp.start(gpu = use_gpu, cluster_tmp_dir='/path_to_temp/temp', log_folder='/path_to_temp/temp', memory='1000G', cache_folder='/path_to_temp/temp')
    # print(sparknlp.version())
    # print(spark.version)
    
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    print(__location__)

    

    for project_settings_json in project_settings_json_list:
        with open( os.path.join(__location__, project_settings_json)) as f:
            project_settings = json.load(f)
            print("project", project_settings['SOURCE']) #full path to directory list
        
        source_paths = project_settings['SOURCE']
        #"SOURCE_FILE_SUFFIX": ["Car.txt","Dis.txt","End.txt","Hnp.txt","Lno.txt","Mic.txt","Opn.txt","Pat.txt","Prg.txt","Pul.txt","Rad.txt","Vis.txt"],
        source_file_suffix = project_settings['SOURCE_FILE_SUFFIX'] # to get target files such as car.txt, vis.txt...
        note_path_prefix = project_settings['NOTE_DIR'] #full path to directory where notes aer stored
        output_directory = project_settings['OUTPUT_DIR'] #full path to directory where you want output to go
        target_table_prefix = project_settings['TARGET_TABLE_PREFIX'] #suggest set to cohort_ "STATIN_"
        task = project_settings['TASK'] #cohort name like "STATIN"
        algorithm = project_settings['ALGORITHM'] #include "embedding" to run this code
        span_start = int(project_settings['SPAN_START']) #set to "0"
        span_end = int(project_settings['SPAN_END']) #set to "365"
        embeddings = project_settings['EMBEDDINGS'] #available: "sentence_detector pos use glove bert elmo electra xlnet biobert covidbert embed_sentence.bert".predict('I love data science!', output_level='token', output_positions=True)
        output_elements = project_settings['OUTPUT_ELEMENTS']
        ## extract 

        print("Task: ", task)
        
        print("loaded embeddings: ",embeddings)

        pipe = nlu.load(embeddings, gpu = use_gpu)

        #patientdict = import_patient_dict(note_path_prefix)
                         
        #run pretrained word embeddings (not tuneable)
        if 'embedding' in algorithm:
            print("Embeddings: ",embeddings)
            
                
                
            path, header = getHeader(embeddings, output_directory, task)
            
            pp = Path(output_directory).name
     
            # Notes were preprocessed and split into sentences
            split_sentences = fr'path_to_data_with_split_sentences'
            new_patientdict = []
            print(rf'Looking for {split_sentences}')
            file_paths = glob(split_sentences)
            for files in glob(split_sentences):
                new_patientdict.append(pd.read_pickle(files))

            if not new_patientdict:
                raise Exception('No Data Found')
     
            patientdict = new_patientdict 
            
            print(f'Starting at index: {start}\nEnding at index: {end}\nTotal length: {len(patientdict)}')

            if end == -1 or end > len(patientdict):
                end = len(patientdict)
            

            for i in range(start, end):

                patientembeddings = generate_standard_embeddings(patientdict[i], True, embeddings, pipe)
                if 'sent' in embeddings:
                    patientembeddings['sentence_embedding_bert'] = patientembeddings['sentence_embedding_bert'].apply(lambda x: x.flatten())
                
                if patientembeddings is None:
                    raise Exception("There are no embeddings")
                
                num = Path(file_paths[i]).stem.split('_')[-1]
           
                    
                print("Writing to parquet...")
                if 'sent' in embeddings:
                    write_toParquet( patientembeddings,path,num,'')
                else:
                    write_toParquet( patientembeddings,path,num,'', group_size=200000)


    

if __name__ == '__main__':
    
    
    args = sys.argv[1:]
    path_list = None
    start = None
    end = None
    if len(args) > 3:
        raise Exception('There are too many arguments')
    else:
        path_list, start, end,  = args
        path_list = [path_list]
        start = int(start)
        end = int(end)
        
    main(path_list, start, end)
   

