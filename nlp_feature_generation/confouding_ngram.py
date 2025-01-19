"""

//******************************************************************************

// FILENAME:           confounding_ngram.py

// DESCRIPTION:        This Natural Language Processing Python code is the

//                     nGram and embeddings service for the Confounding Project.

// PROJECT:            HDPA Confounding Project (Joshua Lin PI, Li Zhou co-I)

// CREATION DATE:      June 1, 2020

// AUTHOR(s):          Jie Yang, Joseph M. Plasek

// LAST MODIFIED DATE: 9/30/2021

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

# Load necessary libraries
import pandas as pd

import numpy as np

import sys

import os

import time

import datetime

from nltk import word_tokenize 

from nltk.util import ngrams

from collections import Counter

import string 

translator = str.maketrans('', '', string.punctuation)

from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")
cachedStopWordsSET = set(cachedStopWords)

from tqdm import tqdm

import pickle

import csv

import json 

from collections import Counter

import multiprocessing
import mpire 
from mpire.utils import make_single_arguments
from itertools import chain
import functools, operator

#Run the following lines on the first use on a new computer system to load NLTK stopwords. 
    #import nltk

    #nltk.download('punkt')

    #nltk.download('stopwords')

#Load the .json file with the settings for the experiment you want to run
project_settings_json = r'/path_to_configuration_file/config.json'

#Adjust max size for csv files as our data is bigger than the default
import sys, csv, ctypes as ct
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
# csv.field_size_limit(sys.maxsize)



__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
#print(__location__)

with open( os.path.join(__location__, project_settings_json)) as f:
    project_settings = json.load(f)
    #print("project", project_settings['SOURCE']) #full path to directory list
    
source_paths = project_settings['SOURCE']
#"SOURCE_FILE_SUFFIX": ["Car.txt","Dis.txt","End.txt","Hnp.txt","Lno.txt","Mic.txt","Opn.txt","Pat.txt","Prg.txt","Pul.txt","Rad.txt","Vis.txt"],
source_file_suffix = project_settings['SOURCE_FILE_SUFFIX'] # to get target files such as car.txt, vis.txt...
note_path_prefix = project_settings['NOTE_DIR'] #full path to directory where notes aer stored
output_directory = project_settings['OUTPUT_DIR'] #full path to directory where you want output to go
target_table_prefix = project_settings['TARGET_TABLE_PREFIX'] #suggest set to cohort_ "STATIN_"
task = project_settings['TASK'] #cohort name like "STATIN"
preprocessor = project_settings['PREPROCESSOR'] #set to "nltk" (default) or "raw" (experimental)
algorithm = project_settings['ALGORITHM'] #include "ngram" 
span_start = int(project_settings['SPAN_START']) #set to "0"
span_end = int(project_settings['SPAN_END']) #set to "365"
ngrams_to_extract = int(project_settings['NGRAMS_TO_EXTRACT']) #set to "20000"
ngrams_extract_all = project_settings['NGRAMS_EXTRACT_ALL'] #additional output when "true"
output_elements = project_settings['OUTPUT_ELEMENTS']


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

        for filename in tqdm(valid_filename_list):

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




#This will calculate the ngram counter for each note for each patient
#this code currently does unigrams and bigrams
def calculate_ngram_counter(chunks):
    
    if(not type(chunks) is dict):
        chunks =  dict(item for item in chunks)
    
    ngcounter1 = Counter()

    ngcounter2 = Counter()

    changes = []


    
    for key in chunks:

        chunks[key].notelist.sort(key=lambda x: x.date, reverse=True)
        
        for i in range(len(chunks[key].notelist)):

            span = (chunks[key].expdate - chunks[key].notelist[i].date).days

            if (span < span_end and span >= span_start):
                
                if(preprocessor == "nltk"):

                    ttext = chunks[key].notelist[i].text.lower()
    
                    ttext = ttext.split("\n",1)[1]
      
                    ttext = ' '.join([word for word in ttext.split() if word not in cachedStopWords])
    
                    ttext = ttext.translate(translator)

                    #patientdict[key].notelist.sort(key=lambda x: x.date, reverse=True)
                    chunks[key].notelist[i].text = ttext 
                   
                    token = word_tokenize(ttext)

                    changes.append((key,i,ttext))
                    
                #the nltk one is what you want to use
                elif(preprocessor == "raw"):
                    token = word_tokenize(chunks[key].notelist[i].text.lower())
                #unigrams
                ngramlist1 = list(ngrams(token, 1))
                #bigrams
                ngramlist2 = list(ngrams(token, 2))

                ngcounter1 += Counter(ngramlist1)

                ngcounter2 += Counter(ngramlist2)
                              
                              
    # print("     --- %s seconds ---" % (time.time() - start_time))

    # print("     unigram num:",len(ngcounter1))

    # print("     bigram  num:",len(ngcounter2))
    

    # print("Ngram counter generated: %s s"%(time.time()-start_time))

    return ngcounter1, ngcounter2,changes

#This will give you all ngrams, rather than just the top K
def generate_raw_ngram_output_file(ngcounter1, ngcounter2, patientdict, task, preprocessor, output_directory):

    ## generate and save patients with unigram/bigram information

    ## featdictlist: (EMPI, unigram/bigram) -> days

    start_time = time.time()

    print("Generating final raw ngram output file...")

    start_time = time.time()


    featlist = [ngcounter1, ngcounter2] #need to change to include both if dual run

    rawfeatdictlist = []

    for j in featlist:

        rawfeatdict = {}

        top_feat = j

        unique_EMPI_dict = {}

        with_valid_note_count = 0

        for key in tqdm(patientdict):

            if len(patientdict[key].notelist) > 0 :

                with_valid_note_count += 1

            for i in range(len(patientdict[key].notelist)):

                span = abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days)

                if (span < span_end and span >= span_start):

                    ttext = patientdict[key].notelist[i].text

                    for featkey in top_feat:

                        if (patientdict[key].EMPI, featkey) not in rawfeatdict:

                            if " ".join(featkey) in ttext:
                                
                                #Choose what you want to output. Default to enhanced output format
                                empi_exp_date = patientdict[key].EMPI + str(patientdict[key].expdate.strftime("%m%d%Y"))

                                if output_elements == "empi":
                                    
                                    rawfeatdict[(patientdict[key].EMPI, featkey)] = span
                                    
                                elif output_elements == "empi_expdate":

                                    rawfeatdict[(patientdict[key].EMPI, patientdict[key].expdate, featkey)] = span
                                    
                                elif output_elements == "empi_expdate_notedate":
                                    
                                    rawfeatdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, featkey)] = span
                                
                                elif output_elements == "empi_expdate_notetype":
                                    
                                    rawfeatdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].type, featkey)] = span
                                                                    
                                elif output_elements == "empi_expdate_notedate_notetype":
                                    
                                    rawfeatdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, patientdict[key].notelist[i].type, featkey)] = span
                                                                    
                                elif output_elements == "empi_expdate_notedate_diff_notetype":
                                    
                                    rawfeatdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days), patientdict[key].notelist[i].type, featkey)] = span
                                    
                                elif output_elements == "comb_empi_expdate_notedate_diff_notetype":
                                    
                                    rawfeatdict[(empi_exp_date, patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days), patientdict[key].notelist[i].type, featkey)] = span
                                
                                else:
                                    
                                    rawfeatdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days), patientdict[key].notelist[i].type, featkey)] = span


                                if patientdict[key].EMPI in unique_EMPI_dict:

                                    unique_EMPI_dict[patientdict[key].EMPI] += 1

                                else:

                                    unique_EMPI_dict[patientdict[key].EMPI] = 1

        print("     Number of unique EMPI:", len(unique_EMPI_dict), "; patient with valid note:", with_valid_note_count)

        print("     Number of featuredict (empi, feature):span:", len(rawfeatdict))

        print("     Feature updated: %s s"%( time.time()-start_time))

        print("     Number of unique EMPI:", len(unique_EMPI_dict))

        rawfeatdictlist.append(rawfeatdict)



    for i in range(len(rawfeatdictlist)):

        filesave = output_directory + task+ '_' + preprocessor +'{}gramraw_'.format((i+1)) + output_elements +'.csv'

        with open(filesave, 'w') as fp:
            
            writer = csv.writer(fp)

            writer.writerows(rawfeatdictlist[i])

    print("All file generated: %s s"%(time.time()-start_time))



def generate_final_output_file(top_feat, patientdict):

    patientdict =  dict(item for item in patientdict)

      ## generate and save patients with unigram/bigram information

    ## featdictlist: (EMPI, unigram/bigram) -> days


    #print("Generating final output file...")

    # start_time = time.time()

    # top_feat1 = ngcounter1.most_common(ngrams_to_extract)

    # top_feat1 = Counter(dict(top_feat1))

    # top_feat2 = ngcounter2.most_common(ngrams_to_extract)

    # top_feat2 = Counter(dict(top_feat2))

    featlist = [top_feat] #need to change to include both if dual run

    featdictlist = []

    featdict = {}

    unique_EMPI_dict = {}

    with_valid_note_count = 0

    for j in featlist:

        top_feat = j

        #for key in tqdm(patientdict):
        for key in patientdict:

            if len(patientdict[key].notelist) > 0 :

                with_valid_note_count += 1

            for i in range(len(patientdict[key].notelist)):

                span = abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days)
                
                #print("EMPI: " + str(patientdict[key].EMPI) + " NoteID: " + str(i) + " ExposureDate: " + str(patientdict[key].expdate) +  " NoteDate: " + str(patientdict[key].notelist[i].date) + " Span: " + str(span) + " Included: " + str(span < span_end and span >= span_start))


                if (span < span_end and span >= span_start):

                    ttext = patientdict[key].notelist[i].text

                    for featkey in top_feat:

                        if (patientdict[key].EMPI, featkey) not in featdict:

                            if " ".join(featkey) in ttext:
                                #Choose what you want to output. Default to enhanced output format
                                empi_exp_date = patientdict[key].EMPI + str(patientdict[key].expdate.strftime("%m%d%Y"))
                                
                                if output_elements == "empi":
                                    
                                    featdict[(patientdict[key].EMPI, featkey)] = span
                                    
                                elif output_elements == "empi_expdate":

                                    featdict[(patientdict[key].EMPI, patientdict[key].expdate, featkey)] = span
                                    
                                elif output_elements == "empi_expdate_notedate":
                                    
                                    featdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, featkey)] = span
                                
                                elif output_elements == "empi_expdate_notetype":
                                    
                                    featdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].type, featkey)] = span
                                                                    
                                elif output_elements == "empi_expdate_notedate_notetype":
                                    
                                    featdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, patientdict[key].notelist[i].type, featkey)] = span
                                                                    
                                elif output_elements == "empi_expdate_notedate_diff_notetype":
                                    
                                    featdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days), patientdict[key].notelist[i].type, featkey)] = span
                                    
                                elif output_elements == "comb_empi_expdate_notedate_diff_notetype":
                                    
                                    featdict[(empi_exp_date, patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days), patientdict[key].notelist[i].type, featkey)] = span
                                
                                else:
                                    
                                    featdict[(patientdict[key].EMPI, patientdict[key].expdate, patientdict[key].notelist[i].date, abs((patientdict[key].expdate - patientdict[key].notelist[i].date).days), patientdict[key].notelist[i].type, featkey)] = span

                                if patientdict[key].EMPI in unique_EMPI_dict:

                                    unique_EMPI_dict[patientdict[key].EMPI] += 1

                                else:

                                    unique_EMPI_dict[patientdict[key].EMPI] = 1
    
    return featdict, unique_EMPI_dict, with_valid_note_count




def count_patient_number_with_notes( patientdict):

    start_time = time.time()

    print("Generating final output file...")

    start_time = time.time()

    unique_EMPI_dict = {}

    with_valid_note_count = 0

    for key in tqdm(patientdict):

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


def print_final_message(featdict, unique_EMPI_dict, with_valid_note_count, start_time, msg = ''):
    print(msg)

    print("     Number of unique EMPI:", len(unique_EMPI_dict), "; patient with valid note:", with_valid_note_count)

    print("     Number of featuredict (empi, feature):span:", len(featdict))

    print("     Feature updated: %s s"%( time.time()-start_time))

    print("     Number of unique EMPI:", len(unique_EMPI_dict))


def write_output_pickle(featdict, i):

    filesave = output_directory + task+preprocessor+'{}gram20k_jmp_norpdr.pickle'.format((i))

    with open(filesave, 'wb') as fp:

        pickle.dump(featdict, fp)
        
    

def write_output_csv(featdict, i):

    csvfilesave = output_directory + task + '_' + preprocessor +'{}gram20k_'.format((i)) + output_elements +'.csv'

    with open(csvfilesave, 'w') as csvfp:
        
        writer = csv.writer(csvfp)

        writer.writerows(featdict)

if __name__ == '__main__':

    print(__location__)
    print("project", project_settings['SOURCE']) #full path to directory list

    ## extract 

    print("Task: ", task, "Output elements: ", output_elements)

    patientdict = import_patient_dict(note_path_prefix)
        
    
    #run ngrams  
    if 'ngram' in algorithm:
        ## generate unigram/bigram of the patients based on the free-text notes
        
        start_time = time.time()
        print("Calculating ngram...")

        #ngcounter01, ngcounter02 = calculate_ngram_counter0(copy.deepcopy(patientdict), preprocessor)

        poolsize = 6
        chunksize = 1
        pool = mpire.WorkerPool(poolsize)
        items = list(patientdict.items())
        chunks = [items[i:i + chunksize ] for i in range(0, len(items), chunksize)]
        

        result = pool.map_unordered(calculate_ngram_counter,make_single_arguments(chunks), progress_bar=True, progress_bar_options={'desc': 'Processing', 'unit': 'items', 'colour': 'blue'},iterable_len=len(chunks))
        pool.join()
        
        ngcounter1 = Counter()
        ngcounter2 = Counter()
       
        for n1, n2, changes in result:
            ngcounter1.update(n1)
            ngcounter2.update(n2)
            
            for c in changes:
                key,i,ttext = c
                patientdict[key].notelist.sort(key=lambda x: x.date, reverse=True)
                patientdict[key].notelist[i].text = ttext 
        
        pool.terminate()

        
        #ngcounter1, ngcounter2 = calculate_ngram_counter(patientdict) 

        print("     --- %s seconds ---" % (time.time() - start_time))
        print("     unigram num:",len(ngcounter1))
        print("     bigram  num:",len(ngcounter2))
        print("Ngram counter generated: %s s"%(time.time()-start_time))
       
        #generate and save the final raw output file.
        if ngrams_extract_all == 'true':
        
            generate_raw_ngram_output_file(ngcounter1, ngcounter2, patientdict, task, preprocessor, output_directory)
    
        ## generate and save the final output file. 

        ## featdictlist: (EMPI, unigram/bigram) -> days

        print("Generating final output file...")
        start_timef = time.time()

        
        top_feat1 = ngcounter1.most_common(ngrams_to_extract)

        top_feat1 = Counter(dict(top_feat1))

        print("     Most common unigrams complete: %s s"%( time.time()-start_timef))

        top_feat2 = ngcounter2.most_common(ngrams_to_extract)

        top_feat2 = Counter(dict(top_feat2))

        print("     Most common bigrams complete: %s s"%( time.time()-start_timef))


        #featlist = [top_feat1, top_feat2] #need to change to include both if dual run


        pool1 = mpire.WorkerPool(int(poolsize),shared_objects=top_feat1, start_method ='spawn')
        pool2 = mpire.WorkerPool(int(poolsize),shared_objects=top_feat2, start_method ='spawn')

        result1 = pool1.map(generate_final_output_file, make_single_arguments(chunks), progress_bar=True, progress_bar_options={'desc': 'Processing', 'unit': 'items', 'colour': 'blue'},iterable_len=len(chunks))
        result2 = pool2.map(generate_final_output_file, make_single_arguments(chunks), progress_bar=True, progress_bar_options={'desc': 'Processing', 'unit': 'items', 'colour': 'green'},iterable_len=len(chunks))
        
        pool1.join()
        pool2.join()

        
        featdict1 = {}
        unique_EMPI_dict1 = {} 
        with_valid_note_count1 = 0
        featdict2 = {}
        unique_EMPI_dict2 = {} 
        with_valid_note_count2 = 0
        
        for f1, u1, v1 in result1:
            featdict1.update(f1)
            unique_EMPI_dict1.update(u1)
            with_valid_note_count1 += v1

        for f2, u2, v2 in result2:
            featdict2.update(f2)
            unique_EMPI_dict2.update(u2)
            with_valid_note_count2 += v2




        print_final_message(featdict1, unique_EMPI_dict1, with_valid_note_count1, start_timef, "Unigram")
        print_final_message(featdict2, unique_EMPI_dict2, with_valid_note_count2, start_timef, "Bigram")
        
        pool1.terminate()
        pool2.terminate()


        print("Writing to output files")
        pool1 = mpire.WorkerPool(6)

        pool1.apply_async(write_output_csv,(featdict1,'test1'))
        pool1.apply_async(write_output_csv,(featdict2,'test2'))

        pool1.apply_async(write_output_pickle,(featdict1,'test1'))
        pool1.apply_async(write_output_pickle,(featdict2,'test2'))

        pool1.join()

        print("All file generated: %s s"%(time.time()-start_time))





