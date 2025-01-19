"""

//******************************************************************************

// FILENAME:           confounding_topicmodeling.py

// DESCRIPTION:        This Natural Language Processing Python code is the

//                     topic modeling service for the Confounding Project.

// PROJECT:            HDPA Confounding Project (Joshua Lin PI, Li Zhou co-PI)

// CREATION DATE:      June 1, 2020

// AUTHOR(s):          Liqin Wang, John Laurentiev, Jie Yang, Joseph M. Plasek

// LAST MODIFIED DATE: 10/21/2021

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
from asyncio import Task
import pandas as pd

import numpy as np

import sys

import os

import time

import datetime

import string 

translator = str.maketrans('', '', string.punctuation)

from tqdm import tqdm

import csv

import json 

from gensim.corpora import Dictionary

from gensim.parsing.preprocessing import preprocess_string, strip_short, remove_stopwords, strip_numeric, strip_punctuation, STOPWORDS

from gensim.models import LdaModel, LdaMulticore, CoherenceModel

from gensim.matutils import corpus2csc

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker


from numpy.polynomial.polynomial import polyfit


from pprint import pprint
from pathlib import Path
import pickle
import concurrent.futures


#Load the .json file with the settings for the experiment you want to run
project_settings_json = r'/path_to_configuration_file/config.json'

#Adjust max size for csv files as our data is bigger than the default
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


def process_file(textfile):
    patientdict = {}

    with open(textfile, 'r', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                empi = row[2]
                expdate = datetime.datetime.strptime(row[1], '%Y-%m-%d').date()

                if empi not in patientdict:
                    patientdict[empi] = Patient(row[0], expdate, empi, row[3])

                rdate = datetime.datetime.strptime(row[5],'%Y-%m-%d').date()
                note = Note(row[4], empi, rdate, row[6])
                patientdict[empi].notelist.append(note)

                line_count += 1

    return patientdict

def parallel_import_patient_dict(note_path_prefix):
    patientdict = {}

    valid_filenames = []

    for subdir, dirs, files in os.walk(note_path_prefix):
        valid_filenames.extend([os.path.join(subdir, filename) for filename in files if filename.endswith(".csv")])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, valid_filenames), total=len(valid_filenames)))

    for result in results:
        patientdict.update(result)

    return patientdict



def count_patient_number_with_notes( patientdict, span_start, span_end):

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


def generate_topic_model(patientdict, num_topics,span_start, span_end, task, iterations, passes, output_directory,
                         save_topic_model, log_topic_metrics, log_top_topics, log_topic_weights):
    progress=-1
    dataset='all'
    start_time = time.time()
    loaddict = {}
    notes = []
    texts = []
    corpus = []
    corpus_key = []
    corpus_id = []


    for key in tqdm(list(patientdict.keys())):

        #sort notes list
        patientdict[key].notelist.sort(key=lambda x: x.date, reverse=True)
        #loook at each note
        for noteID in range(len(patientdict[key].notelist)):
            #calculate span
            span = (patientdict[key].expdate - patientdict[key].notelist[noteID].date).days
            #see if span is in range
            if (span < span_end and span >= span_start):
                # text processing filters in CUSTOM_FILTERS are applied in the order they appear
                # see https://radimrehurek.com/gensim/parsing/preprocessing.html for more info on gensim preprocessing functions
                # custom functions can be applied as well
                CUSTOM_FILTERS = [
                        lambda x: x.lower(),    # set text to lower case
                        strip_punctuation,      # removes the following punctuation characters: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
                        strip_numeric,          # removes digits
                        remove_stopwords,       # see gensim documentation for all stopwords included
                        strip_short             # removes strings of length less than 3
                ]
                    
                #preprocess the notes
                token = preprocess_string(patientdict[key].notelist[noteID].text, filters=CUSTOM_FILTERS)
                
                #add preprocessed notes to notes list, which is used for matching topics to notes later on in the process                    
                notes.append((token, key+str(noteID)))
                            
                #the topic modeling code will use the corpus key (unique for each note)
                corpus_key.append((key,noteID))
                
                #also keep track of each patient 
                corpus_id.append(str(key))
                
                #keep track of the pre-processed notes
                texts.append(token)



    corpus, dictionary = create_dictionary(texts)

    
    #load the dictionary into data structure
    loaddict = {'corpus': corpus, 'dictionary': dictionary, 'corpus_key': corpus_key}
    
    ##'''train'''

    dictionary = loaddict['dictionary']
    corpus = loaddict['corpus']

    print('\ntraining {} model...'.format(task))
    print(
        datetime.datetime.now())  # datetime printing is for debugging, to see how long each step took. Remove if not needed

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token


    print('topics: {}'.format(num_topics))
    print('interations: {}'.format(iterations))
    print('passes: {}'.format(passes))
    print('vocab size: {}'.format(len(dictionary)))
    print('documents in corpus: {}'.format(len(corpus)))


    model_name = 'topic_{}_p{}_i{}_t{}.MODEL'.format(task, passes, iterations,
                                                              num_topics)  # ,str(datetime.date.today())[5:])


    model:LdaMulticore = None

    if Path(output_directory + model_name).exists():
        print(Path(output_directory + model_name))
        progress = 1
        model = LdaMulticore.load(output_directory + model_name)
        print('Model found. Skip training...')

    if progress < 0:
        print('START MODEL TRAINING...')
        ##Train new model with desired parameters (not using batch training here)
        model = LdaMulticore(
            corpus=corpus,  # leave commented out for batch training, uncomment to train on full corpus at once
            id2word=id2word,
            iterations=iterations,
            passes=passes,
            num_topics=num_topics,
            workers=1
        )

        if save_topic_model == 'true':
            try:
                print('saving model...')
                print(output_directory + model_name)
                model.save(output_directory + model_name)
                print('model saved as {}.'.format(model_name))
            except Exception as e:
                print('saving error: {}'.format(e))
    
    
    print("Generating Topic Model took: %s s"%(time.time()-start_time))

    #Extract the top topics from the trained model
    top_topics = model.top_topics(corpus)
    #print the top topics to a file
    if log_top_topics == 'true':
        print('Extracting top topics...')
        index = []
        topics=[]
        n = 0
        for i in top_topics:
            n+=1
            index.append(n)
            output=[]
            vector = i[0]
            for v in vector:
                output.append(v[1])
            topics.append(", ".join(output))
        
        topic_out_filename = os.path.join(output_directory, task+"_topics_"+str(num_topics)+".txt")
                
        with open(topic_out_filename, 'w', encoding="utf8") as topic_out_file:
            for i,v in zip(index,topics):
                topic_out_file.write("Topic %i\n%s\n"%(i,v))
    
    #This is the main output file of interest and is used by SAS
    print('STARTING TOPIC CONTRIBUTIONS! THIS TAKES LONG')
    topic_contribution_file = output_directory + "topic_contributions_" + task + "_k_" + str(num_topics) + ".txt"
    sas_topic_contribution_file = output_directory + "topic_contributions_" + task + "_k_" + str(num_topics) + ".txt"
    csv_topic_contribution_file = output_directory + "topic_contributions_" + task + "_k_" + str(num_topics) + ".csv"
    sas_lines = []
    csv_lines = []
    
    # Get main topic in each document

    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        key, noteID = corpus_key[i]

        sas_lines.append(
            str(corpus_id[i]) +
            str(patientdict[key].expdate) +
            str(patientdict[key].notelist[noteID].date) +
            str(abs((patientdict[key].expdate - patientdict[key].notelist[noteID].date).days)) +
            patientdict[key].notelist[noteID].type + 
            str(row) + '\n'
        )
        csv_lines.append([
            str(corpus_id[i]), 
            str(patientdict[key].expdate), 
            str(patientdict[key].notelist[noteID].date), 
            str(abs((patientdict[key].expdate - patientdict[key].notelist[noteID].date).days)), 
            patientdict[key].notelist[noteID].type, 
            str(row)
        ])

    print('Writing topic contributions...')                
    with open(sas_topic_contribution_file, "w") as sas_output, open(csv_topic_contribution_file,'w') as csv_output:
        csvwriter = csv.writer(csv_output, delimiter='|')
        csvwriter.writerow('empi_expdate_notedate_diff_notetype'.split('_'))  
        csvwriter.writerows(csv_lines) 
        sas_output.writelines(sas_lines)


    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Number of topics:' + str(num_topics))
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    
    # prints list of ((list of top probability,term tuples), topic coherence) tuples
    if log_topic_weights == 'true':
        print('Writing topic weights...')
        topic_output_file = output_directory + "topics_" + task + "_k_" + str(num_topics) + ".txt"
        with open(topic_output_file, "w") as topic_out_file:
            pprint(top_topics, topic_out_file)  
        
    print(datetime.datetime.now())
    
    
    
    #Log metrics about topics to file
    if log_topic_metrics == 'true':
        print('Logging metrics...')
        topic_metrics_file = output_directory + "topic_metrics_" + task + "_k_" + str(num_topics) + ".txt"
        with open(topic_metrics_file, "w") as topic_metrics_file:
            pprint('Average topic coherence'+str(avg_topic_coherence), topic_metrics_file)
            print(datetime.datetime.now(), 'Average topic coherence done!')

            coherence_model_c_v = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v', topn=20)
            coherence_c_v = coherence_model_c_v.get_coherence()
            pprint('c_v Coherence Score: '+ str(coherence_c_v), topic_metrics_file)
            coherence_c_v_pt = coherence_model_c_v.get_coherence_per_topic()
            pprint('c_v Coherence Score: '+ str(coherence_c_v_pt), topic_metrics_file)
            print(datetime.datetime.now(), 'c_v Coherence Score done!')

            coherence_model_c_uci = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_uci',topn=5)
            coherence_c_uci = coherence_model_c_uci.get_coherence()
            pprint('c_uci Coherence Score: '+ str(coherence_c_uci), topic_metrics_file)
            coherence_c_uci_pt = coherence_model_c_uci.get_coherence_per_topic()
            pprint('c_uci Coherence Score: '+ str(coherence_c_uci_pt), topic_metrics_file)
            print(datetime.datetime.now(), 'c_uci Coherence Score done!')

            coherence_model_c_npmi = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_npmi', topn=40)
            coherence_c_npmi = coherence_model_c_npmi.get_coherence()
            pprint('c_npmi Coherence Score: '+ str(coherence_c_npmi), topic_metrics_file)
            coherence_c_npmi_pt = coherence_model_c_npmi.get_coherence_per_topic()
            pprint('c_npmi Coherence Score: '+ str(coherence_c_npmi_pt), topic_metrics_file)
            print(datetime.datetime.now(), 'c_npmi Coherence Score done!')

            coherence_model_u_mass = CoherenceModel(model=model, corpus = corpus, texts=notes, dictionary=dictionary, coherence='u_mass')
            coherence_u_mass = coherence_model_u_mass.get_coherence()
            pprint('u_mass Coherence Score: '+ str(coherence_u_mass), topic_metrics_file)
            coherence_u_mass_pt = coherence_model_u_mass.get_coherence_per_topic()
            pprint('u_mass Coherence Score: '+ str(coherence_u_mass_pt), topic_metrics_file)
            perplexity = model.log_perplexity(corpus)
            pprint('perplexity: '+ str(perplexity), topic_metrics_file)
            print(datetime.datetime.now(), 'u_mass Coherence Score done!')
            print(datetime.datetime.now(), 'FINISHED!!!')
        
    return top_topics, model, corpus, notes, num_topics, id2word, texts, dictionary

def create_dictionary(corpus):
    # input: corpus: list of lists of tokens generated by preprocessing
    # output: corpus: bag of word version of input corpus
    #		dictionary: vocabulary with rare/common terms filtered out
    print('creating vocabulary and finalizing corpus...')
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=20,
                               no_above=0.45)  # changing these numbers can increase/decrease the run time if needed, but too exclusive will lead to worse results
    corpus = [dictionary.doc2bow(tokens) for tokens in corpus]

    print('vocab size: {}'.format(len(dictionary)))
    print('documents in corpus: {}'.format(len(corpus)))
    return corpus, dictionary

def test_perplexity_and_coherence_c_v(x_labels, corpus, id2word, texts, dictionary, task, iterations, passes):
    perplexity_list = []
    coherence_list = []
    font_size=20
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    for num_topics in x_labels:
        model_name = f'_t{num_topics}'
        ##Create new model with desired parameters
        lda_model = LdaMulticore(
            corpus=corpus,  # leave commented out for batch training, uncomment to train on full corpus at once
            id2word=id2word,
            iterations=iterations,
            passes=passes,
            num_topics=num_topics,
            workers=1
        )
        print("Loading model:", model_name)
        perplexity_list.append(lda_model.log_perplexity(corpus))

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        top_topics = lda_model.top_topics(corpus)
        coherence_model_c_v = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v', topn=20)
        coherence_c_v = coherence_model_c_v.get_coherence()
        coherence_list.append(coherence_c_v)

    perplexity_list = np.asarray(perplexity_list)
    coherence_list = np.asarray(coherence_list)

    # ticks
    t = np.asarray(range(len(x_labels)))

    # main plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.xticks(t, x_labels, rotation=90, fontsize=16)
    ax1.set_xlabel('Number of Predicted Topics', fontsize=font_size)

    # subplot 1
    color = '0.45'
    ax1.set_ylabel('Log Perplexity', color=color, fontsize=font_size)
    p1 = ax1.plot(t, perplexity_list, marker='o', color=color, label = 'Log Perplexity')
    b, m = polyfit(t, perplexity_list, 1)
    # plt.plot(t, b + m * t, '--', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim([0, 0.26])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size-1)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    # subplot 2
    color = '0'
    ax2.set_ylabel('Log Topic Coherence c_v', color=color, fontsize=font_size)  # we already handled the x-label with ax1
    p2 = ax2.plot(t, coherence_list, marker='+', color=color, label = 'Log Topic Coherence c_v')
    b, m = polyfit(t, coherence_list, 1)
    # plt.plot(t, b + m * t, '--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.1)

    plt.yticks(fontsize=font_size-1) 
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    lns = p1+p2
    labs = [l.get_label() for l in lns]

    # adjust legends location
    ax1.legend(lns, labs, loc=0, fontsize=14)

    # plt.title("", fontsize=font_size)

    plt.show()
    fig.savefig(f'{task}_perplexity_c_v.png', bbox_inches='tight')
    
def test_perplexity_and_coherence(x_labels, corpus, id2word, task, iterations, passes):
    perplexity_list = []
    coherence_list = []
    font_size=20
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    for num_topics in x_labels:
        model_name = f'_t{num_topics}'
        ##Create new model with desired parameters
        lda_model = LdaMulticore(
            corpus=corpus,  # leave commented out for batch training, uncomment to train on full corpus at once
            id2word=id2word,
            iterations=iterations,
            passes=passes,
            num_topics=num_topics,
            workers=1
        )
        print("Loading model:", model_name)
        perplexity_list.append(lda_model.log_perplexity(corpus))

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        top_topics = lda_model.top_topics(corpus)
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        coherence_list.append(avg_topic_coherence)

    perplexity_list = np.asarray(perplexity_list)
    coherence_list = np.asarray(coherence_list)

    # ticks
    t = np.asarray(range(len(x_labels)))

    # main plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.xticks(t, x_labels, rotation=90, fontsize=16)
    ax1.set_xlabel('Number of Predicted Topics', fontsize=font_size)

    # subplot 1
    color = '0.45'
    ax1.set_ylabel('Log Perplexity', color=color, fontsize=font_size)
    p1 = ax1.plot(t, perplexity_list, marker='o', color=color, label = 'Log Perplexity')
    b, m = polyfit(t, perplexity_list, 1)
    # plt.plot(t, b + m * t, '--', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim([0, 0.26])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size-1)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    # subplot 2
    color = '0'
    ax2.set_ylabel('Log Topic Coherence', color=color, fontsize=font_size)  # we already handled the x-label with ax1
    p2 = ax2.plot(t, coherence_list, marker='+', color=color, label = 'Log Topic Coherence')
    b, m = polyfit(t, coherence_list, 1)
    # plt.plot(t, b + m * t, '--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.1)

    plt.yticks(fontsize=font_size-1) 
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    lns = p1+p2
    labs = [l.get_label() for l in lns]

    # adjust legends location
    ax1.legend(lns, labs, loc=0, fontsize=14)

    # plt.title("", fontsize=font_size)

    plt.show()
    fig.savefig(f'{task}_perplexity_umass.png', bbox_inches='tight')


def main(project_settings_json, num_topic):

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    print(__location__)

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
    algorithm = project_settings['ALGORITHM'] #include "topic_model" to run this code
    span_start = int(project_settings['SPAN_START']) #set to "0"
    span_end = int(project_settings['SPAN_END']) #set to "365"
    iterations = int(project_settings['ITERATIONS']) #set to "50"
    passes = int(project_settings['PASSES']) #set to "10"
    k_topics_list = project_settings['K_TOPICS_LIST']
    plot_x_labels = project_settings['PLOT_X_LABELS']
    plot_perplexity_umass = project_settings['PLOT_PERPLEXITY_UMASS'] #set to "true" to enable functionality
    plot_perplexity_cv = project_settings['PLOT_PERPLEXITY_CV'] #set to "true" to enable functionality
    save_topic_model = project_settings['SAVE_TOPIC_MODEL'] #set to "true" to enable functionality
    log_topic_metrics = project_settings['LOG_TOPIC_METRICS'] #set to "true" to enable functionality
    log_topic_weights = project_settings['LOG_TOPIC_WEIGHTS'] #set to "true" to enable functionality
    log_top_topics = project_settings['LOG_TOP_TOPICS'] #set to "true to enable functionality
    output_elements = project_settings['OUTPUT_ELEMENTS']

    print("Task: ", task)

    patientdict = import_patient_dict(note_path_prefix)
        
    k_topics_list = [num_topic]

    if 'topic_model' in algorithm:
        for k in k_topics_list:
            print(f'Main function is starting topic {k}...')
            model_name = 'topic_{}_p{}_i{}_t{}.MODEL'.format(task, passes, iterations,
                                                              int(k))  # ,str(datetime.date.today())[5:])

            top_topics, ldamodel, corpus, notes, num_topics, id2word, texts, dictionary = generate_topic_model(
                patientdict, int(k), span_start, span_end, task, iterations, passes, output_directory, save_topic_model, 
                log_topic_metrics, log_top_topics, log_topic_weights)
     
        #Plot perplexity and coherence
        #x_labels is what is sent to plot perplexity and coherence graphs
        #do not send more than 7 items, or ones that are very big, as it will take too long
        #ideally, space them out so they are equidistant apart so scale makes sense
        #x_labels = [250,500, 750,1000, 1250,1500, 1750]
        #x_labels = [50, 150, 250, 350, 450, 550, 650]



        # x_labels = []
        # for x in plot_x_labels:
        #     x_labels.append(int(x))
        # if plot_perplexity_umass == 'true':
        #     test_perplexity_and_coherence(x_labels, corpus, id2word, task)
        # if plot_perplexity_cv == 'true': 
        #     test_perplexity_and_coherence_c_v(x_labels, corpus, id2word, texts, dictionary, task)
            
    
if __name__ == '__main__':
    args = sys.argv[1:]
    fp, num = args
    main(fp, num)

  

