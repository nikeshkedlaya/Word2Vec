import gensim
import os
from nltk.corpus import stopwords

# nltk stopward. You can have a list of your own stopwords if you dont want to use from nltk
stop_words = set(stopwords.words('english'))

"""
List, which can be passes to Word2Vec to train the model
It reads file from a directory given and prepares a list for each file and finally adding prepared list to global list list_to_train
"""


def prepare_list(resume_directory):
    list_to_train = []
    directory = os.fsencode(resume_directory)

    for file in os.listdir(directory):
        each_list = []
        file_name = os.fsdecode(file)
        actual_file = open(resume_directory + file_name, 'rt')
        tokens = actual_file.read().split()
        filtered_sentence = [w for w in tokens if not w in stop_words]
        for each_word in filtered_sentence:
            each_list.append(each_word.lower())
        list_to_train.append(each_list)
    return list_to_train


training_list = prepare_list("path_to_files_location_directory")

# Create a model. Passing prepared list from the files
# This model has to be saved in a file using pickel. We dont want to train the same dataset again and again
# This has to be in a seperate file and called once the new data has been added to the directory
# For ease of use i have used the model trained here itself. Its recreates model for every run
model = gensim.models.Word2Vec(training_list, min_count=1, size=300, workers=4)

try:
    print(model.similar_by_word('html'))
    print(model.similarity(w1="html", w2="javascript"))
    # You can use any of the methods from Word2Vec here
except:
    exit()
