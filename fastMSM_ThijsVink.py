import json
import copy
import numpy as np
import re
from sympy import nextprime
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from nltk import ngrams
import Levenshtein
import sympy
import warnings
import matplotlib.pyplot as plt
# All functions

# Stuff for final loop:

# create bootstrap of cleaned data where train set is an random subset of the data of size 63%
def bootstrap(data, percentage):
    train = {}
    test = {}
    keys = list(data.keys())
    np.random.shuffle(keys)  # Shuffle the keys

    n_train = int(percentage * len(keys))
    train_keys = keys[:n_train]
    test_keys = keys[n_train:]

    for key in train_keys:
        train[key] = data[key]
    for key in test_keys:
        test[key] = data[key]

    return train, test

def bootstrapper(data, amountofbootstraps, percentage):
    bootstrap_dict = {}
    for i in range(amountofbootstraps):
        train_set, test_set = bootstrap(data, percentage)
        bootstrap_dict[f"bootstrap_{i}"] = {'train': train_set, 'test': test_set}

    return bootstrap_dict
# create new dictionary with all descriptions corresponding to entries
def completedict(dict):
    newdict = {}
    i = 0
    for key in dict.keys():
        for values in dict[key]:
            newdict[i] = values
            i += 1
    return newdict


# a function that cleans the data
def title_clean(data):
    newdata = copy.deepcopy(data)
    variations_inch = ['Inch', 'inch', 'inches', 'Inches', '-inch', "\"", ' inch', "\'inch"]
    variations_hertz = ['Hz', 'hz', 'Hertz', 'hertz', '-HZ', '-hz', ' hz']
    variations_pounds = ['lbs', ' lbs', 'Lbs', ' Lbs', 'pounds', ' pounds', 'Pounds', ' Pounds']
    # if the title value contains the word inch or hertz then replcae the word with 'inch' or 'hertz' respectively and return the cleaned data

    for key, value in newdata.items():
        newtitle = value['title']
        newtitle = newtitle.lower()
        newfeatures = value['featuresMap']
        for types in variations_inch:
            newtitle = newtitle.replace(types, 'inch')
            #If inch has a space infront remove the space
            for subkey, subvalue in newfeatures.items():
                subvalue = re.compile('[^\sa-zA-z0-9.]+').sub('', subvalue)
                subvalue = subvalue.lower()
                subvalue = subvalue.replace(types, 'inch')


                newfeatures[subkey] = subvalue

        for types in variations_hertz:
            newtitle = newtitle.replace(types, 'hertz')
            for subkey, subvalue in newfeatures.items():
                subvalue = re.compile('[^\sa-zA-z0-9.]+').sub('', subvalue)
                subvalue = subvalue.lower()
                subvalue = subvalue.replace(types, 'hertz')


                newfeatures[subkey] = subvalue
        for types in variations_pounds:
            subvalue = re.compile('[^\sa-zA-z0-9.]+').sub('', subvalue)
            newtitle = newtitle.replace(types, 'pounds')
            for subkey, subvalue in newfeatures.items():
                subvalue = subvalue.lower()
                subvalue = subvalue.replace(types, 'pounds')

                newfeatures[subkey] = subvalue

        newtitle = newtitle.replace("newegg.com", "")
        newtitle = newtitle.replace("best buy", "")
        newtitle = newtitle.replace("Newegg.com", "")
        newtitle = newtitle.replace("Best buy", "")

        newtitle = re.compile('[^\sa-zA-z0-9.]+').sub('', newtitle)

        newdata[key]['title'] = newtitle
        newdata[key]['featuresMap'] = newfeatures

    return newdata

    # make all title values lowercase


# from each title subkey extract the words and return list of words
def get_model_words_title(data):
    model_words = []
    pattern = r'([a-zA-Z0-9]*(([0-9]+(\.[0-9]+)?[^0-9, ]+)|([^0-9, ]+[0-9]+(\.[0-9]+)?))[a-zA-Z0-9]*)'
    # pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)'

    regex_pattern = re.compile(pattern)
    for key, value in data.items():
        for word in regex_pattern.findall(value['title']):
            model_words.append(word[0])
    model_words = list(dict.fromkeys(model_words))
    return model_words



# get the decimal numbers from all key values except 'title' and return list of decimal numbers


# extract model words from single title entry used in TMWM
def get_model_words_title_single(data, entry):
    pattern = r'([a-zA-Z0-9]*(([0-9]+(\.[0-9]+)?[^0-9, ]+)|([^0-9, ]+[0-9]+(\.[0-9]+)?))[a-zA-Z0-9]*)'
    regex_pattern = re.compile(pattern)
    model_words = []
    for word in regex_pattern.findall(data[entry]['title']):
        model_words.append(word[0])
    model_words = list(dict.fromkeys(model_words))

    # remove duplicates
    return model_words

def get_model_words_features(data, keys):
    model_words = []
    result = []
    pattern = r'(\d+(\.\d+)?[a-zA-Z]+$|\d+(\.\d+)?)'  # this is for all numbers with letters at the end
    regex_pattern = re.compile(pattern)

    for key in keys:
        for subkey, subvalue in data['featuresMap'].items():
            if key == subkey:
                for word in regex_pattern.findall(str(subvalue)):
                        model_words.extend(word)
    model_words = list(dict.fromkeys(model_words))
    # if model_words contains letters then delete the letters
    for word in model_words:
        modified_word = re.sub(r'(\d+(\.\d+)?)\D*', r'\1', word)
        result.append(modified_word)

    return result

#############################################################################################################
# EVERYTHING FOR LSH


# characteristic matrix based on model words of title and KVPAIRS
def binarymatrix(data, model_words_title):
    # create a set of the three lists of model words

    # create a list of the set of model words
    model_words = model_words_title
    set(model_words)
    matrix = np.zeros((len(model_words), len(data)), dtype=int)

    # if title contains a modelwordtitle then the corresponding entry in the matrix is 1,
    # if a value from the keys in the features map contains a modelwordKVPAIRS or modelwordtitle,
    # then the corresponding entry in the matrix is 1
    i = 0
    for key, value in data.items():
        for word in model_words_title:
            if word in value['title']:
                matrix[model_words.index(word)][i] = 1

        i += 1

    return matrix

# find the set of used keys in all featuresMap subkeys


# Hashfunction for minhashing
def hashfunction(a, b, p, row_index):
    return (a * row_index + b) % p


# create signature matrix with hashfunction k is number of hashfunctions and n is number of columns (Tv's)

def signaturematrix(matrix, k):
    n_rows, n_columns = matrix.shape
    signature_matrix = np.full((k, n_columns), np.inf)
    # generate possible prime number larger than k
    p = nextprime(n_rows)
    np.random.seed(20)

    for row_index, row in enumerate(matrix):
        list_hashvalues = []
        # set random seed
        # Generate p hash functions

        for i in range(k):
            a = np.random.randint(1, p)  # maximum number for a an b?
            b = np.random.randint(1, p)

            hashvalue = hashfunction(a, b, p, row_index)
            list_hashvalues.append(hashvalue)
            # Iterate through columns using enumerate to get both index and value
        for column_index, column_value in enumerate(row):
            if column_value == 1:
                # Update signature matrix if the hash value is smaller
                for i in range(k):
                    if list_hashvalues[i] < signature_matrix[i][column_index]:
                        signature_matrix[i][column_index] = list_hashvalues[i]

    return signature_matrix



# def custom_hash(band):
#     return ''.join(map(str, band))

# create LSH method
def apply_lsh(signature_matrix, num_bands, band_size, lshdata):
    n_hashes, n_columns = signature_matrix.shape
    listdata = list(lshdata.keys())
    # Check that the number of bands and band size are compatible
    assert num_bands * band_size == n_hashes, "Invalid combination of bands and band size"

    # Initialize a dictionary to store buckets
    hashbuckets = defaultdict(list)

    # Hash each column into buckets based on bands
    for band_index in range(num_bands):
        start_row = band_index * band_size
        end_row = (band_index + 1) * band_size

        # Create a hash for each column in the band
        hashes = [hash(tuple(signature_matrix[start_row:end_row, column])) for column in range(n_columns)]

        # Add each column to the corresponding bucket
        for column_index, hash_value in enumerate(hashes):
            hashbuckets[hash_value].append(listdata[column_index])

    # Identify candidate pairs based on overlapping buckets
    candidate_pairs = set()
    for bucket in hashbuckets.values():
        if len(bucket) > 1:
            # Add pairs of columns within the same bucket
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    pair = (min(bucket[i], bucket[j]), max(bucket[i], bucket[j]))
                    candidate_pairs.add(pair)
    return list(candidate_pairs), hashbuckets


# combine remove_same_shop and remove_different_brand
def remove_same_shop_and_different_brand(candidate_pairs, newdata):
    new_candidate_pairs = []

    for pair in candidate_pairs:
        shop1 = newdata[pair[0]].get('shop')
        brand1 = newdata[pair[0]]['featuresMap'].get('brand')

        shop2 = newdata[pair[1]].get('shop')
        brand2 = newdata[pair[1]]['featuresMap'].get('brand')

        if shop1 != shop2:
            if brand1 == brand2 or brand1 is None or brand2 is None:
                new_candidate_pairs.append(pair)

    return new_candidate_pairs


#############################################################################################################
# EVERYTHIN for SCORING FUNCTIONS
###
# find all true pairs from the data, remove recipricols and return list of true pairs
def find_true_pairs(data):
    true_pairs = []
    keys = list(data.keys())
    n = len(keys)

    for i in range(n):
        for j in range(i + 1, n):
            key1, key2 = keys[i], keys[j]

            if data[key1]['modelID'] == data[key2]['modelID']:
                true_pairs.append((min(key1, key2), max(key1, key2)))
    true_pairs.sort()


    return true_pairs


# find precision recall and F1 score from the candidate pairs where a true pair is with same modelID and false pair
# is with different modelID
def scores(candidate_pairs, true_pairs, lsh):
    candidate_pairs = set(candidate_pairs)
    true_pairs = set(true_pairs)
    true_found_positives = candidate_pairs.intersection(true_pairs)
    false_positives = candidate_pairs.difference(true_pairs)
    false_negatives = true_pairs.difference(candidate_pairs)

    if lsh == True:
        # Avoid division by zero by checking if the denominators are zero
        if len(true_pairs) == 0 or len(candidate_pairs) == 0:
            return 0, 0, 0  # or handle it differently based on your requirements

        PC = len(true_found_positives) / len(true_pairs)
        PQ = len(true_found_positives) / len(candidate_pairs)
        F1_star = 2 * (PC * PQ) / (PC + PQ) if (PC + PQ) != 0 else 0
        return PC, PQ, F1_star
    else:
        # Avoid division by zero by checking if the denominators are zero
        if (len(true_found_positives) + len(false_positives)) == 0 or (
                len(true_found_positives) + len(false_negatives)) == 0:
            return 0, 0, 0  # or handle it differently based on your requirements

        precision = len(true_found_positives) / (len(true_found_positives) + len(false_positives))
        recall = len(true_found_positives) / (len(true_found_positives) + len(false_negatives))
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return precision, recall, F1

# for a range b's where b*r = k, apply lsh and find scores for each b and r
def final_scores(signature_matrix, alldistances, true_pairs, data, threshold, b,r,lsh,clustering):
    n_hashes, n_columns = signature_matrix.shape
    PC_lsh = 0
    PQ_lsh = 0
    F1_star_lsh = 0
    F1 =0
    completedistance = alldistances

    candidate_pairs_lsh, buckets = apply_lsh(signature_matrix, b, r, data)


    if lsh:
        PC_lsh, PQ_lsh, F1_star_lsh = scores(candidate_pairs_lsh, true_pairs, True)

    #print scores
    if clustering:
        candidate_pairs_msm = remove_same_shop_and_different_brand(candidate_pairs_lsh, data)
        # find msm candidates
        distmatrix = filter_distance_matrix(completedistance,candidate_pairs_msm)  # alpha gamma mu update when actual run happens

        clusters = MSMclustering(distmatrix, threshold)
        finalpairs = find_duplicates(clusters)

        precision, recall, F1 = scores(finalpairs, true_pairs, False)


    fraction_comp_lsh = len(candidate_pairs_lsh) / (n_columns * (n_columns - 1) / 2)




    return PC_lsh, PQ_lsh, F1_star_lsh, F1, fraction_comp_lsh



#############################################################################################################
# functions needed for TMWM
#############################################################################################################


def TMWM(entry1, entry2, data, alpha, delta=0.7, beta=0.3):
    # entry1 and entry2 are the keys of the two entries
    # alpha is the threshold for cosine similarity
    # delta is the weight of cosine similarity == fixed (relatively high since when we have same numbers and approx same non numeric characters higher chance of duplicate)
    # beta is the weight of levenshtein similarity == fixed (relatively low since if exact same words in cosine was not true we want to weigh the similarity of the model words higher)

    # from given data entry extract the words from the 'title' string and return list of words
    entry1_words = data[entry1]['title'].split()
    entry2_words = data[entry2]['title'].split()
    # find the cosine similarity of the two lists of words
    intersection = len(set(entry1_words).intersection(entry2_words))
    cosine = intersection / (np.sqrt(len(entry1_words)) * np.sqrt(len(entry2_words)))  # checkk this sqrt thing?
    if cosine > alpha:
        return 1

    #

    else:
        # Extract model words
        mw_entry1 = get_model_words_title_single(data, entry1)
        mw_entry2 = get_model_words_title_single(data, entry2)

        # Check if there is at least one pair where non-numeric characters are approximately the same
        # AND numeric characters are not the same
        for word1 in mw_entry1:
            for word2 in mw_entry2:
                non_numeric_sim = approx_similarity_non_numeric(word1, word2)
                numeric_sim = approx_similarity_numeric(word1, word2)

                if non_numeric_sim > 0.7 and numeric_sim < 1:
                    return -1  # the product names do not represent the same product

        # Compute initial product name similarity
        final_name_sim = beta * cosine + (1 - beta) * average_levenshtein_similarity(entry1_words, entry2_words)

        # Check if there is at least one pair where non-numeric characters are approximately the same
        # AND the numeric characters are the same
        for word1 in mw_entry1:
            for word2 in mw_entry2:
                non_numeric_sim = approx_similarity_non_numeric(word1, word2)
                numeric_sim = approx_similarity_numeric(word1, word2)

                if non_numeric_sim > 0.7 and numeric_sim == 1:
                    # Compute Levenshtein similarity for model words
                    model_word_sim_val = avgLvSimMW(mw_entry1, mw_entry2)
                    final_name_sim = delta * model_word_sim_val + (1 - delta) * final_name_sim
                    return final_name_sim  # the product names represent the same product

        return final_name_sim  # the product names represent the same product if true, false otherwise


def avgLvSimMW(set1, set2):
    total_similarity = 0
    total_pairs = 0

    for word1 in set1:
        for word2 in set2:
            non_numeric_sim = approx_similarity_non_numeric(word1, word2)
            numeric_sim = approx_similarity_numeric(word1, word2)

            if non_numeric_sim > 0.7 and numeric_sim == 1:
                distance = Levenshtein.distance(word1, word2)
                similarity = 1 - distance / max(len(word1), len(word2))
                total_similarity += similarity
                total_pairs += 1

    average_similarity = total_similarity / total_pairs if total_pairs > 0 else 0
    return average_similarity


def average_levenshtein_similarity(set1, set2):
    total_similarity = 0
    total_pairs = 0

    for word1 in set1:
        for word2 in set2:
            distance = Levenshtein.distance(word1, word2)
            similarity = 1 - distance / max(len(word1), len(word2))
            total_similarity += similarity
            total_pairs += 1

    average_similarity = total_similarity / total_pairs if total_pairs > 0 else 0
    return average_similarity


def extract_non_numeric(word):
    return re.sub(r'\d', '', word)


def extract_numeric(word):
    return re.sub(r'\D', '', word)  # Keep only digits


def normalized_levenshtein_distance(word1, word2):
    len_max = max(len(word1), len(word2))
    if len_max == 0:
        return 0  # Handle the case when both strings are empty
    distance = Levenshtein.distance(word1, word2)
    return 1 - distance / len_max


def approx_similarity_non_numeric(word1, word2):
    non_numeric1 = extract_non_numeric(word1)
    non_numeric2 = extract_non_numeric(word2)
    return normalized_levenshtein_distance(non_numeric1, non_numeric2)


def approx_similarity_numeric(word1, word2):
    numeric1 = extract_numeric(word1)
    numeric2 = extract_numeric(word2)
    return normalized_levenshtein_distance(numeric1, numeric2)


#############################################################################################################
# MSM CODE
#############################################################################################################


def insert_dummy_characters(string, q):
    dummy_chars = '*' * (q - 1)
    return dummy_chars + string + dummy_chars


def qgram_similarity(string1, string2, q):
    # remove spaces from the strings
    string1 = string1.replace(" ", "")
    string2 = string2.replace(" ", "")
    # insert dummy characters
    string1 = insert_dummy_characters(string1, q)
    string2 = insert_dummy_characters(string2, q)

    # create qgrams for each string
    qgrams1 = set(ngrams(string1, q))
    qgrams2 = set(ngrams(string2, q))

    # find the intersection of the two lists
    intersection = set(qgrams1).intersection(qgrams2)
    # find the union of the two lists
    union = set(qgrams1).union(qgrams2)
    # calculate the similarity
    similarity = len(intersection) / len(union)
    return similarity


def minfeatures(entry1, entry2, data):
    # entry1 and entry2 are the keys of the two entries
    # data is the data dictionary
    # find the number of features in the featuresMap of the two entries
    entry1_features = data[entry1]['featuresMap']
    entry2_features = data[entry2]['featuresMap']
    # find the minimum number of features
    min_features = min(len(entry1_features), len(entry2_features))
    return min_features


def pairflatten(candidate_pairs):
    candidate_products = [item for sublist in candidate_pairs for item in sublist]
    # check for duplicates
    candidate_products = list(dict.fromkeys(candidate_products))
    # order the products from low to high
    candidate_products.sort()
    return candidate_products



# create disimilarity matrix with the three steps for ALL DATA CHECKED
#THEN USE THE ENTRIES ONLY RELEVANT FOR THE CANDIDATE PAIRS FOR FURHTER CALCULATIONS and faster computaional speed

def optimized_totaldistancematrix(data, alpha, mu):
    n = len(data)
    distancematrix = np.full((n, n), np.inf)

    # Precompute model words and other frequently used values
    precomputed_values = {}
    for i in range(n):
        title = data[i]['title']
        featuresMap = data[i]['featuresMap']
        precomputed_values[i] = {
            'title_words': title.split(),
            'featuresMap_keys': list(featuresMap.keys()),
            'model_words': get_model_words_features(data[i], list(featuresMap.keys())),

        }

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle
            data_i = precomputed_values[i]
            data_j = precomputed_values[j]
            titlesim = TMWM(i, j, data, alpha)

            # Calculate qgram similarity and weights
            sim = 0
            avgSim = 0
            m = 0


            for key1 in data_i['featuresMap_keys']:
                # if the key is in the featuresMap of the second product, calculate the similarity
                if key1 in data_j['featuresMap_keys']:
                    key2 = key1  # matching key in second product
                    value1 = data[i]['featuresMap'][key1]
                    value2 = data[j]['featuresMap'][key2]

                    # calculate the qgram similarity of the keys

                    valuesim = qgram_similarity(value1, value2, 3)
                    sim += valuesim
                    m = m + 1

            if m > 0:
                avgSim = sim / m

            # convert the set to a list


            model_wordskey1_set = set(data_i['model_words'])
            model_wordskey2_set = set(data_j['model_words'])
            matching_words = model_wordskey1_set.intersection(model_wordskey2_set)
            total_words = model_wordskey1_set.union(model_wordskey2_set)
            #makesure to be able to continue if there is /0
            if len(total_words) == 0:
                mwPerc = 0
            else:
                mwPerc = len(matching_words) / len(total_words)


            if titlesim == -1:
                theta1 = m / minfeatures(i, j, data)
                theta2 = 1 - theta1
                hsim = theta1 * avgSim + theta2 * mwPerc
            else:
                theta1 = (1 - mu) * m / minfeatures(i, j, data)
                theta2 = 1 - theta1 - mu
                hsim = theta1 * avgSim + theta2 * mwPerc + mu * titlesim

            distance = 1 - hsim
            distancematrix[i][j] = distance
            distancematrix[j][i] = distance  # Mirror the value

        print(f"Row {i} processed")  # Optional: for progress tracking

    return distancematrix






def filter_distance_matrix(distance_matrix, pairs_to_keep):

    # Set all values in the new matrix to a large number
    new_matrix = np.full_like(distance_matrix, fill_value=100)

    # Set values for the specified pairs
    for pair in pairs_to_keep:
        row_index, col_index = pair
        if 0 <= row_index < distance_matrix.shape[0] and 0 <= col_index < distance_matrix.shape[1]:
            new_matrix[row_index, col_index] = distance_matrix[row_index, col_index]

    return new_matrix

#Clusteing functions
#apply agglomerative clustering on the distance matrix and return clusters
def MSMclustering(filter_distance_matrix, threshold):
    # replace the inf values with the maximum value in the matrix
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                         distance_threshold=threshold).fit_predict(filter_distance_matrix)


    return clustering


def find_duplicates(clustering_result):
    # Step 1: Identify unique cluster labels
    unique_clusters = set(clustering_result)

    # Step 2: For each unique cluster label, find the data points in that cluster
    cluster_data = defaultdict(list)
    for i, label in enumerate(clustering_result):
        cluster_data[label].append(i)

    # Step 3: Create pairs of items within each cluster
    cluster_pairs = []
    for cluster_label, data_points in cluster_data.items():
        if len(data_points) >= 2:
            # Add pairs of items within the cluster
            cluster_pairs.extend([(data_points[i], data_points[j]) for i in range(len(data_points)) for j in range(i + 1, len(data_points))])

    return cluster_pairs

#############################################################################################################
# visualization functions
# plot the scores against the fraction of comparisons #entry 1 is PC, entry 2 is PQ, entry 3 is F1_star

def average_bootstrapped_scores(bootstrapped_scores):
    # Initialize the result dictionary
    averaged_scores = {"train": {}, "test": {}}

    # Iterate over each bootstrap
    for bootstrap_scores in bootstrapped_scores:
        for dataset_type in ["train", "test"]:
            # Get the current dataset (train or test)
            current_dataset = bootstrap_scores[dataset_type]

            for key, metrics in current_dataset.items():
                # If the key doesn't exist in the averaged_scores dictionary, initialize it
                if key not in averaged_scores[dataset_type]:
                    averaged_scores[dataset_type][key] = {}

                # Iterate over metrics (PC, PQ, F1_star, F1_MSM, fraction_comp)
                for metric, value in metrics.items():
                    # If the metric doesn't exist in the averaged_scores dictionary, initialize it
                    if metric not in averaged_scores[dataset_type][key]:
                        averaged_scores[dataset_type][key][metric] = 0.0

                    # Add the current value to the running total
                    averaged_scores[dataset_type][key][metric] += value

    # Divide each total by the number of bootstraps to get the mean
    num_bootstraps = len(bootstrapped_scores)
    for dataset_type in ["train", "test"]:
        for key, metrics in averaged_scores[dataset_type].items():
            for metric in metrics:
                averaged_scores[dataset_type][key][metric] /= num_bootstraps

    return averaged_scores

#plot average scores
def plot_average_scores(averaged_scores, entry, set):
    import matplotlib.pyplot as plt
    x = []
    y = []
    for key, value in averaged_scores[set].items():
        x.append(value['fraction_comp'])
        y.append(value[entry])
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.xlabel('Fraction of Comparisons')
    plt.ylabel(entry)
    plt.title(f'{entry} vs Fraction of Comparisons')
    plt.grid(True)
    plt.legend()
    plt.show()
def bootstrapper2( bootstrappeddata, clustering_thresholds, n_hashes, alldistances):


    bootstrapped_scores = []

    for key,value in bootstrappeddata.items():
        print(f"{key} of {amountofbootstraps}")

        # Generate train and test sets
        train = value['train']
        test = value['test']


        # Create binary matrices and signature matrices
        trainbinary = binarymatrix(train, model_words_title)
        testbinary = binarymatrix(test, model_words_title)
        train_sigmatrix = signaturematrix(trainbinary, n_hashes)
        test_sigmatrix = signaturematrix(testbinary, n_hashes)

        # Find true pairs for train and test sets
        truepairs_train = find_true_pairs(train)
        truepairs_test = find_true_pairs(test)

        # Initialize results dictionary for the current bootstrap
        current_results = {"train": {}, "test": {}}

        # Loop over each B and R combination
        for b in range(1, n_hashes + 1):
            if n_hashes % b == 0:
                r = n_hashes // b

                # Calculate LSH scores on the training set
                Lsh_scores = final_scores(train_sigmatrix, alldistances, truepairs_train, train, 0.6, b, r, True, False)
                current_results["train"][f"b_{b}_r_{r}"] = {
                    "PC": Lsh_scores[0],
                    "PQ": Lsh_scores[1],
                    "F1_star": Lsh_scores[2],
                    "fraction_comp": Lsh_scores[4],
                }

                # Find the best MSM threshold on the training set
                best_threshold = None
                best_f1_msm = -1  # Initialize to a value that will be replaced by the first score

                for threshold in clustering_thresholds:
                    trainscores = final_scores(train_sigmatrix, alldistances, truepairs_train, train, threshold, b, r,
                                               False, True)
                    f1_msm = trainscores[3]
                    if f1_msm > best_f1_msm:
                        best_f1_msm = f1_msm
                        best_threshold = threshold

                # Store best MSM threshold and scores on the training set
                current_results["train"][f"b_{b}_r_{r}"]["best_threshold"] = best_threshold
                current_results["train"][f"b_{b}_r_{r}"]["best_F1_MSM"] = best_f1_msm

                # Calculate scores on the test set using the best threshold found on the training set
                testscores = final_scores(test_sigmatrix, alldistances, truepairs_test, test, best_threshold, b, r,
                                           True, True)
                current_results["test"][f"b_{b}_r_{r}"] = {
                    "PC": testscores[0],
                    "PQ": testscores[1],
                    "F1_star": testscores[2],
                    "F1_MSM": testscores[3],
                    "fraction_comp": testscores[4],
                }
                #progress tracking
                print(f"b = {b}, r = {r} processed")

        bootstrapped_scores.append(current_results)

    return bootstrapped_scores




# Read data from Json file
ogdata = json.load(open('Data/TVs-all-merged.json'))

# perpare the data
data = copy.deepcopy(ogdata)
data = completedict(data)
data = title_clean(data)
alldistances = optimized_totaldistancematrix(data,  0.6, 0.7)
model_words_title = get_model_words_title(data)


clustering_thresholds = [0.1,0.15,0.2,0.25,0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9,0.95]

amountofbootstraps = 5
bootstrappeddata = bootstrapper(data, amountofbootstraps, percentage=0.63)
n_hashes = 800
msm_scores = bootstrapper2(bootstrappeddata, clustering_thresholds, n_hashes, alldistances)

