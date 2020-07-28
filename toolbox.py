################################################################################
# Imports
################################################################################

import torch
import json
import pickle
import re
import numpy as np
import time
import torch
import pandas as pd

import os
import os.path

import agent

import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
################################################################################
# UNIVERSAL CONSTANTS
################################################################################


class UniversalConstants:
    """
    I decided to have a class for final constants in order to organize them and not mess with the actual values and be able to use the same names in the program.
    """

    batchsize = 40
    lr_d = 0.001

    settings = ['curious', 'random', 'plasticity', 'sn']
    final_seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 12, 23, 34, 45, 56, 67, 78, 89, 90, 1, 100]
    epoch_number = 40
    nonlin = "sigmoid"

    batchsize_default = 40

    # paths
    path_models = os.path.join('outfiles', 'models')
    path_vgg = os.path.join(current_dir, 'data', 'ha_bbox_vggs/')
    paths = {
        'data': os.path.join(current_dir, 'data/'),
        'vgg': os.path.join(current_dir, 'data', 'ha_bbox_vggs/'),
        'models': os.path.join('outfiles', 'models'),
        'loss_acc': os.path.join(current_dir, 'loss_acc')
    }
    object_size = 4096
    att_hidden_size = 256  # Number of hidden nodes
    wordemb_size = 256  # Length word embedding

    printerval = 100

    device = torch.device('cuda')

################################################################################
# All the Definitions and Methods and Functions
################################################################################

##########################################
# FILE MANAGEMENT
##########################################


def mkdir(path):
    if not os.path.exists(path):
        print('creating dir {}'.format(path))
        os.mkdir(path)

##########################################
# ANALYSIS
##########################################


def load_numpy(path, agent, split, metric, lr, cc, seed):
    """
    this funciton simply returns a np array with results by creating the filename(/path) from the arguments you give it, loading the np array there, and returning it. since you stored your files differently you may need to adapt the creation of the filename.
    ----
    
    """
    
    filename = "{}_{}_{}_{}_{}_{}.npy".format(
        agent, split, metric, lr, cc, seed)
    with open((path+filename), "rb") as f:
        numpyarray = np.load(f)
    return numpyarray


def load_listener_model(path_models_folder, _word_to_ix, _device, setting: str = 'curious', epoch: int = 0, lr: float = 0.001,
                        seed: int = 123) -> agent.Listener:
    """
    Loads the listener model
    -----

    :param _word_to_ix:
    :type _word_to_ix:
    :param _device:
    :type _device:
    :param path_models_folder:
    :type path_models_folder:
    :param setting:
    :type setting:
    :param epoch:
    :type epoch:
    :param lr:
    :type lr:
    :param seed:
    :type seed:
    :return:
    :rtype:
    """
    path_wanted_model = os.path.join(path_models_folder, setting,
                                     'liModel_{}_{}_{}_ep{}.pth'.format(setting, lr, seed, epoch))

    uc = UniversalConstants()

    ntokens = len(_word_to_ix.keys())

    # initializing the model
    model = agent.Listener(uc.object_size, ntokens, uc.wordemb_size,
                           uc.att_hidden_size, nonlinearity=uc.nonlin).to(_device)
    # load state_dict to it
    checkpoint = torch.load(path_wanted_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\nModel's state_dict:")  # print the loaded state_dict
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print((('-' * 60) + '\n') * 2)

    return model


def load_speaker_model(path_models_folder, _word_to_ix, _device, setting: str = 'curious', epoch: int = 0, lr: float = 0.001,
                        seed: int = 123) -> agent.Speaker:
    """
    Loads the Speaker model
    -----

    :param _word_to_ix:
    :type _word_to_ix:
    :param _device:
    :type _device:
    :param path_models_folder:
    :type path_models_folder:
    :param setting:
    :type setting:
    :param epoch:
    :type epoch:
    :param lr:
    :type lr:
    :param seed:
    :type seed:
    :return:
    :rtype:
    """
    path_wanted_model = os.path.join(path_models_folder, setting,
                                     'spModel_{}_{}_{}_ep{}.pth'.format(setting, lr, seed, epoch))

    uc = UniversalConstants()
    ntokens = len(_word_to_ix.keys())

    # initializing the model
    model = agent.Speaker(uc.object_size, ntokens, uc.att_hidden_size, nonlinearity=uc.nonlin).to(_device)
    # load state_dict to it
    checkpoint = torch.load(path_wanted_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\nModel's state_dict:")  # print the loaded state_dict
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print((('-' * 60) + '\n') * 2)

    return model


def logit_word_ix_map(_language_input, n_obj, n_bt):

    _word_logit_map = _language_input.reshape(n_bt, n_obj)
    _word_logit_map = _word_logit_map.repeat(1, n_obj)
    _word_logit_map = _word_logit_map.reshape(_language_input.shape[0]*n_obj, 1)
    _word_logit_map = _word_logit_map.reshape(_language_input.shape[0], n_obj)

    return _word_logit_map


def load_val_batch_analysis(_dict_words_boxes, batch, _word_to_ix, _data_path, _device):
    """

    """

    # Loads the batches for the validation and test splits of the data
    language_input = []
    visual_input = []
    targets = []

    ha_vggs_indices = load_data(os.path.join(_data_path, "ha_vgg_indices.json"))
    for img in batch:
        vggs = torch.load(os.path.join(_data_path,"ha_bbox_vggs/" + img + ".pt")).to(_device)
        for obj in _dict_words_boxes[img]:
            language_input.append(get_word_ix(_word_to_ix, _dict_words_boxes[img][obj]["word"]))

            bbox_indices = []
            n = 0

            for obj_id in _dict_words_boxes[img]:
                bbox_indices.append(ha_vggs_indices[img][obj_id][0])
                if obj_id == obj:
                    targets.append(n)
                n += 1
            visual_input.append(vggs[bbox_indices, :])

    lang_batch = torch.tensor(language_input, dtype=torch.long, device=_device)
    vis_batch = torch.stack(visual_input)
    targets = torch.tensor(targets, dtype=torch.long, device=_device)
    return lang_batch, vis_batch, targets


def logit_dict_making(_frequency_dict, _logit_lists=True):
    if _logit_lists:
        raw_logit_dict = {}
        raw_logit_dict = raw_logit_dict.fromkeys(_frequency_dict.keys())  # making the initial dict by all the vocab

        for word in _frequency_dict:
            raw_logit_dict[word] = {'freq': _frequency_dict[word],
                                    'match': {'logits': [], 'avg': 0},
                                    'not_match': {'logits': [], 'avg': 0}}
    else:
        raw_logit_dict = {}
        raw_logit_dict = raw_logit_dict.fromkeys(_frequency_dict.keys())  # making the initial dict by all the vocab

        for word in _frequency_dict:
            raw_logit_dict[word] = {'freq': _frequency_dict[word]['freq'],
                                    'match': _frequency_dict[word]['match']['avg'],
                                    'not_match': _frequency_dict[word]['not_match']['avg']}
    return raw_logit_dict


def logit_catch_in_one_batch(_guesses, _word_logit_map, _targets, _logit_dict, _ix_to_word_df):
    guess_index = torch.argmax(_guesses.data, 1)
    n_guesses = _guesses.shape[0]

    for logit_block_index in range(n_guesses):

        n_ob = _guesses[logit_block_index].shape[0]

        for logit_index in range(n_ob):

            word = _ix_to_word_df.at[_word_logit_map[logit_block_index][logit_index].item(), 'word']
            logit = _guesses[logit_block_index][logit_index].item()
            # first it was if (max(_guesses[logit_block_index]) == logit) and (
            #                     _targets[logit_block_index] == guess_index[logit_block_index]):
            # and now it is:
            if _targets[logit_block_index] == logit_index:
                _logit_dict[word]['match']['logits'].append(logit)
            else:
                _logit_dict[word]['not_match']['logits'].append(logit)

    return _logit_dict


def logit_mean(listener_model, _batch_list, _frequency_dict, _dict_words_boxes,
               _word_to_ix, _ix_to_word_df, _path_data, _device):
    """
    """

    logit_dict = logit_dict_making(_frequency_dict)  # a place to hold the results and the final return is this.

    listener_model.eval()

    for batch in range(len(_batch_list)):

        language_input, visual_input, targets = load_val_batch_analysis(
            _dict_words_boxes, _batch_list[batch], _word_to_ix, _path_data, _device)

        obj_guesses = listener_model(language_input, visual_input)

        language_guess_map = logit_word_ix_map(language_input, visual_input.shape[1], len(_batch_list[batch]))
        # obj_guess_values = obj_guesses.detach()  # is it necessary here?

        logit_dict = logit_catch_in_one_batch(obj_guesses, language_guess_map, targets, logit_dict, _ix_to_word_df)

    for _word in logit_dict:
        for case in ['match', 'not_match']:
            logit_dict[_word][case]['logits'] = torch.tensor(logit_dict[_word][case]['logits'], dtype=torch.float32,
                                                             device=_device)
            if logit_dict[_word][case]['logits'].shape[0] > 0:
                logit_dict[_word][case]['avg'] = torch.mean(logit_dict[_word][case]['logits']).item()
    return logit_dict


def model_best_acc_detector(_path_loss_acc, _agent, _setting, which_set, learning):
    _the_best = {'best_seed': 0,
                 'best_ep': 0,
                 'best_acc': 0.0}

    if _agent == 'listener':
        _agent = 'li'
    elif _agent == 'speaker':
        _agent = 'sp'
    else:
        print("put either listener or speaker for agent.")

    for _seed in uc.final_seeds:
        scores = load_numpy(_path_loss_acc, _agent, which_set, 'acc', 0.001, _setting, _seed)
        if max(scores) > _the_best['best_acc']:
            _the_best['best_seed'] = _seed
            _the_best['best_ep'] = scores.argmax() + 1
            _the_best['best_acc'] = scores[_the_best['best_ep'] - 1]

    return _the_best
#########################################


def print_god_settings(god_settings):
    """
    Interpret and print what you write in the bash as the input to the main program.

    Input/Output
    ----
    :param god_settings: sys.argv
    :type god_settings: list
    :return: Batch Size, Learning Rate, Setting
    :rtype: int, float, str
    """

    print("\nSys.argv:", god_settings)
    return int(god_settings[1]), float(god_settings[2]), god_settings[3]


def load_data(data_path):
    """
    loads the dataset as python data variables

    Input/Output
    ----
    :param data_path: Path string
    :type data_path: str
    :return: loaded data
    :rtype: depends on requested
    """
    if re.findall(".+[.]json$", data_path):
        with open(data_path, "rb") as input_file:
            return json.load(input_file)
    elif re.findall(".+[.]txt$", data_path):
        with open(data_path, "rb") as fp:
            return pickle.load(fp)
    else:
        print("Data Type should be P or J")


# ------------------- #
# Data Pre-processing #
# ------------------- #
def make_vocabulary(data_dict: dict):
    """
    Returns a vocabulary and a vocabulary with frequencies

    Structure
    ----
    :param data_dict: dic_words_boxes inputs which contains: dictionary with all images and their object ids
    :type data_dict: dict
    :return: a vocabulary, a vocabulary with frequencies
    :rtype: (list, dict)
    """

    frequency = {}  # Make empty python dictionary
    for file in data_dict.keys():  # counting the word occurances
        for obj in data_dict[file].keys():
            word = data_dict[file][obj]["word"]
            if word in frequency:
                frequency[word] += 1
            else:
                frequency[word] = 1
    vocabulary = list(frequency.keys())  # a list of just vocabulary
    return vocabulary, frequency


def make_index_table(_vocabulary: list) -> dict:
    """
    For all the words in the vocabulary list assigns an index number respectively

    Structure
    ----
    :param _vocabulary:
    :type _vocabulary:
    :return: indexed dictionary of vocabulary
    :rtype: dict
    """

    _word_to_index = {"<UNK>": 0}  # 0 for the unknown words by making also a dictionary
    index = 1  # To start with one
    for word in _vocabulary:  # for all the words in the vocabulary list assigns an index number respectively
        _word_to_index[word] = index
        index += 1  # increment the index
    return _word_to_index


def imgn_per_x_objn(data_dict: dict, data_split: list) -> dict:  # fixme: It was a really really easy function. Just the description was terrible :)
    """
    Name: Image number per each X number of objects

    I am sad to name it this way short. But no other way.
    :param data_dict:
    :type data_dict:
    :param data_split:
    :type data_split:
    :return: :)
    :rtype: dict
    """
    howmany_img_per_objnumber = {}  # new dictionary
    for file in data_split:
        if len(data_dict[file]) not in howmany_img_per_objnumber:
            howmany_img_per_objnumber[len(data_dict[file])] = []
            howmany_img_per_objnumber[len(data_dict[file])].append(file)
        else:
            howmany_img_per_objnumber[len(data_dict[file])].append(file)
    return howmany_img_per_objnumber


def dict_to_batches(no_objs_split: dict, _batchsize: int) -> list:
    """
    Returns a list of batches. A batch is a
    batch-size lists of file/img ids, of images
    containing the same amount of objects.
    The batches are shuffled so that batches
    of different amounts of objects follow
    each other.

    Structure
    ----
    :param no_objs_split:
    :type no_objs_split:
    :param _batchsize: Batch Size
    :type _batchsize: int
    :return: Batch List
    :rtype: list
    """

    _batch_list = []  # making the empty list
    for num in no_objs_split.keys():
        _batch_list.extend(
            [no_objs_split[num][x:x + _batchsize] for x in range(0, len(no_objs_split[num]), _batchsize)])
    np.random.shuffle(_batch_list, )
    return _batch_list


def get_word_ix(word_to_ix, word):
    if word in word_to_ix:
        return word_to_ix[word]
    else:
        return word_to_ix["<UNK>"]

def make_ix_to_word(_word_to_ix):
    """

    """

    word_to_ix_df = pd.DataFrame.from_dict(_word_to_ix, orient='index').reset_index()
    word_to_ix_df = word_to_ix_df.rename(columns={"index": 'word', 0: 'index'})

    return word_to_ix_df
# +__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+
# AGENT FUNCTIONS
# +__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+__+


def sum_weighted(objects, attention):
    """
    input:
    objects: batch x objects in img x visual_features
    attention: batch x weights for each object x dummy dim of size 1
    returns:
    matrix: batch x weighted sum of visual features per img
    for each object, the weighting is done on the whole vector of visual
    features at once. if attention is softmaxed (as expected), therefore, sum of
    dim 1 should be 1 for each item.
    """

    assert len(attention.shape) == 3, "attention should have 3 dimensions: batch * n_objects * 1"
    assert objects.shape[0] == attention.shape[0], "object and attention dim 0 should represent batch"
    assert objects.shape[1] == attention.shape[1], "object and attention dim 1 should be nr of objects"
    assert attention.shape[2] == 1, "attention dim 2 should be 1 (dummy dimension)"

    weighted = attention * objects
    summed = torch.sum(weighted, dim=1)
    return summed

##############################################################
#                   CRUCIAL DEFINITIONS
##############################################################


def curiosity(i, o, condition="normal"):
    # Calculates the curiosity values
    # i and o are torch tensors of shape [batch, ntokens]
    subjective_novelty = abs(i - o)
    plasticity = abs(o - o ** 2)
    try:
        if condition == "plasticity":
            return torch.mean(plasticity, dim=1)
        elif condition == "sn":
            return torch.mean(subjective_novelty, dim=1)
        else:
            return torch.mean((subjective_novelty * plasticity), dim=1)
    except:
        print("problem with CURIOSITY function!")


def calc_accuracy(guesses, targets, average=True):
    """
    in: log probabilities for C classes (i.e. candidate nrs), target 'class'
    indices (from 0 up-to-and-icluding C-1) (object position in your case)
    """
    score = 0
    guess = torch.argmax(guesses.data, 1)

    for i in range(targets.data.size()[0]):
        if guess.data[i] == targets.data[i]:
            score += 1

    if average:
        return score / targets.data.size()[0], targets.data.size()[0]
    else:
        return score, targets.data.size()[0]



# r look at img
# c look at image


####
# rest of Analysis
####

def model_best_acc_detector(_path_loss_acc, _agent, _setting, which_set, _seeds):
    _the_best = {'best_seed': 0,
                 'best_ep': 0,
                 'best_acc': 0.0}

    if _agent == 'listener':
        _agent = 'li'
    elif _agent == 'speaker':
        _agent = 'sp'
    else:
        print("put either listener or speaker for agent.")

    for _seed in _seeds:
        scores = load_numpy(_path_loss_acc, _agent, which_set, 'acc', 0.001, _setting, _seed)
        if max(scores) > _the_best['best_acc']:
            _the_best['best_seed'] = _seed
            _the_best['best_ep'] = scores.argmax() + 1
            _the_best['best_acc'] = scores[_the_best['best_ep'] - 1]

    return _the_best


class UniversalData(UniversalConstants):

    # Object vgg indices (object information)
    path_ha_vggs_indices = os.path.join(UniversalConstants.paths['data'], "ha_vgg_indices.json")
    ha_vggs_indices = load_data(path_ha_vggs_indices)

    # Regular data (dictionary with all images and their object ids, corresponding words)
    path_dict_words_boxes = os.path.join(UniversalConstants.paths['data'], "dict_words_boxes.json")
    dict_words_boxes = load_data(path_dict_words_boxes)

    vocab, freq = make_vocabulary(dict_words_boxes)  # Makes a vocabulary of the entire set of objects
    indexed_vocab = make_index_table(vocab)  # Gives an index number to every word in the vocabulary

    # Train split, image ids
    path_train_data = os.path.join(UniversalConstants.paths['data'], "train_data.txt")
    train_data = load_data(path_train_data)

    # Validation split, image ids
    path_validation_data = os.path.join(UniversalConstants.paths['data'], "validation_data.txt")
    validation_data = load_data(path_validation_data)

    # Test split, image ids
    path_test_data = os.path.join(UniversalConstants.paths['data'], "test_data.txt")
    test_data = load_data(path_test_data)

    # train set
    no_objs = imgn_per_x_objn(dict_words_boxes, train_data)  # Returns a dictionary with the number of objects per image
    batches = dict_to_batches(no_objs, UniversalConstants.batchsize)  # Returns a list of batch-size batches: A batch contains images with
    # the same no. of objs

    # Validation set
    no_objs_val = imgn_per_x_objn(dict_words_boxes, validation_data)
    val_batchlist = dict_to_batches(no_objs_val, UniversalConstants.batchsize)

    # test set
    no_objs_test = imgn_per_x_objn(dict_words_boxes, test_data)
    test_batchlist = dict_to_batches(no_objs_test, UniversalConstants.batchsize)

    ntokens = len(indexed_vocab.keys())
    print("ntokens:", ntokens)
