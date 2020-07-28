import sys
import random

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# general imports
import json
import pickle
import time
import numpy as np

# custom imports
import agent
from toolbox import *

# todo: check this path managing
# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)
###############################################################################
# SETTING CONSTANTS & INITIALIZATION
###############################################################################
# todo: start main
###########
# Constants
###########
uc = UniversalConstants()

seed = int(sys.argv[4])

random.seed(a=seed)

# setting torch seeds
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

np.random.seed(seed)

# todo: mkdir all the needed paths
mkdir(uc.path_models)

path_data = 'data/'

# Print after this many batches:
printerval = uc.printerval

##################################
# Interpret command line arguments
##################################
# batchsize, lr, setting = print_god_settings(sys.argv) todo: four next lines can be done just by this line as well
print("\nSys.argv:", sys.argv)
batchsize = int(sys.argv[1])
lr = float(sys.argv[2])
setting = sys.argv[3]

##################
# Reading the Data
##################

# Object vgg indices (object information)
path_ha_vggs_indices = os.path.join(path_data, "ha_vgg_indices.json")
ha_vggs_indices = load_data(path_ha_vggs_indices)

# Regular data (dictionary with all images and their object ids, corresponding words)
path_dict_words_boxes = os.path.join(path_data, "dict_words_boxes.json")
dict_words_boxes = load_data(path_dict_words_boxes)

# Train split, image ids
path_train_data = os.path.join(path_data, "train_data.txt")
train_data = load_data(path_train_data)

# Validation split, image ids
path_validation_data = os.path.join(path_data, "validation_data.txt")
validation_data = load_data(path_validation_data)

# Test split, image ids
path_test_data = os.path.join(path_data, "test_data.txt")
test_data = load_data(path_test_data)
###############################################################################
# PREPROCESSING
###############################################################################

vocab, freq = make_vocabulary(dict_words_boxes)  # Makes a vocabulary of the entire set of objects
word_to_ix = make_index_table(vocab)  # Gives an index number to every word in the vocabulary todo: change name to
# indexed_vocabulary

path_word_to_ix = os.path.join('./outfiles', 'models', 'word_to_ix/')
mkdir(path_word_to_ix)
with open(os.path.join(path_word_to_ix, 'word_to_ix_{}_{}_{}.json'.format(setting, seed, str(lr))), 'w') as wtx:
    json.dump(word_to_ix, wtx)

# train set
no_objs = imgn_per_x_objn(dict_words_boxes, train_data)  # Returns a dictionary with the number of objects per image
batches = dict_to_batches(no_objs, batchsize)  # Returns a list of batch-size batches: A batch contains images with
# the same no. of objs

# Validation set
no_objs_val = imgn_per_x_objn(dict_words_boxes, validation_data)
val_batchlist = dict_to_batches(no_objs_val, batchsize)

# test set
no_objs_test = imgn_per_x_objn(dict_words_boxes, test_data)
test_batchlist = dict_to_batches(no_objs_test, batchsize)

ntokens = len(word_to_ix.keys())
print("ntokens:", ntokens)
###############################################################################
# SPECIFY MODEL
###############################################################################
# these are the sizes Anna Rohrbach uses. she uses a batch size of 40.
# n_objects = 100
object_size = uc.object_size  # Length vgg vector?
att_hidden_size = uc.att_hidden_size  # Number of hidden nodes
wordemb_size = uc.wordemb_size  # Length word embedding
nonlin = uc.nonlin
print("hidden layer size:", att_hidden_size)

epochs = uc.epoch_number

device = torch.device('cuda')  # Device = GPU

# Makes the listener part of the model:
listener = agent.Listener(object_size, ntokens, wordemb_size,
                          att_hidden_size, nonlinearity=nonlin).to(device)
mkdir(uc.path_models + '/' + setting)  # todo: moving it to top, after removing function definitions
# todo: manage the path
torch.save({
            'epoch': 0,
            'setting': setting,
            'seed': seed,
            'model_state_dict': listener.state_dict()
        }, (os.path.join('./outfiles', 'models', setting, 'liModel_{}_{}_{}_ep0.pth'
                         .format(setting, str(lr), seed))))
# Makes the speaker part of the model:
speaker = agent.Speaker(object_size, ntokens, att_hidden_size, nonlinearity=nonlin).to(device)
torch.save({
            'epoch': 0,
            'setting': setting,
            'seed': seed,
            'model_state_dict': speaker.state_dict()
        }, (os.path.join('./outfiles', 'models', setting, 'spModel_{}_{}_{}_ep0.pth'
                         .format(setting, str(lr), seed))))  # todo: manage the path

# Loss function: binary cross entropy
criterion = nn.CrossEntropyLoss(size_average=True)

###############################################################################
# TRAIN LOOP
###############################################################################
print("parameters of listener agent:")
for param in listener.parameters():
    print(type(param.data), param.size())
listener_optimizer = optim.Adam(listener.parameters(), lr=lr)

print("parameters of speaker agent:")
for param in speaker.parameters():
    print(type(param.data), param.size())
speaker_optimizer = optim.Adam(speaker.parameters(), lr=lr)

# Creating numpy arrays to store loss and accuracy
# for train, validation, and test splits
listener_train_loss = np.empty(epochs)
listener_train_acc = np.empty(epochs)
speaker_train_loss = np.empty(epochs)
speaker_train_acc = np.empty(epochs)
listener_val_loss = np.empty(epochs)
listener_val_acc = np.empty(epochs)
speaker_val_loss = np.empty(epochs)
speaker_val_acc = np.empty(epochs)
# listener_test_loss = np.empty(epochs)
# listener_test_acc = np.empty(epochs)
# speaker_test_loss = np.empty(epochs)
# speaker_test_acc = np.empty(epochs)

# At any point you can hit Ctrl + C to break out of training early.

# ONE


# todo: end main
###############################################################################
# evaluation function
###############################################################################

def evaluate(epoch, split='val'):
    listener.eval()
    speaker.eval()
    if split == 'val':
        batchlist = val_batchlist
    elif split == 'test':
        batchlist = test_batchlist
    n_batches = len(batchlist)
    start_time = time.time()
    li_eval_loss = np.empty(n_batches)
    li_eval_acc = np.empty(n_batches)
    sp_eval_loss = np.empty(n_batches)
    sp_eval_acc = np.empty(n_batches)
    batch_size = np.empty(n_batches)

    batch = 0

    while batch < n_batches:
        language_input, visual_input, targets = load_val_batch(dict_words_boxes,
                                                               batchlist[batch],
                                                               word_to_ix,
                                                               device)

        obj_guesses = listener(language_input, visual_input)
        obj_guess_values = obj_guesses.detach()

        word_guesses = speaker(visual_input, obj_guess_values)
        li_loss = criterion(obj_guesses, targets)
        li_eval_acc[batch], batch_size[batch] = calc_accuracy(obj_guesses, targets)
        li_eval_loss[batch] = li_loss.item() * batch_size[batch]
        li_eval_acc[batch] *= batch_size[batch]  # avg weighted for differing batchsizes

        sp_loss = criterion(word_guesses, language_input)
        sp_eval_loss[batch] = sp_loss.item() * batch_size[batch]
        sp_eval_acc[batch], _ = calc_accuracy(word_guesses, language_input)
        sp_eval_acc[batch] *= batch_size[batch]  # avg weighted for differing batchsizes

        batch += 1
        if batch % printerval == 0:
            print(
                '| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | l.L {:6.4f} | l.A {:5.4f} | s.L {:6.4f} | s.A {:5.4f} |'.format(
                    epoch, batch, n_batches, (time.time() - start_time),
                    np.sum(li_eval_loss[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(li_eval_acc[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(sp_eval_loss[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(sp_eval_acc[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch])))

    avg_li_eval_loss = np.sum(li_eval_loss) / np.sum(batch_size)
    avg_li_eval_acc = np.sum(li_eval_acc) / np.sum(batch_size)
    avg_sp_eval_loss = np.sum(sp_eval_loss) / np.sum(batch_size)
    avg_sp_eval_acc = np.sum(sp_eval_acc) / np.sum(batch_size)

    if split == 'val':
        print('-' * 89)
        print("overall performance on validation set:")
        print('| L.loss {:8.4f} | L.acc. {:8.4f} |'.format(
            avg_li_eval_loss,
            avg_li_eval_acc))
        print('| S.loss {:8.4f} | S.acc. {:8.4f} |'.format(
            avg_sp_eval_loss,
            avg_sp_eval_acc))
        print('-' * 89)
    elif split == 'test':
        print('-' * 89)
        print("overall performance on test set:")
        print('| L.loss {:8.4f} | L.acc. {:8.4f} |'.format(
            avg_li_eval_loss,
            avg_li_eval_acc))
        print('| S.loss {:8.4f} | S.acc. {:8.4f} |'.format(
            avg_sp_eval_loss,
            avg_sp_eval_acc))
        print('-' * 89)
    return avg_li_eval_loss, avg_li_eval_acc, avg_sp_eval_loss, avg_sp_eval_acc


def load_val_batch(_dict_words_boxes, batch, _word_to_ix, _device):
    # Loads the batches for the validation and test splits of the data
    language_input = []
    visual_input = []
    targets = []

    for img in batch:
        vggs = torch.load("./data/ha_bbox_vggs/" + img + ".pt").to(_device)
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


def load_img(_dict_words_boxes, _ha_vggs_indices, _word_to_ix, img, _device, path_vgg):
    vggs = torch.load(path_vgg + img + ".pt").to(_device)  # Edit path
    # dict met obj ids als keys en een dictionary met words : '', bboxes :
    # n = 0
    bbox_indices = []
    words = []
    for obj in _dict_words_boxes[img]:  # For every object in this image
        words.append(get_word_ix(_word_to_ix, _dict_words_boxes[img][obj]["word"]))
        bbox_indices.append(_ha_vggs_indices[img][obj][0])
    visual_input = vggs[bbox_indices, :]
    language_input = torch.tensor(words, dtype=torch.long, device=_device)
    return language_input, visual_input


def curious_look_at_img(_dict_words_boxes, _ha_vggs_indices, img, _setting, word_to_ix):
    language_input, scene = load_img(_dict_words_boxes, _ha_vggs_indices, word_to_ix, img, device, uc.path_vgg)
    # repeat scene n_objects times as input to listener
    visual_input = scene.expand(scene.size()[0], scene.size()[0], scene.size()[1])
    curiosity_targets = torch.eye(visual_input.size()[0], dtype=torch.float, device=device)
    # targets is simply 0, 1, ...., n because they are in order of appearance
    targets = torch.tensor([i for i in range(len(language_input))], dtype=torch.long, device=device)
    # word guesses by child - use as attention over word embeddings
    word_guesses = speaker(visual_input, curiosity_targets, apply_softmax=False)
    # only keep most likely words
    words = torch.argmax(word_guesses, dim=1)
    # give these as input to listener
    object_guesses = listener(words, visual_input)
    curiosity_values = curiosity(curiosity_targets, object_guesses, _setting)
    max_curious = torch.argmax(curiosity_values)
    return language_input[max_curious], scene, targets[max_curious]


def random_look_at_img(_dict_words_boxes, _ha_vggs_indices, img, _word_to_ix):
    language_input, scene = load_img(_dict_words_boxes, _ha_vggs_indices, _word_to_ix, img, device, uc.path_vgg)
    # targets is simply 0, 1, ...., n because they are in order of appearance
    targets = torch.tensor([i for i in range(len(language_input))], dtype=torch.long, device=device)
    i = np.random.randint(len(targets))
    return language_input[i], scene, targets[i]


def load_select_obj(_dict_words_boxes, _ha_vggs_indices, img, _setting, _word_to_ix):
    if _setting == "random":
        return random_look_at_img(_dict_words_boxes, _ha_vggs_indices, img, _word_to_ix)
    elif (_setting == "curious") | (_setting == "plasticity") | (_setting == "sn"):
        return curious_look_at_img(_dict_words_boxes, _ha_vggs_indices, img, _setting, _word_to_ix)
    else:
        print('setting is not correct. It should be random, curious, plasticity, or sn.')

###############################################################################
# Training function
###############################################################################

def train():
    listener.train()
    speaker.train()
    start_time = time.time()
    n_batches = len(batches)
    li_train_loss = np.empty(n_batches)
    li_train_accuracy = np.empty(n_batches)
    sp_train_loss = np.empty(n_batches)
    sp_train_accuracy = np.empty(n_batches)
    batch_size = np.empty(n_batches)

    batch = 0

    # batches shuffled during training
    while batch < n_batches:
        language_batch = []  # All word indices in the batch?
        visual_batch = []  # All vgg vectors in the batch?
        target_batch = []  # All target word indices in the batch?

        for img in batches[batch]:
            language_input, visual_input, target = load_select_obj(dict_words_boxes, ha_vggs_indices, img,
                                                                   setting, word_to_ix)
            language_batch.append(language_input)
            visual_batch.append(visual_input)
            target_batch.append(target)
        language_input = torch.stack(language_batch)
        visual_input = torch.stack(visual_batch)
        targets = torch.stack(target_batch)

        speaker_optimizer.zero_grad()
        listener_optimizer.zero_grad()

        obj_guesses = listener(language_input, visual_input)

        # Saves the batch length for weighted mean accuracy:
        batch_size[batch] = len(batches[batch])

        loss = criterion(obj_guesses, targets)
        loss.backward()  # backward pass
        listener_optimizer.step()  # adapting the weights

        # Loss/accuracy times batch size for weighted average over epoch:
        li_train_loss[batch] = loss.item() * batch_size[batch]
        li_train_accuracy[batch], _ = calc_accuracy(obj_guesses, targets, average=False)

        obj_guess_values = obj_guesses.detach()

        word_guesses = speaker(visual_input, obj_guess_values)

        speaker_loss = criterion(word_guesses, language_input)
        speaker_loss.backward()
        speaker_optimizer.step()

        # Loss/accuracy times batch size for weighted average over epoch:
        sp_train_loss[batch] = speaker_loss.item() * batch_size[batch]
        sp_train_accuracy[batch], _ = calc_accuracy(word_guesses, language_input, average=False)

        batch += 1
        if batch % printerval == 0:
            print(
                '| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | l.L {:6.4f} | l.A {:5.4f} | s.L {:6.4f} | s.A {:5.4f} |'.format(
                    epoch, batch, n_batches, (time.time() - start_time),
                    np.sum(li_train_loss[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(li_train_accuracy[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(sp_train_loss[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch]),
                    np.sum(sp_train_accuracy[batch - printerval:batch]) / np.sum(batch_size[batch - printerval:batch])))

    avg_li_train_loss = np.sum(li_train_loss) / np.sum(batch_size)
    avg_li_train_acc = np.sum(li_train_accuracy) / np.sum(batch_size)
    avg_sp_train_loss = np.sum(sp_train_loss) / np.sum(batch_size)
    avg_sp_train_acc = np.sum(sp_train_accuracy) / np.sum(batch_size)

    print('-' * 89)
    print("overall performance on training set:")
    print('| L.loss {:8.4f} | L.acc. {:8.4f} |'.format(
        avg_li_train_loss,
        avg_li_train_acc))
    print('| S.loss {:8.4f} | S.acc. {:8.4f} |'.format(
        avg_sp_train_loss,
        avg_sp_train_acc))
    print('-' * 89)
    return avg_li_train_loss, avg_li_train_acc, avg_sp_train_loss, avg_sp_train_acc


# Train
li_val_loss, li_val_acc, sp_val_loss, sp_val_acc = evaluate(0)  # first run evaluate to get random baseline
# todo: move to position ONE after moving evaluate to toolbox

try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()  # todo: should we do anything with this time?
        li_train_loss, li_train_acc, sp_train_loss, sp_train_acc = train()
        listener_train_loss[epoch - 1], listener_train_acc[epoch - 1] = li_train_loss, li_train_acc
        speaker_train_loss[epoch - 1], speaker_train_acc[epoch - 1] = sp_train_loss, sp_train_acc

        torch.save({
            'epoch': epoch,
            'setting': setting,
            'seed': seed,
            'model_state_dict': listener.state_dict()
        }, (os.path.join('./outfiles', 'models', setting, 'liModel_{}_{}_{}_ep{}.pth'
                         .format(setting, str(lr), seed, epoch))))

        torch.save({
            'epoch': epoch,
            'setting': setting,
            'seed': seed,
            'model_state_dict': speaker.state_dict()
        }, (os.path.join('./outfiles', 'models', setting, 'spModel_{}_{}_{}_ep{}.pth'
                         .format(setting, str(lr), seed, epoch))))

        li_val_loss, li_val_acc, sp_val_loss, sp_val_acc = evaluate(epoch)
        listener_val_loss[epoch - 1], listener_val_acc[epoch - 1] = li_val_loss, li_val_acc
        speaker_val_loss[epoch - 1], speaker_val_acc[epoch - 1] = sp_val_loss, sp_val_acc

        # li_test_loss, li_test_acc, sp_test_loss, sp_test_acc = evaluate(epoch, 'test')
        # listener_test_loss[epoch - 1], listener_test_acc[epoch - 1] = li_test_loss, li_test_acc
        # speaker_test_loss[epoch - 1], speaker_test_acc[epoch - 1] = sp_test_loss, sp_test_acc

# To enable to hit Ctrl + C and break out of training:
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

mkdir(uc.paths['loss_acc'])
# Saving the loss and accuracy numpy arrays:
np.save('loss_acc/li_train_loss_{}_{}_{}'.format(
    str(lr), setting, seed), listener_train_loss)
np.save('loss_acc/li_train_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_train_acc)
np.save('loss_acc/sp_train_loss_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_train_loss)
np.save('loss_acc/sp_train_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_train_acc)
np.save('loss_acc/li_val_loss_{}_{}_{}'.format(
    str(lr), setting, seed), listener_val_loss)
np.save('loss_acc/li_val_acc_{}_{}_{}'.format(
    str(lr), setting, seed), listener_val_acc)
np.save('loss_acc/sp_val_loss_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_val_loss)
np.save('loss_acc/sp_val_acc_{}_{}_{}'.format(
    str(lr), setting, seed), speaker_val_acc)
# np.save('loss_acc/li_test_loss_{}_{}_{}'.format(
#    str(lr), setting, seed), listener_test_loss)
# np.save('loss_acc/li_test_acc_{}_{}_{}'.format(
#    str(lr), setting, seed), listener_test_acc)
# np.save('loss_acc/sp_test_loss_{}_{}_{}'.format(
#    str(lr), setting, seed), speaker_test_loss)
# np.save('loss_acc/sp_test_acc_{}_{}_{}'.format(
#    str(lr), setting, seed), speaker_test_acc)