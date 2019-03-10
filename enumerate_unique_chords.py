import itertools as it
from collections import Counter

import pandas as pd
import json


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def sorted_powerset(iterable):
    """
    sorted_powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    """
    s = list(iterable)
    chain = it.chain.from_iterable(it.combinations(s, r) for r in range(1, len(s)+1))
    ps_list = [list(x) for x in chain]

    return(sorted(ps_list, key = len))



def modz(chord,z=12):
     """ 
     :param chord:  A list of pitches (0 through 11)
     :param z:  int 12 for chromatic (Western) harmony, other ints for xenharmonic tunings
     :return:  List
     """ 
     return [x % z for x in chord]

def is_same_chord_class(chord1,chord2,z = 12):
    # chord "species"
    # like pitch class, but with chords.
    same_bool = False
    # make sure chord is valid
    chord2 = modz(chord2,z=z)
    for i in range(0,z):
        transposed_chord1 = [(x + i) % z for x in chord1]
        if set(chord2) == set(transposed_chord1):
            same_bool = True

    return same_bool 


def find_unique_chords(sorted_powerset):
    """ 
    :sorted_powerset:  List of lists 
    :return:  List of dictionaries
    """ 
    i = 0  # global chord number
    k = 0  # count of chords per layer

    # first iterate over "layer" or size of chords, starting from the bottom.
    list_of_layers = []  # this will be a list of dicts 
    for layer in range(1,13):
        i += k
        k = 0
        # print(f'The layer is {layer}')
        uniq_in_layer = Counter()
        for chord in sorted_powerset[i:]:
            if len(chord) > layer:
                break
            k += 1
            # print(f'Chord in ps is {chord}')
            if not bool(uniq_in_layer):  # if dict is empty when we are starting new layer
                uniq_in_layer[tuple(chord)] = 1
                # print('empty dict, not searching dict just appending new chord')
            else:
                # print('searching for duplicates in dictionary')
                unique = True
                for key in uniq_in_layer:
                    # print(f'Chord is {chord}, key is {key}')
                    if is_same_chord_class(chord, list(key)):
                        # print('is a duplicate chord class...  incrementing!')
                        uniq_in_layer[key] += 1
                        unique = False
                        break
                # found a unique chord
                if unique:
                    # print('found a new chord')
                    uniq_in_layer[tuple(chord)] = 1
                        
        # finish up!
        if bool(uniq_in_layer):  # if dictionary is not empty
           list_of_layers.append(uniq_in_layer) 
    
    return list_of_layers 



# this should agree with Polya's enumeration theorem.
def validate_num_unique_chords_per_layer(data_frame):
    chord_lengths = [len(x) for x in data_frame]
    polya_counter = Counter(chord_lengths)
    print(polya_counter)
    with open('num_unique_chords_per_layer.json', 'w') as f:
        json.dump(polya_counter, f, sort_keys=True, indent=4)
    return polya_counter


# this should agree with binomial theorem

def validate_total_chords_per_layer(powerset_list):
    all_chord_lengths = [len(x) for x in powerset_list]
    all_counter = Counter(all_chord_lengths)
    print(all_counter)
    with open('num_chords_per_layer.json', 'w') as f:
        json.dump(all_counter, f, sort_keys=True, indent=4)
    return all_counter


def list_of_dicts_2_dataframe(lod):
    df = pd.DataFrame([[1, key, i + 1, value ] for i, (key, value) in enumerate(lod[0].items())])
    for layer, counter in enumerate(lod[1:]):
        df2 = pd.DataFrame([[layer + 2, key, i + 1, value] for i, (key, value) in enumerate(counter.items())])
        df = df.append(df2)

    df.columns = ['layer', 'chord_class', 'num_in_layer', 'count', ]
    df = df.sort_values(['layer', 'num_in_layer'], ascending=[True, False])
    return df 



import matplotlib.colors as mcolors

# from stackoverflow: "Barplot colored according a colormap?"
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
[c('red'), 0.125, c('red'), c('orange'), 0.25, c('orange'),c('green'),0.5, c('green'),0.7, c('green'), c('blue'), 0.75, c('blue')])
# then pass in  color=rvb(x/N)



def plot_polya_hists(df, min_layer = 6, max_layer = 6):
    df = df[df['layer'] <= max_layer]
    df = df[df['layer'] >= min_layer]
    # to make sure correct format
    #tips = sns.load_dataset("tips")
    # print(tips)
    g = sns.FacetGrid(df, col = "layer",  margin_titles=True, sharex = False, hue = 'layer', palette = ['#28FE14'])
    g.map(plt.bar, "num_in_layer", "count")#, color=rvb((df.num_in_layer.values)/60))
    plt.show()


def make_enum(parent_chord_or_scale):
    test_ps = sorted_powerset(parent_chord_or_scale)
    test_enum  = find_unique_chords(test_ps)
    df_test_enum  = list_of_dicts_2_dataframe(test_enum)

    return df_test_enum


def test_polya_chords(parent_chord_or_scale):
    test_ps = sorted_powerset(parent_chord_or_scale)
    print(test_ps)
    test_enum  = find_unique_chords(test_ps)
    print(test_enum)
    
    df_test_enum  = list_of_dicts_2_dataframe(test_enum)
    print(df_test_enum)
     
    plot_polya_hists(df_test_enum)


# Make sure the algorithm is returning the same number of chords as 
# I calculated analytically with Polya's enumeration theorem.

def count_chords_per_layer(chromatic = range(12), mode = 'print'):
    # all chords 
    all_chords_ps = sorted_powerset(chromatic)

    if mode == 'print':
        print([x for x in all_chords_ps])

    validate_total_chords_per_layer(all_chords_ps)

    uniq_enum_df = make_enum(chromatic)

    validate_num_unique_chords_per_layer(uniq_enum_df)
    return uniq_enum_df


def main():
    count_chords_per_layer()

    ## Inspect output of the enumeration algorithm

    apply_dark_style()
    print (bcolors.OKBLUE + "All chords:" \
          + bcolors.ENDC)
    test_polya_chords(range(12)) # all 12 chromatic notes

def apply_dark_style():
    plt.style.use('dark_background')



if __name__ == '__main__':
    main()

