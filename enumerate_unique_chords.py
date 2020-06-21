import itertools as it
from collections import Counter

import numpy as np
import pandas as pd
import json
import os
import time

from scipy.cluster.hierarchy import dendrogram, linkage

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



def modz(chord, z = 12):
     """ 
     :param chord:  A list of pitches (0 through 11)
     :param z:  int 12 for chromatic (Western) harmony, other ints for xenharmonic tunings
     :return:  List
     """ 
     return [x % z for x in chord]

def is_same_chord_class(chord1, chord2, z = 12):
    # chord "species"
    # like pitch class, but with chords.
    same_bool = False
    # make sure chord is valid
    chord1 = modz(chord1, z=z)
    chord2 = modz(chord2, z=z)
    for i in range(0,z):
        transposed_chord1 = [(x + i) % z for x in chord1]
        if set(chord2) == set(transposed_chord1):
            same_bool = True

    return same_bool 


def find_unique_chords(sorted_powerset, z = 12):
    """ 
    :sorted_powerset:  List of lists 
    :return:  List of dictionaries
    """ 
    i = 0  # global chord number
    k = 0  # count of chords per layer

    # first iterate over "layer" or size of chords, starting from the bottom.
    list_of_layers = []  # this will be a list of dicts 
    for layer in range(1, z + 1):
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
                    if is_same_chord_class(chord, list(key), z):
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
    with open('data/calculated_num_unique_chords_per_layer.json', 'w') as f:
        json.dump(polya_counter, f, sort_keys=True, indent=4)
    return polya_counter


# this should agree with binomial theorem

def validate_total_chords_per_layer(powerset_list):
    all_chord_lengths = [len(x) for x in powerset_list]
    all_counter = Counter(all_chord_lengths)
    print(all_counter)
    with open('data/calculated_num_chords_per_layer.json', 'w') as f:
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


def make_enum(parent_chord_or_scale, z = None, timer = False, save = False):
    start_time = time.time()
    if z is None:
        z = len(parent_chord_or_scale)
    test_ps = sorted_powerset(parent_chord_or_scale)
    test_enum  = find_unique_chords(test_ps, z)
    df_test_enum  = list_of_dicts_2_dataframe(test_enum)
    end_time = time.time()
    if save:
        if not os.path.exists('edo_enums'):
            os.mkdir('edo_enums')
        df_test_enum.to_csv(f'edo_enums/enum_{z}.csv')
    if timer:
        return (end_time - start_time)
    return df_test_enum

def time_make_enum(max_n, save):
    log = np.zeros(max_n)
    for x in range(1, max_n+1):
        log[x - 1] = make_enum(range(0, x), timer = True,  save = save)
    if not os.path.exists('edo_enums'):
        os.mkdir('edo_enums')
    with open('edo_enums/compute_time_seconds_1_to_N_edo.npy', 'wb') as f:
        np.save(f,log)
    return log


def test_polya_chords(parent_chord_or_scale, z = None):
    if z is None:
        z = len(parent_chord_or_scale)
    test_ps = sorted_powerset(parent_chord_or_scale)
    print(test_ps)
    test_enum  = find_unique_chords(test_ps, z)
    print(test_enum)
    
    df_test_enum  = list_of_dicts_2_dataframe(test_enum)
    print(df_test_enum)
     
    plot_polya_hists(df_test_enum)


# Make sure the algorithm is returning the same number of chords as 
# I calculated analytically with Polya's enumeration theorem.

def count_chords_per_layer(chromatic = range(12), z = None, mode = 'print'):
    if z is None:
        z = len(parent_chord_or_scale)
    # all chords 
    all_chords_ps = sorted_powerset(chromatic)

    if mode == 'print':
        print([x for x in all_chords_ps])

    validate_total_chords_per_layer(all_chords_ps)

    uniq_enum_df = make_enum(chromatic, z)

    validate_num_unique_chords_per_layer(uniq_enum_df)
    return uniq_enum_df


def main_old():
    count_chords_per_layer()

    ## Inspect output of the enumeration algorithm

    apply_dark_style()
    print (bcolors.OKBLUE + "All chords:" \
          + bcolors.ENDC)
    test_polya_chords(range(12)) # all 12 chromatic notes

def count_transpositional_siblings(parent_chord_class, child_chord_class, z = 12):
    s = 0 # number of siblings
    # transpose ccc z times
    for t in range(0, z):
        t_child_chord_class = [(x + t) % z for x in child_chord_class]
        if set(t_child_chord_class).issubset(parent_chord_class):
            s +=1
    return s
   

def make_matrix(enum_df, layer1, layer2, z, rename = True):
    if layer1 == layer2:
        raise ValueError('layers must be different') 
    l1_chords = enum_df[enum_df['layer'] == layer1].chord_class.values 
    l2_chords = enum_df[enum_df['layer'] == layer2].chord_class.values 
    # initialize matrix with rows for each chord class in layer 1,
    # columns for each chord class in layer 2. 
    matrix = np.zeros((len(l1_chords),len(l2_chords)))
    # naming convention is that layer 1 is parent, layer 2 is child
    for i, p_i in enumerate(l1_chords):
        for j, c_j in enumerate(l2_chords):
            # fill in matrix with the number of transpositional siblings
            matrix[i,j] = count_transpositional_siblings(p_i, c_j, z)  
    df_matrix = pd.DataFrame(data = matrix, columns = list(l2_chords), index = list(l1_chords))
    #df_matrix.rename(index=list(l1_chords), inplace=True)
    if rename:
        rename_dict = pd.Series(enum_df.chord_class_name.values,index=enum_df.chord_class).to_dict()
        df_matrix = df_matrix.rename(columns=rename_dict, index=rename_dict) 
    return df_matrix


def make_all_2layer_matrices(enum_df, z):
    # returns a dictionary of matrices
    layers = enum_df['layer'].unique()
    matrix_dict = {}
    for x in layers:
        for y in layers:
            if y >= x:
                continue
            matrix_dict[(x,y)] = make_matrix(enum_df, x, y, z)
    return matrix_dict
  
def assert_common_names_not_equivalent_pitch_class_sets(common_names, z):
    for c1 in common_names.keys():
        for c2 in common_names.keys():
            if c1 == c2:
                continue
            elif is_same_chord_class(c1, c2, z):
                scale1 = common_names[c1]
                scale2 = common_names[c2]
                print(scale1 + ' and ' + scale2 + ' are the same pitch class set.')

# only applies to 12-tet scales 
def build_common_name_dict_12_tone():
    common_names = {(0, 4, 7) : 'major_triad', (0, 3, 7) : 'minor_triad', \
    (0, 3, 6) : 'diminished_triad', (0, 4, 8) : 'augmented_triad'} 
    # more trichords 
    common_names[(0, 2, 4)] = 'do_re_mi' # same as fa so la, beginning of lydian, mixolydian, ionian
    common_names[(0, 2, 3)] = 're_mi_fa' # beginning of dorian and aeolian modes
    common_names[(0, 1, 3)] = 'mi_fa_so' # beginning of phrygian and locrian modes 
    common_names[(0, 2, 3)] = 're_mi_fa' # beginning of dorian and aeolian modes
    common_names[(0, 3, 10)] = 'weird_fishes' # first arpeggio in 'weird fishes' 
    common_names[(0, 4, 11)] = 'willy_wonka' # melody in 'come with me' in 'pure imagination' in willy wonka and the chocolate factory
    common_names[(0, 5, 10)] = 'quartal_triad' # mccoy tyner plays lots of these
    # seventh chords
    common_names[(0, 4, 7, 11)] = 'major_seventh'
    common_names[(0, 3, 7, 10)] = 'minor_seventh'
    common_names[(0, 4, 7, 10)] = 'dominant_seventh'
    common_names[(0, 3, 6, 10)] = 'half_diminished_seventh'
    common_names[(0, 3, 7, 11)] = 'minor_major_seventh'
    common_names[(0, 4, 8, 11)] = 'augmented_major_seventh'
    common_names[(0, 3, 6, 9)] = 'diminished_seventh'
    # 4-note jazz building block pieces 
    # these are things I add all the time for color in extended jazz chords.
    common_names[(0, 4, 6, 11)] = 'fabe' # rootless voicing for 7th chord, minor 6 9, etc
    common_names[(0, 5, 7, 11)] = 'cfgb' # negative harmony of fabe, dom 7-like, maj 7 like 
    common_names[(0, 3, 6, 11)] = 'diminished_major_seventh' # common in jazz. found in octatonic scale 
    common_names[(0, 5, 8, 11)] = 'inv_dim_major_seventh' # negative harmony of diminished major seven, also in octatonic 
    common_names[(0, 3, 5, 10)] = '4_stacked_fourths' # most versatile chord
    common_names[(0, 4, 5, 7)] = 'primordial_ooze' # ape escape soundtrack
    common_names[(0, 2, 4, 7)] = 'ff_major' # final fantasy prelude 
    common_names[(0, 4, 9, 11)] = 'careless_whisper' # george michael's 'careless whisper'
    #common_names[(0, 2, 3, 7)] = 'ff_minor' # final fantasy prelude, minor arp.  same class as careless whisper 
    # 5-note scales 
    common_names[(0, 3, 5, 7, 10)] = 'pentatonic_scale' # equivalent to minor7 add 11 chord
    common_names[(0, 2, 3, 7, 8)] = 'hirajoshi_scale' # japanese shamisen scale
    common_names[(0, 2, 3, 7, 9)] = 'insen_scale' # aka kumoi. brighter than hirajoshi, differs by one note  
    common_names[(0, 2, 3, 7, 10)] = 'minor_ninth' # versatile in electronic music
    #common_names[(0, 4, 7, 9, 11)] = 'major_7_add_13' # same ambiguity in major scale as major 7 chord  
    # same as minor_ninth
    # 6-note scales 
    common_names[(0, 2, 4, 7, 9, 11)] = 'major_7_no_avoid' #  major 7 compatible scale without 4th (lydian or ionian sound)
    common_names[(0, 3, 4, 7, 8, 11)] = 'augmented_scale' # contains 3 major 7 chords 
    common_names[(0, 2, 4, 6, 8, 10)] = 'wholetone_scale' # 
    # 7-note scales
    common_names[(0, 2, 4, 5, 7, 9, 11)] = 'major_scale'
    common_names[(0, 2, 3, 5, 7, 9, 11)] = 'altered_scale'
    common_names[(0, 2, 3, 5, 7, 8, 11)] = 'harmonic_minor_scale'
    common_names[(0, 2, 4, 5, 7, 8, 11)] = 'harmonic_major_scale'
    # 8-note scales
    common_names[(0, 1, 3, 4, 6, 7, 9, 10)] = 'octatonic'
    common_names[(0, 2, 4, 5, 7, 8, 9, 11)] = 'bebop_scale'
    # chromatic scale 
    common_names[(0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] = 'chromatic_scale'
    assert_common_names_not_equivalent_pitch_class_sets(common_names, z = 12)
    return common_names

def build_common_name_dict_19_tone():
    common_names = {(0, 6, 11) : 'major_triad', (0, 5, 11) : 'minor_triad'}
    common_names[(0, 5, 8, 11, 16)] = 'pentatonic_scale'
    common_names[(0, 3, 6, 8, 11, 14, 17)] = 'major_scale'
    common_names[(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)] = 'chromatic'
    assert_common_names_not_equivalent_pitch_class_sets(common_names, z = 19)
    return common_names


common_names_12tet_dict = build_common_name_dict_12_tone()
                
def rename_common_scales(df, z, common_names_dict = None):
    if common_names_dict is None:
        chromatic_key = tuple(range(0,z))
        common_names_dict = {chromatic_key : 'chromatic'}
    error_string = f"Pitch classes in common_names_dict are not numerically less than {z} in EDO {z}. Make sure pitch class sets are ascending."
    assert max([key[-1] for key in common_names_dict.keys()]) < z, error_string
    df = df.reset_index()
    unique_chord_classes = df.chord_class.unique()
    df['chord_class_name'] = df['chord_class']
    for i, chord_class in enumerate(unique_chord_classes):
        for common_chord in common_names_dict.keys():
           if is_same_chord_class(chord_class, common_chord, z):
               df.at[i,'chord_class_name'] = common_names_dict[common_chord]
    return df              

def get_inverse(chord, z = 12):
    inverse_chord_unmodded = [(z - note) for note in chord]
    inverse_chord = [note % z for note in inverse_chord_unmodded]
    return inverse_chord

def is_enantiomer(row, common_names_dict, z = 12):
    if common_names_dict is None:
        chromatic_key = tuple(range(0, z))
        common_names_dict = {chromatic_key : 'chromatic'}
    chord = row['chord_class']
    inverse = get_inverse(chord, z)
    if is_same_chord_class(chord,inverse, z):
        return 'non-chiral'
    else: # find common name of inverse
        for common_chord in common_names_dict.keys():
            if is_same_chord_class(common_chord, inverse, z):
                return common_names_dict[common_chord]
        return inverse 

def add_enantiomer(df, z, common_names_dict = None):
    if z == 12:
        if common_names_dict is None:
            common_names_dict = common_names_12tet_dict 
    df['enantiomer'] = df.apply(is_enantiomer, args = (common_names_dict, z), axis = 1)
    return df


def transpose_to_x_scale(chord, scale, z = 12):
    '''
    Specify a scale (not a class but a specific scale like D# Major, D# altered), 
    and a pitch class set will be transposed and returned as a subset of the specific scale.
    If transposition cannot bring into the scale, returns ""
    '''
    chord = list(chord)
    scale = list(scale)
    for i in range(0,z):
        transposed_chord = [(n + i)%z for n in chord]
        if set(transposed_chord).issubset(set(scale)):
            return sorted(transposed_chord)
        # for rare case where chord is the scale
        if set(transposed_chord) == (set(scale)):
            return sorted(transposed_chord)
    # if none of these transpositions work,
    # return empty chord 
    return [] 

    
    

def plot_matrix(matrix, color_style = "qual"):
    f, ax = plt.subplots(figsize=(11, 9))
    
    if color_style == "div":
        cmap = sns.diverging_palette(120, 10, s = 85, as_cmap=True, center = 'dark')
    elif color_style == "qual":
        cmap = sns.diverging_palette(120, 300, s = 85, as_cmap=True, center = 'dark')
    sns.set(font_scale = 0.6)
    
    # Draw the heatmap with correct aspect ratio
    sns.heatmap(matrix, cmap=cmap, center=0,
                square=True,  cbar_kws={"ticks":[0,1,2,3,4,5,6,7,8,9,10,11,12], "shrink": .5})
    #plt.show()

def matrix_dict_heatmaps(matrix_dict, subdir = ''):
    # heatmaps 
    heatmap_path = os.path.join('heatmaps', subdir)
    if not os.path.exists(heatmap_path):
        os.makedirs(heatmap_path)
    for matrix in matrix_dict.keys():
        plot_matrix(matrix_dict[matrix])
        filename = str(matrix[0]) + '_' + str(matrix[1])
        plt.savefig(os.path.join(heatmap_path, filename))
        plt.close()


def matrix_dict_dendrograms(matrix_dict, subdir = ''):
    # make all dendrograms given a list of matrices
    dendrogram_path = os.path.join('dendrograms', subdir)
    if not os.path.exists(dendrogram_path):
        os.makedirs(dendrogram_path)
    for matrix in matrix_dict.keys():
        print(matrix)
        if matrix_dict[matrix].shape[0] <= 1:
            continue 
        Z = linkage(matrix_dict[matrix], 'ward')
        dendrogram(Z, leaf_rotation=0, leaf_font_size=4, labels=matrix_dict[matrix].index, orientation='left') 

        filename = str(matrix[0]) + '_' + str(matrix[1])
        plt.savefig(os.path.join(dendrogram_path, filename)) 
        plt.close()

def matrix_dict_clustermaps(matrix_dict, z = None, subdir = ''):
    if z is None:
        z = max([matrix[0] for matrix in matrix_dict.keys()])
    clustermap_path = os.path.join('clustermap', subdir)
    if not os.path.exists(clustermap_path):
        os.makedirs(clustermap_path)
    dpi = 72.27
    for matrix in matrix_dict.keys():
        # if non-trivial clusterable
        # depends on type of matrix
        matrix_shape = matrix_dict[matrix].shape
        if (matrix_shape[0] > 1) & (matrix_shape[1] > 1):
            if ( matrix_dict[matrix].shape[0] > 20) | (matrix_dict[matrix].shape[1] > 20):
                f = 15
            else:
                f = 8
            sns.clustermap(matrix_dict[matrix],xticklabels=True, yticklabels=True, \
                figsize = (f,f), cbar_kws={'label': 'n transpositional siblings'})
            filename =  str(matrix[0]) + '_' + str(matrix[1])
            plt.savefig(os.path.join(clustermap_path, filename))
            plt.close()
    
def matrix_dict_clustermaps_all_subsets(matrix_dict, z = None, subdir = ''):
    # plot 'WIDE' matrix.
    if z is None:
        z = max([matrix for matrix in matrix_dict.keys()])
    clustermap_path = os.path.join('clustermap_wide_format', subdir)
    if not os.path.exists(clustermap_path):
        os.makedirs(clustermap_path)
    dpi = 72.27
    for matrix in matrix_dict.keys():
        # make sure elements are int.  
        # exclude matrix_dict[1] because it's empty
        if matrix_dict[matrix].shape[0] <= 1:
            continue 
        if isinstance(matrix, int) & (matrix > 1) :
            # cluster where possible
            cluster_rows = True 
            cluster_cols = True 
            # these matrices have only 1 row, so they can't be clustered 
            if (matrix == z) or (matrix == z - 1):
                cluster_rows = False
            # this matrix has only 1 column, so it can't be clustered 
            elif matrix == 2:
                cluster_cols = False

            print(matrix_dict[matrix])
            if matrix > 7:
                f = 30 
            elif matrix > 6:
                f = 25 
            elif matrix > 5:
                f = 20 
            elif ( matrix_dict[matrix].shape[0] > 20) | (matrix_dict[matrix].shape[1] > 20):
                f = 15
            else:
                f = 8
            sns.clustermap(matrix_dict[matrix],xticklabels=True, yticklabels=True, \
                col_cluster = cluster_cols, row_cluster = cluster_rows,\
                figsize = (f,f), cbar_kws={'label': 'n transpositional siblings'})
            filename =  str(matrix) 
            plt.savefig(os.path.join(clustermap_path, filename))
            plt.close()

def make_all_subset_matrices(two_layer_matrix_dict,  z):
    # 'WIDE' matrix
    # takes a dictionary of 2-layer matrices. 
    # returns an 11 element dictionary
    # shape of element number 5 is (n 5 scales, n_4_scales + n_3_scales + n_2_scales + n_1_scales).
    all_subset_dict = {}
    for x in reversed(range(1,z + 1)): 
        matrices_in_x = []
        for y in reversed(range(1, x)):
            matrices_in_x.append(two_layer_matrix_dict[(x,y)])
        #print(matrices_in_x)
        if len(matrices_in_x) > 1:
            all_subset_dict[x] = pd.concat(matrices_in_x, axis = 1)
        elif x != 1:
            all_subset_dict[x] = two_layer_matrix_dict[(x,y)]
    return all_subset_dict

def get_twelve_tone_table():
    x = make_enum([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    x = rename_common_scales(x, z = 12, common_names_dict = common_names_12tet_dict)
    x = add_enantiomer(x,  z = 12)
    return x

def get_nineteen_tone_table():
    x = make_enum(range(0, 19))
    print('renaming:')
    edo19_common_names_dict = build_common_name_dict_19_tone()
    x = rename_common_scales(x, z = 19, common_names_dict = edo19_common_names_dict)
    print('checking for enantiomers:')
    x = add_enantiomer(x, z = 19)
    return x

def main_twelve_tone():
    ## make a dataframe with the results from enumerating chords.
    ## Polya enumeration of unique chords invariant to transposition.
    x = get_twelve_tone_table()
    data_dir = 'data' 
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    x.to_csv(os.path.join(data_dir,'twelve_tone_pitch_class_sets_enumerated.csv'))

    omg = make_all_2layer_matrices(x)
    matrix_dict_dendrograms(omg)
    matrix_dict_clustermaps(omg, subdir = '')
    matrix_dict_heatmaps(omg)
    wide = make_all_subset_matrices(omg)

def main_edos(max_n):
    for edo in range(2, max_n+1):
        x = make_enum(range(0, edo))
        print(x)
        x = rename_common_scales(x, z = edo)
        x = add_enantiomer(x, z = edo)
        # to do:  add enantiomer to tables!
        edo_dir = "edo" + str(edo)
        print(edo_dir)
        omg = make_all_2layer_matrices(x, z = edo)
        print('making heatmaps')
        matrix_dict_heatmaps(omg, subdir = edo_dir)
        print('making dendrograms')
        matrix_dict_dendrograms(omg, subdir = edo_dir)
        print('making clustermaps')
        matrix_dict_clustermaps(omg, subdir = edo_dir)
        print('making wide matrix')
        wide = make_all_subset_matrices(omg, z = edo)
        matrix_dict_clustermaps_all_subsets(wide, z = edo, subdir = edo_dir)
    
    
def apply_dark_style():
    plt.style.use('dark_background')


if __name__ == '__main__':
    main_edos(max_n = 16)

