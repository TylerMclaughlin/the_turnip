import itertools as it
from collections import Counter

CHROMATIC_SCALE = range(0,12)

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
        print(f'The layer is {layer}')
        uniq_in_layer = Counter()
        for chord in sorted_powerset[i:]:
            if len(chord) > layer:
                break
            k += 1
            print(f'Chord in ps is {chord}')
            if not bool(uniq_in_layer):  # if dict is empty when we are starting new layer
                uniq_in_layer[tuple(chord)] = 1
                print('empty dict, not searching dict just appending new chord')
            else:
                list_of_new_unique_chords = []
                print('searching for duplicates in dictionary')
                for key in uniq_in_layer:
                    print(f'Chord is {chord}, key is {key}')
                    if is_same_chord_class(chord, list(key)):
                        print('is a duplicate chord class...  incrementing!')
                        uniq_in_layer[key] += 1
                        break
                    else:
                        # found a unique chord
                        print('found a new chord')
                        list_of_new_unique_chords.append(tuple(chord))
                        
                for unique_new_chord in list_of_new_unique_chords:
                    uniq_in_layer[unique_new_chord] = 1
        # finish up!
        if bool(uniq_in_layer):  # if dictionary is not empty
           list_of_layers.append(uniq_in_layer) 
    
    return list_of_layers 

test_ps = sorted_powerset([0, 1, 2, 3])
print(test_ps)
print(find_unique_chords(test_ps))


# this should agree with Polya's enumeration theorem.
def validate_num_chords_per_layer(dict_of_chords):
    n = [len(x) for x in dict_of_chords]
    print(n)


def calculateChordFrequencyInScale(chord,scale):
    # how many major triads in C Major?
    # calculateChordFrequencyInScale([0,2,4],cMajor.getNotes() )
    # returns frequency "3" because CEG, FAC, and GBD are the three major scales
    chord = mod12(chord)
    scale = set(mod12(scale))
    frequency = 0
    for i in range(0,12):
        trialChord = [(x+i)%12 for x in chord]
        trialSet = set(trialChord)
        if trialSet.issubset(scale):
            frequency += 1
    return frequency


