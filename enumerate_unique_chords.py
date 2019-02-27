import itertools as it
from collections import Counter

CHROMATIC_SCALE = range(0,12)

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    chain = it.chain.from_iterable(it.combinations(s, r) for r in range(1, len(s)+1))
    ps_list = [list(x) for x in chain]

    return(sorted(ps_list, key = len))


#print(powerset([0,2,4,6]))

def modz(chord,z=12):
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


### FIX ME!!!
def find_unique_chords(sorted_powerset):
    i = 0  # global chord number
    k = 0  # count of chords per layer
    # first iterate over "layer" or size of chords, starting from the bottom.
    list_of_layers = []  # a list of dicts 
    for layer in range(1,13):
        i += k
        k = 0
        print(f'The layer is {layer}')
        uniq_in_layer = Counter()
        for chord in sorted_powerset[i:]:
            if len(chord) > layer:
                break
            k += 1
            print(chord)
            #print(k)
            if not bool(uniq_in_layer):  # if dict is empty when we are starting new layer
                uniq_in_layer[tuple(chord)] = 1
                print('empty dict')
            else:
                list_of_new_unique_chords = []
                for key in uniq_in_layer:
                    print(f'Chord is {chord}, key is {key}')
                    if is_same_chord_class(chord, list(key)):
                        print('duplicate!')
                        uniq_in_layer[key] += 1
                    else:
                        # found a unique chord
                        list_of_new_unique_chords.append(tuple(chord))
                for unique_new_chord in list_of_new_unique_chords:
                    uniq_in_layer[unique_new_chord] = 1
        # finish up!
        if bool(uniq_in_layer):  # if dictionary is not empty
           list_of_layers.append(uniq_in_layer) 
    
    return list_of_layers 

test_ps = powerset([0, 1, 2, 3])
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


