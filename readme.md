# The Turnip

"Music is the space between notes" - Claude Debussy.

A mathematical/computational investigation into the structure of ambiguity in Western harmony.

# Driving Curiosity  

What chords are the most versatile?   By versatile, I mean what chords can be used in the most contexts?

In this sense, by versatile I mean... "ambiguous".  
Why do I care about ambiguity?  Well, the more ambiguous a chord is, the more opportunities a composer has to use that chord.
But also, the more opportunities the composer has to break or follow the expectations of the listener.  
Playing with expectations--going back and forth between satisfying predictions and--is 

Take for example the C Major triad.  This is composed of the 3 notes C, E, and G.
This chord is written as [0, 4, 7], using pitch class notation.   

To what musical key signature does this chord belong?  Short answer:  C Major, G Major, and F Major.

Additionally, it belongs to multiple key signatures, but also numerous additional scales, some jazzy, some exotic, some scales perhaps never deliberately before used in recorded music.

Quick refresher on key signature:  I use the term  *key signature* and a diatonic major scale interchangeably, whereas a lot of musicians don't. 
If you're familiar with the concept "key signature of A minor", using my conventions 'A minor' is not a key signature, but rather it is equivalent to the key signature of C major.
There are twelve unique key signatures just as there are 12 unique diatonic Major scales.  
If you're familiar with modes, D dorian is equivalent to C Ionian or the C Major scale.  C dorian is equivalent to Bb Ionian or the Bb Major scale.

What about modes?
You actually don't have to worry about modes in this research project.  
Modes are where you "start" in a fixed collection of pitches--that is, which note is placed at the bottom.  
This project is about unique collections, not where you start.  
In this project, a C Major scale you can think of as all 7 modes at the same time, or superimposed.


So back to the question, which key signatures does C Major triad belong to?
If you know all twelve key signatures if you learned the basics of the theory of musical harmony,
 it's not too hard to run through all of them and check whether the C Major triad can be constructed using the 7 pitches in each key signature.  
You can of course find C Major Triad in C Major scale.  
But you can also find it in G Major scale. And the F Major scale! 

These three key signatures are neighbors on the circle of fifths.  
Notes further away on the circle of fifths have fewer notes in common and so it is less likely that you'd find the same chord in two scales if those two scales are far apart from one another on the circle of fifths.

We can say that C Major triad is a *member* of G Major, C Major, and F Major scales.

A more mathematical way of saying *member* is *subset*.  C Major triad is a subset of the G Major scale.



# What is a chord?

A chord is a collection of pitches.  
Chords generalize single notes, intervals, triads, and everything in between triads and scales.

Since a scale can have at most 12 notes corresponding to the chromatic scale 

# Definition of chord class

A *chord class* is a chord that is unique under inversion and transposition.

# Assumptions of chord classes

* Inversion doesn't matter.  ```G Bb D F``` is the same  as ```Bb D F G```, is the same as ```G D Bb F```
* Repeating pitch classes doesn't matter.  ```G Bb D F``` is the same as ```G G G G Bb D F G```
* Transposition doesn't matter for uniqueness.
* Chords that are reflections of one another are 

# Chord Enumeration

Chord Size | Number of unique chord classes | Example chord name and pitches
---------- | ------------------------------ | ------------------------------
0          |                              0 | NA
1          |                              1 | The note C natural.  [ 0 ]
2          |                              6 | Major third interval [ 0, 4 ]
3          |                             12 | Major third triad    [ 0, 4, 7 ]
4          |                             29 | Dominant 7th chord   [ 0, 4, 7, 10 ]
5          |                             38 | Pentatonic scale     [ 0, 2, 4, 7, 9 ]
6          |                             50 | Wholetone scale      [ 0, 2, 4, 6, 8, 10 ]
7          |                             38 | Major Scale          [ 0, 2, 4, 5, 7, 9, 11 ]
8          |                             29 | Octatonic Scale      [ 0, 1, 3, 4, 6, 7, 9, 10 ]
9          |                             12 |
10         |                              6 | 
11         |                              1 | Chromatic scale minus 1 note [ 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ]
12         |                              1 | The chromatic scale [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ]

For an excellent explanation of the mathematics behind the calculation of the number of unique chord classes, please refer to either "Enumeration in music Theory" by David Reiner or "Polya's Counting Theory" by Mollee Huisinga


# Dependencies

Python 3

# References

* Reiner, David L. "Enumeration in music theory." The American mathematical monthly 92.1 (1985): 51-54.
* Hook, Julian. "Why are there twenty-nine tetrachords? a tutorial on combinatorics and enumeration in music theory." Music Theory Online 13.4 (2007).



