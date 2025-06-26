cosine_similarity_analysis = """#### Analysis

The brighter colors correspond to higher similarity scores, and the darker
colors to lower similarity scores.


The most prominent structure that emerges is the diagonal of squares through
the middle. Each individual square corresponds to a cluster of translations of
the same passage. It is to be expected that different translations of the same
passage should have high similarities; it is interesting however to see that
some translations have high similarities with translations of other passages,
and some translations even have highER similarities with at least one
translation of a different passage than with a translations of the same passage.


Intra-passage translation pairs correspond to translations of the same passage,
inter-passage translations correspond to translations of different passages.

I expected intra-passage translation pairs to have higher similarities than
inter-passage translation pairs, because intra-passage TPs correspond to the
same source material, whereas intra-passage TPs correspond to different
translation pairs.

There


##### Intra-passage translation pairs

##### Inter-passage translation pairs and Passage clusters

A passage cluster is a cluster of TPs that all correspond the same passage.
These form the most prominent feature of the heatmap: the diagonal of squares
found in the image. These clusters have a dimension of `n * n`, where `n`
corresponds to the number of translations included for a given passage. `n=4`
in our case.

The different passage clusters differ in their coloring. A uniformly- and
brightly- colored cluster indicates that the various TPs have a high similarity
with each other. The inference I draw from this is that the different
translations have a high amount of agreement in how they've translated the
original text. These types of clusters may point to concepts that are well
understood.


On the other hand, a uniformly- and darkly- colored cluster indicates that the
various TPs have a low similarity with each other. The inference I draw from
this is that the different translations have a low amount of agreement in how
they've translated the original text. These types of clusters may point to
concepts that are poorly understood, or at least widely-understood differently
by the various translators and readers.


A remaining option is that a cluster may be colored heterogeneously, with both
brighter and darker individual squares within the `n*n` cluster. The inference
I draw from this is that the differing translations have varying levels of
agreement with one another. This may be for a number of reasons. If two are
highly similar, where others diverge, it may be the case that one author based
their translation not only on the original text, but also on the other author's
translation. This would be an interesting feature to examine, because it could
offer a quantitative and computational way of analyzing the different lineages
of translations.

It is also possible that most of the translations are in agreement, and an
outlier is present among them. This analysis offers a way to identify the
outliers, and further analysis can be conducted. There.

Another interesting aspect to investigate would be the effect that an
additional translation adds to a reader's understanding of the subject matter.
It may be the case that the two least similar translations offer the most
differing descriptions of the subject and possibly thus the most information
total. I see parallels here between information theory, entropy and statistical
mechanics.

It may be the case that there is a sweet spot in terms of the translation
differences such that translations that are too differing and too similar are
both less effective at providing additional understanding.


##### Inter-passage translation pairs

##### Correlated clusters

"""


home_text = """
This might be a good spot for an introduction of the subject matter. Maybe even
list out all the quotes.

I've been working my way through the Nicomachean Ethics. I initially started
with Chapter 2 because that's where the doctrine of the mean is introduced,
(aka the golden mean), but I backed up to Chapter 1 to get some additional
context, and I've continued reading because I've been finding it interesting
and useful.

Aristotle asks this question, what are the proper objects of inquiry? That's
such a good question. There is so much out there, what are the things that are
worth your time, attention and effort? Proper conduct is one of them.

I have a long history of physics, math, analytics, programming, natural
language processing and data science, so I figured I apply those skills to
these objects of inquiry. My initial approach for reading the Nicomachean
Ethics has been going back and forth between two English translations that are
freely available on the internet. They're written not exactly in "modern
English", and any philosophical text in general usually requires a few reads,
so my general approach has been to read a passage one or more times in one
source, read the same passage one or more times in the other source, and
repeat that cycle one or more times. Even just rereading the same thing over
and over is extraordinarily helpful, but having an additional translation
available offers even more insights. Ironically, there's an similarity to the
allegory of the cave here. Aristotle's original text is in ancient greek, and
reading that directly is currently beyond my grasp. That's akin to the ideal
Forms that are out of view of the cave dwellers. As a cave dweller, I only
have access to the Shadows, and in this case, the shadows would be the various
translations of the original texts. The shadows themselves can have different
shapes, although they are cast by the same Form. In a similar manner, each
translation of the original work has differences although they are translated
from the same original work.


I was in a book store a couple months back and decided to pick up a physical
copy of another translation of the Nicomachean Ethics. I figured it'd be
interesting to see how one more translation differs from the two I'd been
reading, and to have the opportunity to read off-screen. I wound up walking
out with three.

It's been interesting to see how similar some of the passages in some of the
translations have been, but I think it's been most interesting to see how
different some of the passages in some of the translations have been. There's
a concept in natural language processing called "maximum marginal relevance"
(MMR), and it's used to find a broad span of similar things.

I used some quotes I'd already picked out when writing a previous article
(!add link here), and for some type of consistency, I used the opening few
sentences of each part of where Aristotle introduces a new virtue, for a total
of 18 quotes. I took the various translations from four different sources.

The first thing I wanted to take a look at is the similarity between all the
translations of the same quote, and of different quotes. At the very minimum,
the different translations of the same quote/passage should be more similar to
each other than they are to translations of different passages.

Honing in more closely to just look at the translations of one passage, I
would also expect that there are varying degrees of similarity amongst them.
They might be able to be meaningfully averaged, and the translation that is
closest to that average might be a good representative of the group. There may
be no meaningful average and no general consensus amongst the group. This may
be indicative of notoriously difficult passages to translate. There may be
trends in the translations of a particular translator. Maybe one translator is
consistently the most divergent from the group. Certain translators may have
used previously existing translations as a starting point and may be largely
similar.
            """
