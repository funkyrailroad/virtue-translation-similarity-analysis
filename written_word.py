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
