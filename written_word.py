dataset_explanation = """
The objects of inquiry (i.e. the dataset), consist of various excerpts from a
number of translated publications of the Nicomachean Ethics into English.

The Nicomachean Ethics is one of Aristotle's best-known works, in which he
considers a number of virtues that are useful for living a good life (among
other things).


A number of passages are considered (`m`). I've selected passages that
introduce the various virtues. He treats each virtue in much more depth. I've
somewhat arbitrarily taken the first few sentences of each section, there was
some manual effort involved in typing out the passages found in my hard copies.
For the digital copies, I was just able to copy and paste the relevant text.
The passages correspond to the various virtues that Aristotle expands upon in
the above-named work.

For each passage, the corresponding translations are extracted from each source
and compared.

Each piece of text is a translation.

The object of this piece is to see what we can determine from the various
translations with the help of modern methods from the field of natural language
processing (NLP). It is educational, I'll explain some of the basic concepts
involved and provide sources for further understanding; and it is exploratory,
I'll be taking a look at things I've been interested in lately.

"""
cosine_similarity_methods = """
### Methods

At the heart of this analysis is the numerical representation of text data. The
text is represented as a vector, and the dimensionality depends on the
particular embedding model used. This vectorized representation of text is also
known as an embedding. To determine the similarity of any pair of texts (or
*translations* as I refer to them here), I make use of the cosine similarity
metric. Vectors have a length and a direction, and the cosine similarity is a
metric that tells us by how much any two vectors are pointing in the same
direction (independent of the lengths of each individual vector). If two pieces
of text are identical, they will be represented by the same embedding/vector,
which will have maximal cosine similarity with itself (the maximal value is 1).
Very similar texts will also have a high cosine similarity, and as the texts
begin to talk about different topics or about the same topic in different ways,
the cosine similarity between them will decrease. Eventually, once the texts
are considered to be completely unrelated to each other and the embeddings are
essentially orthogonal, the cosine similarity tends toward zero. The cosine
similarity may also become negative, and when maximally negative, that means
the embeddings are pointing in opposite directions.

As a first step, each translation is vectorized, and the cosine similarity is
calculated between each pair of vectors.

$$
\\text{cosine_similarity}(\\vec{A}, \\vec{B}) =
\\frac{\\vec{A} \\cdot \\vec{B}}{\\|\\vec{A}\\| \\|\\vec{B}\\|}
$$

"""

cosine_similarity_analysis = """## Analysis

The brighter colors correspond to higher similarity scores, and the darker
colors to lower similarity scores.

The figure above is a two-dimensional heatmap/histogram.


### Passage clusters and intra-passage translation pairs

The most prominent structure that emerges is the diagonal of squares through
the middle. Each individual square corresponds to a cluster of translations of
the same passage. I call one of these squares a *passage cluster*, since all
the translations in it correspond to the same passage. It is to be expected
that different translations of the same passage should have high similarities;
it is interesting however to see that some translations have high similarities
with translations of other passages, and some translations even have highER
similarities with at least one translation of a different passage than with at
least one translation of the same passage.

A passage cluster consists of many cells, and each cell corresponds to two
translations and a similarity score.

Intra-passage translation pairs correspond to translations of the same passage,
inter-passage translations correspond to translations of different passages.

I expected intra-passage translation pairs to have higher similarities than
inter-passage translation pairs, because intra-passage TPs correspond to the
same source material, whereas intra-passage TPs correspond to different
translation pairs.

There


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


### Inter-passage translation pairs

### Correlated clusters

"""

introduction_text = """

This is a little research project I undertook as a result of reading through
some of the Nicomachean Ethics. The Nicomachean Ethics is one of Aristotle's
best-known works, in which he considers a number of virtues that are useful for
living a good life (among other things).

-

"""

home_text = """

I started this project as a result of trying to understand just what the heck
Aristotle is going on about in one of his best-known works, the *Nicomachean
Ethics*.

I initially started reading the *Nicomachean Ethics* by finding a freely
available translation on Wikisource. It was an older translation from the early
1900s, and it's a work of philosophy, so I didn't exactly find it to be a light
read. Reading the same passage multiple times was helpful, but some things were
still a bit cryptic. I noticed that Wikisource had a second version of the
*Nicomachean Ethics*, translated by a different author. I began to read that as
well, switching between the two for a given passage to help me triangulate the
meaning of the original text.

I found this technique helpful, and also incredibly interesting. I noticed that
sometimes the translations were very similar, and sometimes they were very
different. I also started to see words used in the same context that I hadn't
consciously considered or fully realized were related before. It made sense I
thought about it, but I just hadn't thought about it before. (Just one example:
characteristic, disposition, trait.) It also piqued my curiosity about what the
original Greek words were that were being translated.

About a month or two after starting to read the versions on Wikisource, I
decided to pick up a physical copy. I figured it'd be interesting to see how
another translation differs from the two I'd been reading, and nice to have the
ability to read off-screen. I wandered into a local bookstore, and wound up
walking out with not just one, but three additional translations. These were
more modern, with publication dates ranging from the 1970s to the 2010s, and
they offered additional views of the original work.

One inconvenience that started to present itself was that going back and forth
between multiple books and web browser windows was tedious, and there wasn't
really any overall view I could make use of to facilitate drawing any kind of
aggregate conclusions. Being already very familiar with the methods and tools
of natural language processing from my day job, and being particularly disposed
to investigatory inquiries from my educational upbringing in the natural
sciences, I saw this is the perfect opportunity to create something useful. 

In this project I quantify just how similar the various translations are from
each other, present my findings in an interactive data dashboard, all while
giving myself additional exposure to this rich material I've been enjoying.

"""
