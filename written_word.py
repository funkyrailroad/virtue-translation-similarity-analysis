dataset_explanation = """

The objects of inquiry (i.e. the dataset), consist of a selection of passages
from the *Nicomachean Ethics* and multiple translations of each passage into
English. The bulk of the passages introduce the various character virtues, but
I've also included a number of other passages I enjoyed.


In an attempt to have uniformity of structure I've consistently taken the
introductory sentences of each passage that correspond to the character
virtues, but there is some grammatical variation in the way each is introduced.
This doesn't impact comparing different translations of the same passage, but
it may for comparing translations across different passages. There was some
manual effort involved in typing out the passages found in my hard copies, but
for the digital copies, I was able to copy and paste the relevant text.

Below are all of the translations grouped by passage.

"""
cosine_similarity_methods = """
### Methods

At the heart of this analysis is the numerical representation of text. Text can
be represented as a vector by a machine learning model, and the dimensionality
of that vector depends on the specific model used. This vectorized
representation of text is also known as an embedding, and the model that
converts the text to a vector can be referred to as the embedding model.

To quantify how similar any pair of translations are, I make use of a metric
known as the *cosine similarity*. Vectors have a length and a direction, and
the cosine similarity is a metric that tells us to what extent any two vectors
are pointing in the same direction (independent of the lengths of each
individual vector). If two pieces of text are identical, they will be
represented by the same embedding/vector, which will have maximal cosine
similarity with itself (the maximal value of the cosine similarity metric is
1). Very similar texts will also have a high cosine similarity, and as the
texts begin to talk about different topics or about the same topic in different
ways, the cosine similarity between them will decrease. In an extreme case of
when the texts are completely unrelated to each other, the embeddings will be
orthogonal and the cosine similarity will be zero. The cosine similarity may
also become negative, and when maximally negative, that means the embeddings
are pointing in exactly opposite directions. The cosine similarity of two
embeddings $$\\vec{A}$$ and $$\\vec{B}$$ is calculated as follows:

$$
\\text{cosine_similarity}(\\vec{A}, \\vec{B}) =
\\frac{\\vec{A} \\cdot \\vec{B}}{\\|\\vec{A}\\| \\|\\vec{B}\\|}
$$

Where $$\\vec{A} \\cdot \\vec{B}$$ is the dot product of the two vectors, and
$$\\|\\vec{A}\\|$$ is the magnitude of the vector.

"""

cosine_similarity_analysis = """## Analysis

For this analysis, each translation was vectorized and the cosine similarity
was calculated for every pair of translations. The results of these
calculations are visualized in the figure below. Each cell in the figure
corresponds to a pair of translations and the computed cosine similarity
between them. By clicking on a cell, the associated translations will appear to
the right.

The displayed cells correspond to a default range of cosine similarity values
(e.g. between 0.5 and 1), but this range can be dynamically adjusted with the
slider below the figure. For example, to visualize only the most similar pairs,
the bounds of the slider can be adjusted to e.g. 0.9 and 1.0.


"""


cosine_similarity_discussion = """## Discussion

The figure above is a two-dimensional histogram, which is also known as a
density heatmap. It consists of many cells, and each cell corresponds to two
translations and a similarity score. The brighter colors correspond to higher
similarity scores, and the darker colors to lower similarity scores.

*Intra*passage translation pairs correspond to translations of the same
passage, while *inter*passage translations correspond to translations of
different passages.

The most prominent structure that emerges is the diagonal of squares through
the middle. Each individual square is a cluster of the multiple translations of
a single passage i.e. intrapassage TPs. I call one of these squares a *passage
cluster*, since all the translations in it correspond to the same passage.

Outside of the main diagonal are cells that correspond to *inter*passage TPs
i.e. TPs from different passages. High similarities across many or all
intrapassage TPs from any two passages is indicative of a relationship between
those two passages, and I'll refer to them as *correlated passage clusters*.



### Intrapassage TPs and Passage Clusters

The different passage clusters differ in their coloring. A uniformly and
brightly colored cluster indicates that the various TPs have a high similarity
with each other. The inference I draw from this is that the different
translations have a high amount of agreement in the words used to convey the
meaning of the original text. This type of clusters may point to concepts that
are well understood, or at least uniformly understood amongst the various
translators. An easy way to identify these clusters is to increase the lower
limit of the "Cosine Similarity Range" slider until only the most brightly
colored passage clusters remain. (You may also click the lower limit and use
the arrow keys to adjust it.) The passage clusters for the virtue definition
and the virtue of courage are good examples of this.


On the other hand, a uniformly and darkly colored cluster indicates that the
various TPs have a relatively low similarity with each other. The inference I
draw from this is that the different translations have a relatively low amount
of agreement in the words used to convey the meaning of the original text.
These types of clusters may point to concepts that are poorly understood, or at
least understood differently by the various translators. These clusters can
also be identified interactively in the figure by decreasing the upper limit on
the slider until the passage clusters start to disappear. The passage cluster
for good-temperedness is a good example of this.


A remaining option is that a cluster may be colored heterogeneously, with both
brighter and darker individual squares within the passage cluster. The
inference I draw from this is that the differing translations have varying
levels of agreement with one another. This may be for a number of reasons. If
two are highly similar, where others diverge, it may be the case that one
author based their translation not only on the original text, but also on the
other author's translation. This would be an interesting feature to examine,
because it could offer a quantitative and computational way of analyzing the
different lineages of translations. It is also possible that most of the
translations are in agreement, and an outlier is present among them. This
analysis offers a way to identify the outliers, and further analysis can be
conducted. The passage cluster for shame has TPs with both high and low
similarity scores.


### Interpassage TPs and Correlated Clusters

It is no surprise that intrapassage TPs (translations of the same passage) have
high similarities; it is interesting however to see that some interpassage
translations also have a high similarity. These can be identified in the figure
above by the brighter spots that are off of the main diagonal of passage
clusters. These may be single, isolated cells, surrounded by TPs that have
otherwise low similarity scores, but they may also be surrounded by other TPs
with equal or greater similarity scores. In cases where multiple translations
of one passage have high similarities with multiple translations from another
passage, this likely indicates a significant similarity between the different
passages. The interpassage TPs for the passages on liberality and magnificence
offer examples of these.



translations have
high similarities with translations of other passages.



and some translations
even have highER similarities with at least one translation of a different
passage than with at least one translation of the same passage.
- give example of specific TPs that fulfill this


There are


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
thought about it, but I just hadn't thought about it before. It also piqued my
curiosity about what the original Greek words were that were being translated.

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

In this project I quantify just how similar the various translations are to
each other and present my findings in an interactive data dashboard, all while
giving myself additional exposure to this rich material I've been enjoying.

"""

conclusion_text = """

- overview
    - what this method of analysis allows one to do
    - some of the interesting findings
    - interesting words that are paired up:
        - pursuit and choice equivalency in opening passage

Another interesting aspect to investigate would be the effect that an
additional translation adds to a reader's understanding of the subject matter.
It may be the case that the two least similar translations offer the most
differing descriptions of the subject and possibly thus the most information
total. I see parallels here between information theory, entropy and statistical
mechanics.
- this is a key point, mention this earlier


It may be the case that there is a sweet spot in terms of the translation
differences such that translations that are too differing and too similar are
both less effective at providing additional understanding.



"""
