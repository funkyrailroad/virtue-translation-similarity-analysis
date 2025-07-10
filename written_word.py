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
translations in our case here), I make use of the cosine similarity metric.
Vectors have a length and a direction, and the cosine similarity is a metric
that tells us by how much any two vectors are pointing in the same direction
(independent of the lengths of each individual vector). If two pieces of text
are identical, they will be represented by the same embedding/vector, which
will have maximal cosine similarity with itself. Very similar texts will also
have a high cosine similarity, and as the texts begin to talk about different
topics or about the same topic in different ways, the cosine similarity between
them will decrease. Eventually, once the texts are considered to be completely
unrelated to each other, the embeddings will become orthogonal. At this point,

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

The initial motivation for this particular inquiry was to bring a different
type of object to the forefront of analysis, inspired by the very first words
of the *Nicomachean Ethics*.

> "Every art and every inquiry, and similarly every action and pursuit, is
thought to aim at some good; and for this reason the good has rightly been
declared to be that at which all things aim. But a certain difference is found
among ends..."

My educational and professional background includes physics, computer
programming and natural language processing. In line with Thomas Kuhn's ideas
of *normal science* and *exemplars*, each discipline typically has a set of
methods and some classic, canonical examples that are used as an introduction
and foundational instruction. For example in physics, one of the foundational
examples is describing the trajectory of a falling apple to help understand
gravity. In computer programming, there are the famous "Hello world!" programs
to show how to output text to a screen in a given programming language. In
natural language processing, there's the email spam detector to show how to
train an AI model to automatically classify a selection of texts into
categories.

These are fantastic and relevant examples for their use cases and the ends they
pursue, but going back to the quote above, "... a certain difference is found
among ends..." Not everybody values each of these ends and respective means
equally, and there's no reason they should either. Everybody has their own
interests, problems and priorities. If you don't have a desire or need to
understand gravity, being able to describe the trajectory of a falling apple
may remain completely irrelevant to you. If you don't have a desire or need to
write a computer program, understanding how to output text to a screen in a
computer language might not be worth your time. If you don't have a desire or
need to automatically classify a selection of texts into categories, you very
well may be perfectly able to "rest in peace" without ever having built your
own email spam detector.

Having drawn that distinction between people and the relevancy of any given
end, any given pair of individuals also has a tremendous amount in common, if
for no other reason than that they are both human. So instead of just focusing
on differences and points of disagreement, are there any common aims and ends
that might be of interest to all humans? If so, what are they? And since we can
deem certain ends more and less relevant for a given individual, might we also
be able to do the same for humanity as a whole? And if this is also the case,
could there be one or more *supreme* ends that are deemed higher and more
important than the rest? If so, wouldn't it be nice to know what those ends
are? If you've responded in the affirmative to my leading questions, then you
are in luck, because this is exactly the subject matter of the Nicomachean
Ethics.


x talk about the arbirary nature of the ends
- introduce the answer to that question in the Nicomachean ethics
- having an answer to the question is all well and good, but how about also a
  methodology and even a framework get there?
- the exemplars are largely rooted in the discipline itself, and remain there.
  That is nice because it makes it self contained, but a consequence is that it
  is detached from anything potentially more significant than itself.
    - but there's a lot to be gained from interdisciplinary approaches
        - why not encourage interdisciplinary collaborations between
          disciplines we value?
- include a mention of or reference to Kuhn
- create new exemplars
    - I think an exemplar is a little broader than the examples I gave above
    - I think it would be interesting to create an exemplar that's based on one
      of those higher goods
        - This can be seen as an act of generosity, from one discipline to
          another. I'm gifting the methods of NLP to those disciplines that are
          also interested in Aristotle's virtue ethics.
- I would like to introduce a new toy example for NLP and semantic similarity.
  Comparing various translations of great works.
- I want to introduce the virtues
- this in to additional exemplars


==============================================================================

While working on this project, I came across a number of methods, and each has
their own canonical examples for demonstrating a particular type of analysis.

For example, Natural Language Processing (NLP) has typical examples like spam
detection in emails and classifying the sentiment of movie, product and
restaurant reviews. Another example, in dimensionality reduction, a typical
example used is related to the dimensions of flower petals in what is know as
the Iris flower dataset.

These are fantastic and relevant examples for their use cases and the ends they
pursue, but going back to the quote above, `... a certain difference is found
among ends...` Our lives are finite so we can't pursue every good and end we
come across or conceive, so which are the most worthy ends and goods to pursue?
If there is a single highest and most worthy aim/goal/end to pursue, what is
it? This is exactly the subject matter Aristotle covers in the Nicomachean
Ethics.


One of his conclusions is that a way to achieve the highest and most important
aim of life is to live virtuously by doing virtuous acts. He goes on to
delineate a number of virtues by explaining what they are and how people stray
from them by engaging in their associated vices.


I initially started reading the Nicomachean Ethics by finding a freely
available translation on Wikisource. Some of the words the translator used were
a little more difficult to understand. It was an older translation from the
early 1900s, and it's a work of philosophy, so I didn't exactly have an easy
time of understanding it. Reading the same passage multiple times was
necessary. I also noticed that Wikisource had a second version of the
Nicomachean Ethics, translated by a different author. I began to read that one
as well, switching between the two works for a given passage to help me
triangulate the meaning in the text.

I found this technique helpful, and also incredibly interesting. I started to
see words used in the same context that I'd never even considered were related
before. When you think about it makes sense, but I'd just never thought about
it before. (Just one example: characteristic, disposition, trait.) It also made
me curious about what the original Greek words were that they were trying to
translate.


About a month or two after starting to read the versions on Wikisource, I
decided to pick up a physical copy. I figured it'd be interesting to see how
another translation differs from the two I'd been reading, and to have the
ability to read off-screen. I wandered into a local bookstore, and wound up
walking out with three newer translations.

These were more modern, with publication dates

While going back and forth between the different sources, I noticed that
sometimes the translations were very similar, and sometimes they were very
different. Being already very familiar with the methods and tools of Natural
Language Processing from my day job, and being particularly disposed to
investigatory inquiries from my educational upbringing in the natural sciences,
I decided to try to quantify just how similar and different the different
translations are from each other, present my findings in an interactive data
dashboard, and give myself additional exposure to this rich material I've been
enjoying.


Also in an extremely hand-waving capacity, it's a slight look at how well AI
understands some ethics, virtues and vices.





#################

=============================

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
these objects of inquiry.

My initial approach for reading the Nicomachean Ethics has been going back and
forth between two English translations that are freely available on the
internet. They're written not exactly in "modern English", and any
philosophical text in general usually requires a few reads, so my general
approach has been to read a passage one or more times in one source, read the
same passage one or more times in the other source, and repeat that cycle one
or more times. Even just rereading the same thing over and over is
extraordinarily helpful, but having an additional translation available offers
even more insights. Ironically, there's a similarity to the allegory of the
cave here. Aristotle's original text is in Ancient Greek, and reading that
directly is currently beyond my grasp. The original text is akin to the ideal
Forms that are out of view of the cave dwellers. As a cave dweller, I only have
access to the Shadows, and in this case, the shadows would be the various
translations of the original texts. The shadows themselves can have different
shapes, although they are cast by the same Form. In a similar manner, each
translation of the original work has differences although they are all
embodiments from the same original work.


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
