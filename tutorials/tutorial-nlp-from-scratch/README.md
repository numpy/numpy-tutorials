# Data used for building the [NLP from scratch tutorial](https://github.com/Dbhasin1/numpy-tutorials/blob/ethics-tutorial/content/tutorial-nlp-from-scratch.md)

## [IMDb Reviews Dataset](https://github.com/Dbhasin1/numpy-tutorials/blob/ethics-tutorial/content/tutorial-nlp-from-scratch/IMDB%20Dataset.csv)

**Purpose**: Training the Deep Learning model

> Information courtesy of
IMDb
(http://www.imdb.com).
Used with permission.

IMDB Reviews Dataset is a large movie review dataset collected and prepared by
Andrew L. Maas from the popular movie rating service, IMDB. The IMDB Reviews
dataset is used for binary sentiment classification, whether a review is
positive or negative. It contains 25,000 movie reviews for training and 25,000
for testing. All these 50,000 reviews are labeled data that may be used for
supervised deep learning. For ease of reproducibility, we'll be sourcing the
data from [Zenodo](https://zenodo.org/record/4117827#.YVQZ_EZBy3Ihttps://zenodo.org/record/4117827#.YVQZ_EZBy3I).

> Andrea Esuli, Alejandro Moreo, & Fabrizio Sebastiani. (2020). Sentiment 
Quantification Datasets [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.4117827

---

## [Glove Embeddings](https://github.com/Dbhasin1/numpy-tutorials/blob/ethics-tutorial/content/tutorial-nlp-from-scratch/glove.6B.50d.txt)

**Purpose**: To represent text data in machine-readable i.e numeric format
> Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

GloVe is an unsupervised algorithm developed for generating word embeddings by
generating global word-word co-occurence matrix from a corpus. You can download
the zipped files containing the embeddings from
https://nlp.stanford.edu/projects/glove/. 
Here you can choose any of the four options for different sizes or training
datasets, we opted for the least resource-heavy file with 50 dimensional
representations for each word. 

---

## [Speech Dataset](https://github.com/Dbhasin1/numpy-tutorials/blob/ethics-tutorial/content/tutorial-nlp-from-scratch/speeches.csv)

**Purpose**: The trained Deep Learning Model will perform sentiment analysis on
this data 
> Curated by the authors of the tutorial 

We have chosen speeches by activists around the globe talking about issues like
climate change, feminism, lgbtqa+ rights and racism. These were sourced from
newspapers, the official website of the United Nations and the archives of
established universities as cited in the table below. A CSV file was created
containing the transcribed speeches, their speaker and the source the speeches
were obtained from. 
We made sure to include different demographics in our data and included a range
of different topics, most of which focus on social and/or ethical issues. The
dataset is subjected to the CC0 Creative Common License, which means that is
free for the public to use and there are no copyrights reserved.

| Speech                                           | Speaker                 | Source                                                     |
|--------------------------------------------------|-------------------------|------------------------------------------------------------|
| Barnard College Commencement                     | Leymah Gbowee           | [Barnard College](https://barnard.edu/news/transcript-speech-nobel-peace-prize-winner-leymah-gbowee)                         |
| UN Speech on youth Education                     | Malala Yousafzai        | [The Guardian](https://www.theguardian.com/commentisfree/2013/jul/12/malala-yousafzai-united-nations-education-speech-text)                                              |
| Remarks in the UNGA on racial discrimination     | Linda Thomas Greenfield | [United States mission to the United Nation](https://usun.usmission.gov/remarks-by-ambassador-linda-thomas-greenfield-at-a-un-general-assembly-commemorative-meeting-for-intl-day-for-the-elimination-of-racial-discrimination/)                 |
| How Dare You                                     | Greta Thunberg          | [NBC](https://www.nbcnews.com/news/world/read-greta-thunberg-s-full-speech-united-nations-climate-action-n1057861)                                   |
| The speech that silenced the world for 5 minutes | Severn Suzuki           | [Earth Charter](https://earthcharter.org/new-voices-after-26-years-of-the-girl-who-silenced-the-world-for-5-minutes/)                                             |
| The Hope Speech                                  | Harvey Milk             | [Museum of Fine Arts, Boston](https://www.mfa.org/exhibitions/amalia-pica/transcript-harvey-milks-the-hope-speech)                 |
| Speech at the time to Thrive Conference          | Ellen Page              | [Huffpost](https://www.huffpost.com/entry/time-to-thrive_b_4794251)                                                  |
| I have a dream                                   | Martin Luther King      | [Marshall University](https://www.marshall.edu/onemarshallu/i-have-a-dream/)                    |
