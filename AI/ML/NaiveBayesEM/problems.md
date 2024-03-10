## Instructions

There are 24 points possible for this assignment. 2 points are for the setup,
16 points for the code, and 6 points for the free-response questions. The setup
portion is due earlier than the other pieces -- all deadlines are on Canvas.

### Setup (2 points)

All you need to do for these points is pass the `test_setup` case. This
requires putting your NetID in the `netid` file and creating five PDF files
titled `XXX_qYYY.pdf` where `XXX` is replaced with your NetID, and `YYY`
ranges from 1 to 5. The content of these PDFs won't be graded, this is just to
ensure that you can set up your repository to be autograded.

Your final submission must also pass the `test_setup` test, or you will lose
these points.

# Coding (16 points)

Start by solving the practice problems in `src/sparse_practice.py`; these
will help you understand how scipy sparse matrices work.

In `src/utils.py`, you will write `softmax` and `stable_log_sum` functions,
making sure that both are numerically stable. These will be helpful in your
Naive Bayes models.

Then, you will implement two versions of a Naive Bayes classifier.  In
`src/naive_bayes.py`, the `NaiveBayes` classifier considers the case where all
the data is labeled.  In `src/naive_bayes_em.py`, `NaiveBayesEM` classifier
will use the Expectation Maximization algorithm to also learn from unlabeled
data.

The lecture slides and the PDF write-up (posted to Canvas by February 24) will
be helpful.  We have also provided extensive documentation in the provided
code, please read it carefully!  For example, when implementing the
`NaiveBayesEM` classifier, be careful to correctly initialize your parameters
and correctly update your inferred class distribution during the E-step (i.e.,
do not overwrite the observed labels with your predicted probabilities).

Your goal is to pass the test suite that is run by `python -m pytest`.
Once the tests are passed, move on to the free-response questions below.
The tests build on and sometimes depend on each other. We suggest that you
implement them in the order they appear in `tests/rubric.json`. That file also
allows you to see how many (relative) points each test is worth and which other
tests it may depend on. 

You may not use `sklearn` or `scipy` to implement the functions in this
assignment. Please do not use the python internal modules or functions
`importlib`, `getattr`, or `globals`. The `test_imports` case will try to alert
you if you use this disallowed packages or functions; please do not try to
circumvent these checks. If you think the test case is erroneously penalizing
you, please make a private Piazza post.
 
The grade given to you by the autograder on Canvas is the grade you should
expect receive. If something goes wrong (your code times out, you import a
disallowed package, you accidentally push a syntax error, etc.) and you need us
to grade your code manually, we will do so but subtract a 2 point penalty.
Please be careful and read the feedback that the autograder is giving you.

## The speeches dataset

The dataset provided in `data/speeches.zip` (which should be automatically
unzipped by the `src/data.py` code) contains [State of the Union
addresses](https://en.m.wikisource.org/wiki/Portal:State_of_the_Union_Speeches_by_United_States_Presidents)
by United States presidents dating back to 1790. In recent history
(specifically, since the [Civil Rights Act of
1964](https://simple.wikipedia.org/wiki/Party_realignment_in_the_United_States#1960s-80s)),
all US presidents have belonged to one of two political parties, which have
remained relatively ideologically stable.

In this data, we treat the words of each speech as the features, and the
political party of the speaker as the label.  For presidents prior to 1964, we
will treat their party membership as unobserved, as it does not necessarily
correspond to today's two-party system. The `NaiveBayes` classifier will only
consider fully-labeled data -- it cannot use speeches prior to 1964. The
`NaiveBayesEM` classifier will also use unlabeled speeches to learn its
probabilities.

If the provided code fails to unzip the `speeches.zip` file, please ask for
help.

# Free-response questions (6 points)

As before and as described above, each question should be answered in its own
PDF file. We will grade these anonymously; please do not include your name,
NetID, or other personally-identifying information. We will deduct points for
this if necessary.

## 1. Comparing Naive Bayes with and without unlabeled data (1 point)

Try running `python -m free_response.q1`. The code creates a dataset using
`src.data.build_dataset` with 100 documents, at most 2000 words per document,
and a vocabulary size of 1000 words. As above, we will consider speeches prior
to 1964 as unlabeled. The code fits your `NaiveBayes` and `NaiveBayesEM` models
on the `data` matrix and `labels` array. For the `NaiveBayesEM` model, it
trains with `max_iter` values of 1, 2, and 10.

If your implementation is correct, we expect it to produce values similar to
those in the table below. However, even if your code does not produce similar
values, please answer this question using the values below.

| Model | `max_iter` | Accuracy | Log Likelihood |
| ---   | ---        | ---      | ---            |
| NB    | N/A        | 100%     | -40029.1       |
| NB+EM | 1          | 100%     | -60296.3       |
| NB+EM | 2          | 96.6%    | -59930.9       |
| NB+EM | 10         | 96.6%    | -59878.4       |

Consider the differences in accuracy and likelihood between the `NaiveBayes`
and `NaiveBayesEM` models. Why do the `NaiveBayesEM` models have a lower
likelihood than the `NaiveBayes` model? Why is accuracy not positively
correlated with likelihood?

## 2. Interpreting `beta` values (1 point)

Try running `python -m free_response.q2`. The code is similar to that of
`q1.py` above; it trains a `NaiveBayesEM` model on a similar dataset.  After
training, it extracts some specific values of the model's learned `beta` and 
plots them in a table.

If your implementation is correct, we expect it to produce values similar to
those in the table below. If your code produces very different values, please
answer this question using the table below.

|Word       |  Beta[:, 0] | Beta[:, 1] |
| ---       | ---         | ---        |
|banks      |  0.311      | 0.051      |
|loans      |  0.600      | 0.119      |
|ought      |  0.311      | 0.068      |
|terror     |  0.044      | 0.237      |
|strategic  |  0.111      | 0.542      |
|allies     |  0.133      | 0.644      |
|set        |  0.644      | 0.644      |
|policies   |  0.644      | 0.644      |
|expect     |  0.356      | 0.356      |

In your own words, what is this code doing? What does the table above say about
words like "banks," "terror," and "set"?

## 3. Probabilistic predictions (2 points)

Try running `python -m free_response.q3`. The code is similar to that of
`q1.py` and `q2.py` above; it trains a `NaiveBayesEM` model on a similar
dataset. After training, it uses the `predict_proba` function to predict the
probability that each labeled speech in the training data is from a Democratic
or Republican president. We can think about these probabilities as describing
the "confidence" of the classifier -- if it outputs a probability of 50% for
both labels, the model is unconfident. When the model's probability of the
predicted label is close to 100%, it is very confident in its prediction.

If your implementation is correct, we expect it to produce output similar to
the below.  However, if your implementation gives very different results,
please use these results instead.

```
Across 56 correct predictions,  NBEM has average confidence 99.51%
Across 2 incorrect predictions, NBEM has average confidence 99.82%
```

(a) What is one aspect of the data and/or model that could contribute to the
patterns you see in these results? Justify your answer as best you can.

(b) Suppose we were using a machine learning classifier in a high-stakes domain
such as classifying whether images of tumors are benign or malignant. What
might be one danger of having a classifier that is always confident, even when
incorrect?

(c) In the results above, out of 58 predictions the model was correct 96.6% of
the time, but its average confidence was above 99%. If the [model were
calibrated](https://scikit-learn.org/stable/modules/calibration.html), a
prediction with confidence of X% would be correct X% of the time. What would be
at least one benefit of a calibrated classifier, where its confidence
represents the probability that it makes a correct prediction?

(d) Suppose you wanted to introduce a form of regularization to make our
NaiveBayesEM model less overconfident in its predictions. To do this, you might
modify how we use `smoothing` or use a prior probability to guide how we
learn `beta`. Propose a method for doing this and justify why this would work.

## 4. Fairness metrics revisited (1 point)

(a) Go back to the HW 1 materials and look at the definitions for Equalized
Odds and Demographic Parity. Rephrase these definitions as statements about
conditional independence of variables.

(b) 2NB is a Naive Bayes method modified to make fairer classifications,
described [in this paper](
https://link.springer.com/article/10.1007/s10618-010-0190-x).
It uses demographic parity (also known as statistical parity) to measure
fairness. Suppose you use a 2NB model to diagnose whether or not a
patient has sickle cell anemia (a blood disorder which is [especially common
among Black Americans](https://www.cdc.gov/ncbddd/sicklecell/data.html)).
What is one issue that could arise from using demographic parity to ensure
this diagnoses are fair? Justify your answer.

(c) What is one other way you could try to make your model fairer in these
predictions? Explain why this would be a good choice.

## 5. Naive Bayes for medical diagnosis (1 point)

Suppose you have developed a Naive Bayes model to diagnose patients' tumors as
benign or malignant.  How might you explain this model to medical practitioners
who wish to use it effectively and responsibly?

This is meant to be open-ended, but for full points consider at least the
following three ideas:
- Informed consent: Patient data may be used to train and evaluate the model,
  and the model may make errors. What do patients need to know?
- Harm prevention: How do you minimize unnecessary treatments, especially ones
  that are risky or have harmful side effects (such as chemotherapy)? How do
  you maximally improve patient survival?
- Accountability: What are some possible scenarios in this field where an NB
  model might be less accurate or less fair?
