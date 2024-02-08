## Instructions

There are 20 points possible for this assignment. 2 points are for the setup,
8 points for the code, and 10 points for the free-response questions. The setup
portion is due earlier than the other pieces -- all deadlines are on Canvas.

### Setup (2 points)

All you need to do for these points is pass the `test_setup` case. This
requires putting your NetID in the `netid` file and creating seven PDF files
titled `XXX_qYYY.pdf` where `XXX` is replaced with your NetID, and `YYY`
ranges from 1 to 7. The content of these PDFs won't be graded, this is just to
ensure that you can set up your repository to be autograded. You can just
copy and rename the provided `blank.pdf` file, if you would like.

There is no `password` requirement for this assignment. Your final submission
must also pass the `test_setup` test, or you will lose these points.

### Coding (8 points)

You need to write code in every function in `src/` that raises a
`NotImplementedError` exception. Your code is graded automatically using the
test cases in `tests/`.  To see what your grade is going to be, you can run
`python -m pytest`; make sure you have installed the packages from
`requirements.txt` first. If the autograder says you get 100/100, it means you
get all 8 points.

The tests build on and sometimes depend on each other. We suggest that you
implement them in the order they appear in `tests/rubric.json`. That file also
allows you to see how many (relative) points each test is worth and which other
tests it may depend on. 

You may not use `fairlearn`, `sklearn`, or `scipy` to implement the functions
in this assignment.  However, you are welcome to look at the relevant
documentation; for example, for the [PolynomialFeatures](
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
and [LinearRegression](
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
classes.  Do not use the numpy functions `polynomial`, `polyfit` or `polyval`
for any of your solutions. Please do not use the python internal modules or
functions `importlib`, `getattr`, or `globals`. The `test_imports` case will
try to alert you if you use this disallowed packages or functions; please do
not try to circumvent these checks. If you think the test case is erroneously
penalizing you, please make a private Piazza post.
 
The grade given to you by the autograder on Canvas is the grade you should
expect to receive. If something goes wrong (your code times out, you import a
disallowed package, you accidentally push a syntax error, etc.) and your final
autograder grade is a 0, we will manually grade your code but subtract a 2
point penalty. Please be careful and read the feedback that the autograder is
giving you.

### Free response (10 points)

There are seven free response questions. Your answer to each should be in its
own PDF file, titled `XXX_qYYY.pdf`, where `XXX` is replaced with your NetID
and `YYY` is the number of the question. So if your netid were `xyz0123`, the
answer to question 1 should be in a file named `xyz0123_q1.pdf`.  For questions
with multiple parts, put all parts in the single PDF and just clearly label
where each part begins.  Please *do not put your name in these PDFs* -- we will
grade your work anonymously using only your netid in the filenames.

## Free response questions

### Question 1 (1 point)

You should not need to write any code for this question. For this question, you
will need to run some experiments using the code in `free_response/q1.py`. Note
that this file relies on your `PolynomialRegression` and `KNearestNeighbor`
code as well as some functions from `sklearn`. The code will create several
plots, some of which you will include in your answers to the parts below. 

If you run `python -m free_response.q1` from your repository's root directory
(i.e., the same place where you run `python -m pytest`), it should fill your
`free_response/plots/` folder with 32 different plots and add two "meta-plots"
to `free_response/`. You do not need to push these plots to GitHub!

- a. Look at the two "meta-plots" created by `free_response/q1.py`, one in
  `free_response/regression.png` and one in `free_response/knn.png`. Include
  these two meta-plots (not all 32 individual plots!) in your PDF for this
  question. Then, read the code in `q1.py` and figure out how it works.

  For each of these two plots, provide a 2-3 sentence description of the plot.
  What does it show? What is the X-axis and the Y-axes? Where does this plot
  show that models are overfitting or underfitting? Does this match up with
  what we discussed in lecture?

- b. Look through the 32 plots created by the `*_regression_experiment()`
  functions in `free_response/q1.py`, that will be saved to
  `free_response/plots/`. Find one plot where the model (either
  PolynomialRegression or KNearestNeighbor) is clearly overfitting, and add it
  to your PDF. How can you tell that the model is overfitting? Then, find one
  plot where the model is clearly underfitting, and do the same. For each plot,
  include at least a one-sentence explanation.

### Question 2 (1 point)

The [MovieLens dataset](https://grouplens.org/datasets/movielens/100k/) is a
dataset of 100,000 movie ratings, from which the provided `movielens` dataset
is sampled. In the data, the 1,000 users give ratings from 1 to 5 to movies
chosen from a list of 1,700 titles. In the data matrix (denoted `X`), if user `i`
rated movie `j`, then `X[i, j]` is that rating (an integer from 1 to 5). If
that user did not rate that movie, then `X[i, j] = 0`. Thus, most entries in
`X` are 0, because most users only rate a small number of movies.  You may
assume that every user has rated at least one movie with each of the five
ratings; that is, in every row, there is at least one 1, one 2, one 3, one 4,
and one 5.

Suppose we wanted to use a K Nearest Neighbor classifier to recommend movies to
users using this data. That is, for a given user who has rated some movies, we
want to predict the numerical score that user would give to a movie they have not
yet seen.

Consider the distance measures we discussed in class: Eucliean, Manhattan, and
Cosine. Which, if any, are a good choice for this problem? Why? Your answer
should discuss the fact that most entries in `X` are 0.

### Question 3 (2 points)

Suppose we have a data `X_train` with shape `(n1, n_features)` and a data
matrix `X_test` with shape `(n2, n_features)`, each with a corresponding array
of labels. Imagine we first fit a *k nearest neighbor regression* to `X_train`
and then use it to predict on `X_test`, and then second fit a *polynomial
regression* of degree `degree` to `X_train` and then use it to predict on
`X_test`.

For each question, compare the KNN and Polynomial Regression models. Provide an
explanation of at least one sentence that compares the models' behavior in
terms of `n1`, `n2`, and/or `n_features`.

- a. Which model takes longer to train on `X_train`? Why?
- b. Which model takes longer to predict on `X_test`? Why?
- c. Suppose you wanted to create a `.zip` file that, along with your code,
  would allow someone to make the exact same predictions as you on `X_test`.
  Which model would require a larger filesize to store it? Why?

### Question 4 (1 point)

For this and the following two questions, consider the following hypothetical:

> The city of Metropolis adopts a predictive policing system integrated with
> facial recognition technology powered by a K Nearest Neighbor classifier. Law
> enforcement uses historical crime data and real-time facial recognition to
> identify potential suspects in public spaces.

A KNN classifier typically makes predictions by majority vote -- it looks at
the `k` nearest neighbors and takes the most common class. 

How might relying on historical arrest data to train this facial recognition
classifier perpetuate racial discrimination in law enforcement? Explain how
this could affect the classifier's decisions, and what larger impact this could
have on law enforcement in the city.

### Question 5 (1 point)

For this question, refer back to the hypothetical from Question 4.

Suppose this predictive policing system is deployed and preliminary evidence
suggests it is effective at mitigating crime. However, critics of the system
mention issues with privacy and civil liberty.

Discuss a trade-off between crime prevention and another value, such as privacy
or fairness, which would affect your decision to implement the system.

How might you collect or analyze data to investigate the costs and benefits of
the system with regards to the trade-off you discussed above?

### Question 6 (1 point)

For this question, refer back to the hypothetical from Question 4.

Imagine you are a public defense lawyer and your client has just been arrested
based on a facial recognition classifier's analysis of security camera footage.
When law enforcement in the United States arrests someone, they're required to
demonstrate a "reasonable suspicion" of a specific individual based on
"specific and articulable facts", which are "taken together with rational
inferences from those facts". Using this standard of reasonable suspicion, how
might you argue that your client's arrest was unreasonable?

## Question 7 (3 points)

For this question, look at the code provided in `free_response/q7.py`.  You
shouldn't need to write any code for this question. Once you've implemented the
fairness metrics in `src/metrics.py`, you can run this script with `python -m
free_response.q7`. It will train some models, print out some important metrics
about those models, and visualize them in `free_response/fairness.png`.

The data for this problem (in `data/fairness_data.csv`) has four columns:
Income, Credit, Group, and Loan. Suppose that we want to predict whether an
applicant will receive a loan based on their income and credit. Loan is a
binary variable, and Income and Credit are continuous.  Group is some
demographic category (e.g. star-bellied or plain-bellied) which is binary in
this data.  We want our classifier to be fair -- it should not perform
differently overall for individuals with G=0 or G=1. The provided code will
train several LogisticRegression models on this data, and you will analyze
these models' behavior.

- a.  Look at the LogisticRegression model trained in `part_a()` and shown in
  the top left plot. In `free_response/fairness.png`, the area shaded grey
  shows positive predictions; the area shaded red shows negative predictions.
  * To what extent does the classifier treat individuals in Group 0 differently
    from those in Group 1? Answer in terms of the boundary and/or the metrics.
  * If you were applying for a loan and this classifier were making the
    decision, would you rather be a member of Group 0 or Group 1? Why? 

- b.  Consider the LogisticRegression model trained in `part_b()` and shown in
  the top right plot of `free_response/fairness.png`. Look at the code for
  `part_a` and `part_b`.
  * What's different about how this model was trained
    compared to the model from part __a.__?
  * How does this model's decision boundary and metrics differ from those of
    the model in part __a.__? Does this surprise you? Why or why not?

- c.  Look at the code for both LogisticRegression models trained in
  `part_c()` and visualized in the bottom two plots of `free_response/fairness.png`.
  * What is different about how each of these two models were trained?
  * If you were applying for a loan and were a member of Group 0, would you
    rather be classified by the part __c.__ "G=0" classifier or the classifier
    from part __b.__? Why?
  * What do you find interesting or surprising about the comparison between
    the model from part __b__. and the models from part __c.__? Why?
