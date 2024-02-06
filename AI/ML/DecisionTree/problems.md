## Instructions

There are 18 points possible for this assignment. 2 points are for the setup,
11 points for the code, and 5 points for the free-response questions. The setup
portion is due earlier than the other pieces -- all deadlines are on Canvas.
Please carefully read this entire README before starting the assignment.

### Setup (2 points)

The setup for this assignment requires passing the `test_setup` cases:
`test_setup_netid` and `test_setup_password`.

To pass `test_setup_netid`, all you need to do is put your NetID in the `netid`
file and creating five PDF files titled `XXX_qYYY.pdf` where `XXX` is replaced
with your lower-case netid, and `YYY` ranges from 1 to 5 (for the five
free-response questions). For these setup points only, the content of these
PDFs won't be graded; you'll need to update these PDFs to contain your
free-response questions by the final deadline.

To pass `test_setup_password`, please follow the instructions written in
`tests/test_a_setup.py` as documentation for the `test_setup_password()`
function.

The purpose of these test cases is to make sure we're able to automatically
aggregate and grade your work. The autograder isn't sentient and can't fix your
mistakes for you -- these two setup points reward you for making it easy for us
to grade your work. Your final submission of the coding and free-response PDFs
should also pass the `test_setup` tests. If you delete your Net ID or if your
code has syntax errors, the autograder may give you a zero. If you make a
mistake that requires us to manually regrade your code or recompile your PDFs,
you will lose these setup points (and possibly other points).

### Coding (11 points)

You need to write code in every function in `src/` that raises a
`NotImplementedError` exception. Your code is graded automatically using the
test cases in `tests/`.  To see what your grade is going to be, you can run
`python -m pytest`; make sure you have installed the packages from
`requirements.txt` first. If the autograder says you get 100/100, it means you
get all 11 points.

For the numpy practice problems in `src/numpy_practice.py`, pay extra attention
to the docstring. The tests will make you implement it using certain numpy
functions, and will expect you to write each function in only one or two lines
of code. To enforce that you solve the test with the requisite number of lines
of code, we're checking to make sure that you haven't imported any additional
packages and that you haven't modified the docstrings we provide.

For this and all subsequent assignments, the test cases build on and sometimes
depend on each other. We suggest that you implement them in the order they
appear in `tests/rubric.json`. That file also allows you to see how many points
each test is worth and which other tests it may depend on. 

### Free response (5 points)

There are five free response questions. Your answer to each should be in its
own PDF file, titled `XXX_qYYY.pdf`, where `XXX` is replaced with your
lower-case NetID and `YYY` is the number of the question. So if your netid were
`xyz0123`, the answer to question 1 should be in a file named `xyz0123_q1.pdf`
and the answer to question 2 should be in a separate file named
`xyz0123_q2.pdf`. For questions with multiple parts, put all parts in the
single PDF and just clearly label where each part begins.

*Do not identify yourself in these PDFs* -- we will grade your work anonymously
using only your netid in the filenames. We may deduct points if you include
your name or Net ID in the content of these PDFs.

## Free response questions

### Question 1 (1 point)

The ID3 algorithm we implemented is relatively simple, in that it has
no hyperparameters (i.e., design choices about the algorithm that need to be
made before running it on a given dataset). Read the [scikit-learn
DecisionTreeClassifier documentation](
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
and read through some of the hyperparameters that the model allows
you to adjust. Pick one of the following hyperparameters:

- `min_samples_split`
- `max_features`
- `max_leaf_nodes`
- `min_impurity_decrease`

For each hyperparameter you choose, give a one-sentence explanation of what
it does to affect the kinds of trees that the algorithm learns.
Then, describe whether a small or large value for this hyperparameter
is more likely to cause the tree to *overfit*.

An example answer to this question for the `max_depth` hyperparameter,
(which you aren't allowed to use), might look like:

> The `max_depth` argument controls the maximum depth (longest path from the
> root to any leaf) of the tree. Using a large value of `max_depth` is more
> likely to overfit, because if the tree gets so deep that each leaf only
> corresponds to a single training example, we wouldn't expect it to generalize
> well to new unseen test data.

### Question 2 (1 point)

Look at the formulas for the Equalized Odds and Demographic Parity metrics.
We discussed these in lecture, or you can refer to
[Wikipedia](https://en.wikipedia.org/wiki/Fairness_(machine_learning)#Definitions_based_on_predicted_outcome).
Note that on Wikipedia, these metrics are framed as equalities -- when a
classifier is not fair, such an equality will not hold.  Then, for each metric,
answer the following questions:

* What do the variables in the formula mean? 
* In your own words, what is the formula doing?

### Question 3 (1 point)

In class, we have discussed examples such as predicting housing prices or predicting
whether someone will survive the sinking of the Titanic. For parts a. and b., make
up your own examples for the situation. These can just be a sentence or two, but make it clear
how the situation matches what the formula is doing.

* a. Create an example of a situation where two classifiers are equally
accurate, but one is "fairer" according to **Equalized Odds** metric. What factors
might lead us to prefer the "fairer" classifier over the other?

* b. Create a separate example of a situation where two classifiers are equally
accurate, but one is "fairer" according to **Demographic Parity**. What
factors might lead us to prefer the "fairer" classifier over the other.

### Question 4 (1 point)

Suppose you have two classifiers A and B, such that A scores better than B on
Equalized Odds but A scores worse than B on Demographic Parity.

* a. Based on the formulas, why might this happen?
* b. If you were trying to decide which classifier to use, what additional
questions might you ask to decide whether you care more about Equalized Odds or
Demographic Parity?


### Question 5 (1 point)

In the ID3 algorithm, we recursively build a decision tree by choosing an
attribute to split that will maximize information gain. Suppose we wanted to
incorporate Equalized Odds or Demographic Parity into our decision tree
learning algorithm.

* a. How might this work? You don't need to write an entire algorithm,
but try to provide as many specific details as you can.

* b. What challenges might you encounter in implementing your approach?

## Citations

If you discussed the homework with other students in any way (except via
Piazza), please disclose those collaborations in the [`CITATIONS`
file](CITATIONS). If any online resources may have influenced your approach to
solving these questions (e.g., you saw a helpful guide to the ID3 algorithm),
please link to them. If you used a large language model (e.g., ChatGPT) in any
way, please describe that use.
