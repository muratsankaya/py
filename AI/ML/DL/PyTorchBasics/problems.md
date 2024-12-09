## Instructions

There are 20 total points possible for this assignment. The setup is worth 2
points; the coding and free response questions together are worth 18 points.
See `README.md` for the late work policy.

### Setup (2 points)

The setup for this assignment requires you to pass the `test_setup` cases. To
do so, all you need to do is put your NetID in the `netid` file and push that
change to GitHub. If you're familiar with `git`, this should take you about
two minutes -- if you aren't, this should help you learn the basics that
you need for

### Coding (14 points)

You need to write code in every function in `src/` that raises a
`NotImplementedError` exception. Your code is graded automatically using the
test cases in `tests/`. To see what your grade is going to be, you can run
`python -m pytest`; make sure you have installed the packages from
`requirements.txt` first. If the autograder says you get 100/100, it means you
get all 14 points.

The tests build on and sometimes depend on each other. We suggest that you
implement them in the order they appear in `tests/rubric.json`. That file
also allows you to see how many points each test is worth and which other
tests it may depend on.

The script in `models/save_model_without_deleting.py` provides some example
code for how you might save models into the `models/` folder to pass the
`test_saved_add_dataset` and `test_saved_multiply_dataset` cases. You can run
that code with `python -m models.save_model_without_deleting`. If you are
struggling to choose good hyperparameters for `src/experiments.py` to pass
those two tests, you might find this article helpful:
[http://karpathy.github.io/2019/04/25/recipe/](http://karpathy.github.io/2019/04/25/recipe/).

### Free response (4 points)

There are two free response questions. Your answer to each should be uploaded
to Canvas in its own PDF file, titled `qXXX.pdf`, where `XXX` is the number of
the question. So the answer to question 1 should be in a file named `q1.pdf`
and the answer to question 2 should be in a separate file named `q2.pdf`. For
questions with multiple parts, put all parts in the single PDF and just clearly
label where each part begins. Don't worry about how Canvas renames
resubmissions (e.g., `q1-2.pdf` for the second upload of a file named
`q1.pdf`), but please make sure that the first integer that appears in the
filename is the number of the question you're answering.

_Do not identify yourself in these PDFs_ -- we will grade your work anonymously
using only your netid in the filenames. We may deduct points if you include
your name or Net ID in the content of these PDFs.

## Free response questions

### Question 1 (2 point)

- 1a. What kinds of experiments did you run in your attempts to pass the
  `test_add_dataset` and `test_saved_add_dataset` cases? What hyperparameters
  seemed to be the most important? How does this match or conflict with your
  intuition about which hyperparameters are generally important? Why?

- 1b. What kinds of experiments did you run in your attempts to pass the
  `test_saved_multiply_dataset` cases? What hyperparameters seemed to be the
  most important? How does this match or conflict with your intuition about
  which hyperparameters are generally important? Why?

### Question 2 (2 points)

- 2a. Take a look at the [Pytorch documentation for DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). In your
  own words, why is it important that we use a `DataLoader` object to
  container our data, rather than something like a simple list of Tensors?

- 2b. Try running `python -m free_response.batch_sizes <num_examples>
<batch_size>` with different values of `num_examples` and `batch_size`. In
  your own words, what trends do you notice?

- 2c. Using `free_response/batch_sizes.py` as above, find a _single value_ of
  `num_examples` and _two values_ of `batch_size` such that the **LARGER**
  value of `batch_size` allows the model to train _faster_. Say which values
  you used for this to be true, and explain why this happens.

- 2d. Using `free_response/batch_sizes.py` as above, find a _single value_ of
  `num_examples` and _two values_ of `batch_size` such that the **SMALLER**
  value of `batch_size` allows the model to train _faster_. Say which values
  you used for this to be true, and explain why this happens.
