## Instructions

There are 22 points possible for this assignment. 2 points are for the setup,
8 points for the code, and 12 points for the free-response questions. The setup
portion is due earlier than the other pieces -- all deadlines are on Canvas.

### Setup (2 points)

All you need to do for these points is pass the `test_setup` case. This
requires putting your NetID in the `netid` file and creating five PDF files
titled `XXX_qYYY.pdf` where `XXX` is replaced with your NetID, and `YYY`
ranges from 1 to 9. The content of these PDFs won't be graded, this is just to
ensure that you can set up your repository to be autograded.

There is no `password` requirement for this assignment. Your final submission
must also pass the `test_setup` test, or you will lose these points.

### Coding (8 points)

You need to write code in every function in `src/` that raises a
`NotImplementedError` exception. Your code is graded automatically using the
test cases in `tests/`.  To see what your grade is going to be, you can run
`python -m pytest`; make sure you have installed the packages from
`requirements.txt` first. If the autograder says you get 100/100, it means you
get all 10 points.

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

Note that `test_custom_transform` may require significant trial and error and
is only worth 0.5 points towards your final grade. *Please don't spend several
hours on it.* After you've given it an initial try, please take a look at FRQ2;
you can get points for describing what you tried to do, even if you didn't pass
the test case.

Hints:
  - For all questions in this assignment, you can use any numpy function. A
    helpful numpy function is np.sign: it returns the sign of each element in
    an array. That is, -1 if the element is less than 0, 1 if the element is
    greater than 0, and 0 if the element is 0.

  - For the `test_custom_transform`, you are similarly allowed to use any numpy
    functions – e.g., np.sin, np.cos, and related functions. The output of
    your custom_transform function should be a matrix of shape (N, 3) or (N, 2),
    but each of those columns should be new features you have created.
    That is, you don't need to copy the original (N, 2) input and only create
    one new feature – you should create 2 or 3 new features. Your choice of these
    features can and should rely on your knowledge that this function is a
    spiral. The only thing that's off limits is hardcoding the labels from
    spiral.csv into your feature transform. E.g., a trivial solution is to load
    the dataset and do a manual lookup for the label; don't do that. You may
    find it helpful to read [this article on
    spirals](https://en.wikipedia.org/wiki/Spiral).

### Free-response (12 points)

As before and as described above, each question should be answered in its own
PDF file. We will grade these anonymously; please do not include your name,
NetID, or other personally-identifying information. We will deduct points for
this if necessary.

## Free-reponse Questions

### 1. MNIST Classification (2 points)

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a famous
image classification dataset of handwritten digits that has been used as a
benchmark for many years. In this question, you'll train a couple models to
classify those digits and compare the results.  Look at the code provided to
you in `free_response/q1.py`, which you can run with `python -m
free_response.q1`.

Note that this FRQ requires downloading a ~15MB dataset and training models on
it. If you are running into trouble running this script, see
[`help_with_mnist.md`](help_with_mnist.md) or the hints below.

Hints for running faster:
  - In both **a.** and **b.**, you can decrease `n=10000` to a smaller number.
    This limits the size of the dataset. Note that if you limit it too much
    (e.g., n=20) you won't be able to say anything interesting.
  - In both **a.** and **b.**, you can change `max_iter` to reduce the amount of
    training time.
  - In **a.**, you can try commenting out all the code that uses Model from
    your `src/` directory. If you did something particularly inefficient, the
    runtime of **a.** might be affected. If only using sklearn models is still
    slow, try reducing `n` or `max_iter`.

**a.** In `two_class()`, make tweaks to the dataset and hyperparameters by
editing these lines:
```python
X, y = load_digits(classes=(0, 1), n=10000)
hidden_layer_sizes = (8, )
max_iter = 200

```

Describe at least two experiments you ran using different settings: you can
change `classes`, `n`, `hidden_layer_sizes`, and `max_iter`. You can also tweak
`build_model()` if you want. 

For each experiment you describe, write down the settings you chose and the
results that are printed out.  Then, describe what trends you notice. Do the
results match up with what we discussed in class?  Does anything surprise you?

**b.** In `ten_class()`, experiment with changes to the sklearn MLP by editing
the following lines:
```python
hidden_layer_sizes = (32,)
max_iter = 100
sklearn_kwargs = {}
X, y = load_digits(classes=range(10), n=10000)
```

Run at least two experiments using different settings. You may want to check
the [sklearn.neural_network.MLPClassifier docs](
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
to get a sense of the variety of arguments you can add into `sklearn_kwargs`.
For each experiment write down the settings you chose and the accuracies
printed out. Then, describe what trends.  What settings gave the best test set
accuracy?  Did anything surprise you?

### 2. Custom transform and the spiral dataset (2 points)

**a.** Were you able to pass the `test_custom_transform` case? If so,
explain what you did. If not, explain what you tried to do and why.
Provide as much detail as you can!

**b.** Run `python -m free_response.q2`; it will show a figure and save a copy
to your `free_response/` folder.  After running it once, take a look at the
code in `free_response/q2.py`. You can change whatever hyperparameters you
want, but focus on the regularization strength and penalty. Include a plot with
the best **test accuracy** you achieved and list the hyperparameter values you
used.  Which hyperparameters that you varied had the greatest effect on your
test accuracy results?

### 3. Loss Landscapes and the Effects of Batching (1 point)

Remember from lecture that we mentioned three types of gradient descent:
"batch," which uses the entire dataset to compute the gradients; "stochastic,"
which uses only a single example at a time (as with perceptron); and "mini-batch,"
which uses a batch size in between 1 and the size of the dataset.

Read through the code in `free_response/q3.py` to understand what it's doing,
and then vary the `batch_size` command-line argument to visualize how the loss
landscape changes between batches. Run the code with `python -m
free_response.q3 <batch_size>`, where `<batch_size>` must be an integer greater
than 0 and less than 11. When you run this script, it will save figures to your
`free_response/` folder.

**a.** Choose two batch sizes greater than 0 and less than 11 such that the
standard deviation printed out in the last line of the main function is much
larger for one batch size than the other. What explains the difference between
your two choices of batch size? Include at least two plots to back up your
explanation, but you must also include a written explanation.

**b.** What do your explorations tell you about the trade-off between batch
gradient descent, minibatch gradient descent, and stochastic gradient descent?

### 4. PyTorch autograd (1 point)

While we will not use PyTorch to build neural networks in this class (but you
should take CS449 Deep Learning in future quarters!), it and machine
learning frameworks like it make it much easier to develop custom networks than
the `src/` directory of this homework might indicate.

To get a sense of why this is, read [this PyTorch tutorial](
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). You'll
certainly need to read [A Gentle Introduction to
torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html),
but the sections titled "Tensors," and "Neural Networks," and "Training a
Classifier" will be helpful. You may also find [this article on backpropagation
and automatic differentiation](
https://akashe.io/blog/2020/11/09/backpropagation-and-automatic-differentiation/)
helpful.

**a.** What benefits can we obtain by using autograd? Consider the forward()
and backward() functions that you wrote in `src/` and how Pytorch handles similar
functions.

**b.** Why does PyTorch require that most arrays and matrices be stored in its
special tensor class (`torch.Tensor`) when `numpy` arrays are very similar?

### 5. Activation Functions (1 point)

**a.** We've mentioned in class some different activation functions. For an
activation to be useful in the intermediate layers of a neural network, what
are the **two** most important properties it must have? Why?

**b.** In class, we've discussed sigmoid and ReLU activations. Look through
some of the [many activation functions that Pytorch
provides](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions).
Pick one, describe it in detail, and explain one benefit it might have over
Sigmoid and/or ReLU activations.

### 6. Explainability versus Performance (2 points)

**a.** In your own words, what does it mean for a model to be explainable? Why
do we want models to be explainable?

**b.** When training and deploying a model, why might there be a trade-off
between explainability and performance (e.g., accuracy)?

**c.** When might we prefer a classifier that is more explainable, even if it
has worse accuracy? Create a hypothetical real-world example that demonstrates
this.

**d.** When might we not care about explainability at all? Or at least, care
much more about performance than explainability? Create a hypothetical
real-world example that demonstrates this.

### 7. Interpretability and LIME (1 point)

Interpretability techniques can provide more insight into a model's
prediction, but may come with certain challenges: lack of generalizability to unseen
data, a trade-off with performance, and vulnerability to adversarial attacks.

**a.** What are two arguments *in favor* of using Locally Interpretable Model
Explanations (LIME) to explain neural network predictions?

**b.** What is one argument *against* using LIME to explain neural network
predictions?

### 8. LIME and Fairness (1 point)

Your company develops an MLP to detect sensitive or obscene content in images
(nudity, blood, etc) on a social media website. The plan is to use that
classifier to filter such content out to improve users' experience on the
platform.  After implementing the model, early tests reveal that pictures of
women of color are more likely to be marked obscene than pictures of white
women in similar clothing/settings.

To address concerns of fairness, you decide to run LIME on this model to
explain its predictions. Assume these explanations come in the form of regions
of the image being shaded as "important for this prediction." You are welcome,
but not required, to go deeper into this question by reading some of [this
paper](http://proceedings.mlr.press/v139/garreau21a/garreau21a.pdf).

**a.** What evidence, if any, from these LIME explanations might persuade you
that the model **was** discriminatory against women of color? What evidence, if
any, would persuade you that it was **not** discriminatory?

**b.** Suppose LIME does not provide any evidence of discrimination. You have
to decide whether to keep using the MLP you developed to detect and filter user
content. What trade-offs would you consider in making this decision? What
would you tell the users of your social media platform?

### 9. Tumor Classifier (1 point)

You have a highly accurate model that's determining whether pictures of tumors
are benign or malignant. In order to bring this model to the medical field, you
have been asked to provide doctors with a better sense of how the model works.

You think you could use *post-hoc* interpretation tools like LIME to
understand your model. However, some team members propose the longer approach
of building a new model which is more *intrinsically interpretable*. This
approach in the long run could give you a deeper understanding of the model
than the post-hoc approach.

With limited resources, which project do you focus on first? There is no wrong
answer, but make sure to justify your decision-making based on how you could
best contribute to the medical field.
