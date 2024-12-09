## Free-response questions (6 points)

Your answers to all these questions should be submitted to Canvas as a single
PDF titled `hw3.pdf`. You can use whatever text editor you want to write your
answers (Word, LaTeX, Google Doc, etc.) but you must export your answers as a
single PDF file.

### 1. Loss functions (2 point)

In the model that you trained, there are loss functions for the autoencoder,
the generator, and the discriminator. In a standard GAN that just has a
generator and discriminator, the two losses are clearly in conflict: the
generator is being trained to fool the discriminator, and the discriminator is
being trained not to be fooled. Consider the other two pairs of losses in our
model; are the losses in conflict?

- a. Are the generator and autoencoder in conflict? That is, if you kept one
  submodel fixed and trained the other, would reducing the loss of the trained
  submodel cause the fixed submodel's loss to increase? Why or why not?

- b. Are the discriminator and autoencoder in conflict? That is, if you kept
  one submodel fixed and trained the other, would reducing the loss of the
  trained submodel cause the fixed submodel's loss to increase? Why or why not?

### 2. Class conditioning (3 points)

Suppose you wanted to add in class-conditioning to [the generator and
discriminator](https://arxiv.org/pdf/1411.1784.pdf) and [the
autoencoder](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)
in your model. Skim those linked papers. How would class-conditioning change
the implementation of your model? In particular, let's denote the image $X$ and
its label (which digit it is) as $y$. Consider our three submodels as
functions: $E(x), G(z), D(x)$. For each, say whether and how $y$ would factor
into these submodels' inputs. How would the inclusion of class-conditioning in
this submodel change the entire model's behavior? Give a two-sentence
explanation for each submodel.

- a. Encoder E(X)

- b. Generator G(z)

- c. Discriminator D(X)

### 3. Variational autoencoder (1 points)

Suppose you wanted to make our model more complicated by turning the
autoencoder we use above into a variational autoencoder. What would you need to
change in terms of model structure and/or loss functions? What might be one
benefit of doing this? Provide a 2-3 sentence explanation.
