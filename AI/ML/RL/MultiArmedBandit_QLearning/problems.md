# Coding (8 points)

Your task is to implement two reinforcement learning algorithms:

1.  Multi-armed Bandits (in `src/multi_armed_bandits.py`)
1.  Q-Learning (in `src/q_learning.py`)

Note that while these reinforcement learning methods inherently depend
on randomization, we provide a `src/random.py` package that will randomize
things in the same way for all students. Please use `src.random` anywhere
that you might have otherwise used `np.random`.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are
passed, you will use your code to answer FRQ 2 and 3.

We suggest that you try to pass the tests in the order they are listed in
`tests/rubric.json`.

Your grade for this section is defined by the autograder. If it says you got an 75/100,
you get 75% of the coding points.


# Free-response questions (16 Total points)

To answer some of these questions, you will have to write extra code (that is
not covered by the test cases). You may (but are not required to) include your
experiments in new files in the `free_response` directory. See
`free_response/q2.py` for some sample code. You can run any experiments you create
within this directory with `python -m free_response.<filename>`. For
example, `python -m free_response.q2` runs the example experiment.

## 1. (2 points) Tic-Tac-Toe
Suppose we want to train a Reinforcement Learning agent to play the game of
[Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe), and need to construct
an environment with states and actions. Assume our agent will simply choose
actions based on the current state of the game, rather than trying to guess
what the opponent will do next.

- a. What should be the states and actions within the Tic-Tac-Toe Reinforcement
  Learning environment? Don't try to list them all, just describe how the rules
  of the game define what states and actions are possible.  How does the
  current state of the game affect the actions you can take?
- b. Design a reward function for teaching a Reinforcement Learning agent to
  play optimally in the Tic-Tac-Toe environment.  Your reward function should
  specify a reward value for each of the 3 possible ways that a game can end
  (win, loss, or draw) as well as a single reward value for actions that do not
  result in the end of the game (e.g., your starting move). Explain your
  choices.
- c. Suppose you were playing a more complicated game with a larger board, and
  you want the agent to learn to win as fast as possible. How might you change
  your reward function to encourage speed?


## 2. (3 points) Bandits vs Q-Learning

- a. Run `python -m free_response.q2`; it will create three plots:
  `2a_SlotMachines_Comparison.png`, `2a_FrozenLake_Comparison.png`, and
  `2a_SlipperyFrozenLake_Comparison.png`. It might help to read a bit about the
  [FrozenLake environment](
  https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) .
  Each plot will show a comparison of your MultiArmedBandit and QLearning
  models on the named environment (e.g., SlotMachines). Include those plots
  here. For each plot, provide a one-sentence description of the most notable
  trend. Pay attention to the scale on the y-axis.

- b. In which of the above plots does QLearning appear to receive higher
  rewards on average than MultiArmedBandit? Provide an explanation for
  why that happens, based on your understanding of QLearning.

- c. Following b.: in the environment(s) where MultiArmedBandit was the
  **worse** model, is there any way you could change your choice of
  hyperparameters so that MultiArmedBandit would perform as well as QLearning?
  Why or why not?

- d. In which of the above plots does MultiArmedBandit appear to receive higher
  rewards on average than QLearning? Provide an explanation for
  why that happens, based on your understanding of MultiArmedBandit.

- e. Following d.: in the environment(s) where QLearning was the **worse**
  model, is there any way you could change your choice of hyperparameters so
  that QLearning would perform as well as MultiArmedBandit?  Why or why not?

## 3. (2 total points) Exploration vs. Exploitation

- a. Look at the code in `free_response/q3.py` and  run `python -m
  free_response.q3` and include the plot it creates
  (`free_response/3a_g0.9_a0.2.png`) as your answer to this part. In your own
  words, describe what this code is doing.

- b. Using the above plot, describe what you notice. What seems to be the
  ``best'' value of epsilon? What explains this result?

- c. The above plot trains agents for 50,000 timesteps each. Suppose we instead
  trained them for 500,000 or 5,000,000 timesteps. How would you expect the
  trends to change or remain the same for each of the three values of epsilon?
  Give a one-sentence explanation for each value.

- d. When people use reinforcement learning in practice, it can be difficult to
  choose epsilon and other hyperparameters. Instead of trying three options
  like we did above, suppose we tried 30 or 300 different choices. What might
  be the danger of choosing epsilon this way if we wanted to use our agent in a
  new domain?


## 4. (4 total points) Fair ML in the real world

Read [Joy Buolamwini and Timnit Gebru, 2018. "Gender shades: Intersectional
accuracy disparities in commercial gender classification." Conference on
fairness, accountability and
transparency](http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf),
then use it to help answer the following questions.

- a. Buolamwini and Gebru use PPV and FPR as metrics to measure fairness. Find
  the definition of these in the paper, then look up the corresponding
  definition for NPV and FNR (these appear in the slides).  Assuming you were
  applying for a loan and you know a ML classifier is deciding whether to grant
  it to you: would you rather have that decision made by a system with a high
  FPR or a high FNR? Why? Provide a detailed justification.

- b. Assuming you were applying for a loan and you know a ML classifier is
  deciding whether to grant it to you: would you rather have that decision made
  by a system with a high PPV or a high NPV? Why? Provide a detailed
  justification.

- c. What recommendations do Buolamwini and Gebru make regarding accountability
  and transparency of ML systems? How does this relate to specific metrics such
  as PPV or FPR?

- d. What is *intersectional* about the analysis conducted by the authors? What
  does that analysis show?

- e. In Section 4.7, the authors say that their "findings ... do not appear to
  be confounded by the quality of sensor readings." What do they mean by
  "confounded" in this context? Why is it important to the thesis of this paper
  to check whether their findings are confounded?


## 5. (1 point) Fairness metrics, again

- a. We have talked about two fairness metrics: demographic parity (DP) and
  equalized odds (EO).  Using [this
  tutorial](https://www.youtube.com/watch?v=jIXIuYdnyyk) or [this Wikipedia
  article](https://en.wikipedia.org/wiki/Fairness_(machine_learning)), find a
  third fairness metric, define it in detail, and create an example where that
  definition might better capture "fairness" than either DP or EO.

- b. The [US Equal Credit Opportunity Act
  (ECOA)](https://www.justice.gov/crt/equal-credit-opportunity-act-3) forbids
  "credit discrimination on the basis of race, color, religion, national
  origin, sex, marital status, age, or whether you receive income from a public
  assistance program." Suppose a bank wants to use machine learning to decide
  whether to approve or reject a loan application, but also wants to comply
  with the ECOA.  What are __two__ additional challenges that might be
  introduced if your ML system needs to fulfill a given fairness metric with
  respect to eight protected attributes instead of just one?


## 6. (4 points) Reading reflections

Pick one article from each of two different groups (A, B, or C) below. For each
of the two articles you read, write the following:
- A two-sentence summary of the article. Why does this article matter?
- Pick at least two stages of the "machine learning lifecycle" that we
  discussed in lecture (from page 30 of [this
  tutorial](https://drive.google.com/file/d/1rUQkVS0NzSH3IEqZDsczSxBbhYHbjamN/view)),
  from Task Definition to Feedback. For each stage, consider an ethical
  question that is relevant to that stage in the development of the AI
  system(s) discussed in the article you read. For some of these readings,
  those AI system(s) may be hypothetical -- you can create your own examples as
  necessary, but provide as much detail as you can.
- Finally, share one question that this article raised for you that you'd like
  to learn more about.

### Group A: Privacy and Accountability
- https://www.cbsnews.com/news/unitedhealth-lawsuit-ai-deny-claims-medicare-advantage-health-insurance-denials/
- https://www.reuters.com/legal/legalindustry/privacy-paradox-with-ai-2023-10-31/
- https://www.publicbooks.org/the-danger-of-intimate-algorithms/
- https://spectrum.ieee.org/bionic-eye-obsolete

### Group B: Climate
- https://news.climate.columbia.edu/2023/06/09/ais-growing-carbon-footprint/
- https://arxiv.org/pdf/2104.10350.pdf
- https://oecd.ai/en/wonk/how-much-water-does-ai-consume

### Group C: Labor
- https://time.com/6247678/openai-chatgpt-kenya-workers/
- https://www.kqed.org/news/11971467/protesting-project-nimbus-what-rights-do-silicon-valley-employees-have
- https://apnews.com/article/hollywood-ai-strike-wga-artificial-intelligence-39ab72582c3a15f77510c9c30a45ffc8
- https://hbr.org/2017/01/the-humans-working-behind-the-ai-curtain
