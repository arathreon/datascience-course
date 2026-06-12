# Module 1 — Describing One Variable

*From Zero to Competent in Data Science · Phase 1: Foundations*

You know calculus and you can program. So I'm not going to teach you what a "for loop" is or spend a paragraph telling you a derivative measures a rate of change. I'm going to use the calculus you already have to show you *why* the basic statistical quantities are defined the way they are — because almost everyone learns these as formulas to memorize, and that's exactly why almost everyone misunderstands them.

This module covers one thing: how to describe a single column of numbers. Mean, variance, standard deviation. It sounds trivial. It is not. The squared-error idea you meet here is the same idea that, three modules from now, becomes linear regression, and five modules from now becomes the loss function of a neural network. Get the mechanism here and the rest of the curriculum is built on rock.

---

## 0. The one mental model everything rests on

Here is the single most important idea in all of statistics, and it is conceptual, not mathematical.

**Your data is not the thing you care about. Your data is a sample drawn from the thing you care about.**

You spent time in a lab measuring boiling points, yields, concentrations. Suppose you measure the boiling point of a solvent ten times and get ten slightly different numbers — thermometer noise, timing, the operator. There is a *true* boiling point under the experimental conditions. You never observe it. You observe ten noisy draws around it.

Statistics gives those two things separate names, and you must keep them separate for the rest of your life:

- The **population** (or **data-generating process**): the true, usually-unknown thing. The true boiling point. The true distribution that produces measurements. Its properties are called **parameters** — written with Greek letters: $\mu$ (mu) for the true mean, $\sigma$ (sigma) for the true standard deviation.
- The **sample**: the finite set of numbers you actually have. Your ten readings. Quantities you compute *from the sample* are called **statistics** or **estimates** — written with Latin letters or hats: $\bar{x}$ (x-bar) for the sample mean, $s$ for the sample standard deviation.

Everything you will ever do in data science is some version of this move: *use the sample (which you have) to make a defensible statement about the population (which you don't)*. When you forget the distinction, you start believing your data is the truth instead of a noisy, finite window onto the truth. That single confusion is responsible for a large fraction of bad data analysis in the world.

Hold that. Now the mathematics.

---

## 1. The mean, understood three ways

The first reading gives you the formula. The third reading gives you the reason squared error runs through all of machine learning.

### 1a. The mean as an average

Given $n$ numbers $x_1, x_2, \dots, x_n$, the **arithmetic mean** is

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.$$

You knew this. Add them up, divide by how many. Move on.

### 1b. The mean as a balance point

Picture your data points as equal masses placed on a number line. The mean is the position of the fulcrum where the line balances — the **center of mass**. This isn't a metaphor; it's literally the same formula. Center of mass is $\frac{\sum m_i x_i}{\sum m_i}$, and with equal masses ($m_i = 1$) that collapses to $\frac{1}{n}\sum x_i$.

Two consequences fall straight out of the physics, and both matter:

The deviations from the mean cancel. If the mean is the balance point, the "torques" on either side sum to zero:

$$\sum_{i=1}^{n}(x_i - \bar{x}) = 0.$$

This is not an empirical observation, it's an identity. Proof, one line: $\sum(x_i - \bar{x}) = \sum x_i - n\bar{x} = n\bar{x} - n\bar{x} = 0$, using $\sum x_i = n\bar{x}$ directly from the definition. Remember this fact — it is the reason behind the $n-1$ in the variance, which trips up nearly everyone.

A single far-away point drags the fulcrum. Put one mass very far to the right and the balance point shifts toward it. The mean is **not robust to outliers**. One data-entry error of 10,000 where you meant 10.000 will visibly move it. (The median — the middle value — barely moves, because moving a far point doesn't change which value sits in the middle. That's the whole robustness difference between the two, and it has a precise mathematical source, which is next.)

### 1c. The mean as the minimizer of squared error — the one that matters

Here is the reading that is almost never taught and that quietly governs the rest of the field.

Ask a different question. Forget "what's the average." Ask: *if I had to summarize this whole dataset with a single number $c$, and I'm penalized by the **square** of how far each data point is from my guess, what single number minimizes my total penalty?*

Total penalty as a function of your guess $c$:

$$f(c) = \sum_{i=1}^{n}(x_i - c)^2.$$

This is a function of one variable $c$. You know exactly how to minimize it. Differentiate with respect to $c$, set to zero. Using the chain rule on each term:

$$f'(c) = \sum_{i=1}^{n} 2(x_i - c)\cdot(-1) = -2\sum_{i=1}^{n}(x_i - c).$$

Set $f'(c) = 0$:

$$\sum_{i=1}^{n}(x_i - c) = 0 \;\Longrightarrow\; \sum x_i - nc = 0 \;\Longrightarrow\; c = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}.$$

And it's genuinely a minimum, not a maximum or saddle: $f''(c) = 2n > 0$ everywhere, so $f$ is convex and this stationary point is the global minimum.

Read what just happened. **The mean is not an arbitrary definition. It is the answer to a question: it is the single number that is "closest" to all your data when closeness is measured by squared distance.** Average and least-squares-summary are the *same object* viewed from two sides.

Now the payoff, because this generalizes immediately. Repeat the exercise with **absolute** distance instead of squared:

$$g(c) = \sum_{i=1}^{n}|x_i - c|.$$

Minimizing this gives the **median**, not the mean. (Sketch of why: the derivative of $|x_i - c|$ with respect to $c$ is $-1$ when $c < x_i$ and $+1$ when $c > x_i$. The total derivative is (number of points above $c$) minus (number below). That's zero when $c$ splits the data in half — the median. One honest caveat: for an even number of points, *any* value between the two central observations gives the same minimum, which is exactly why the median of an even-sized sample is conventionally defined as the midpoint of those two.)

So:

$$\text{squared error} \;\longleftrightarrow\; \text{mean}, \qquad \text{absolute error} \;\longleftrightarrow\; \text{median}.$$

This is why the mean is sensitive to outliers and the median is not, stated precisely: squaring makes a far point's penalty grow quadratically, so the squared-error minimizer (the mean) lunges to appease it; absolute error grows only linearly, so the absolute-error minimizer (the median) ignores how far the outlier is and only cares how many points sit on each side.

Why you should care beyond this module: when you fit a regression line by "least squares" in Module 4, you are doing exactly $\min_c \sum (x_i - c)^2$ with $c$ replaced by a line. When a model trains on "mean squared error" loss, it is being pushed toward the conditional mean of the target. The choice of squared error everywhere in ML is not a coincidence or a mere convenience — it is a choice to estimate means, with all the efficiency and all the outlier-fragility that implies. You now understand the foundation of that choice.

---

## 2. Spread: variance and standard deviation

The mean tells you where the data sits. It tells you nothing about how tightly it clusters. These two samples have the **identical** mean of 10:

```
A: 10, 10, 10, 10, 10
B:  2, 6, 10, 14, 18
```

A is a precise instrument; B is junk. Same center, completely different behavior. You need a number for the spread.

### 2a. Defining variance

Natural first instinct: "average distance from the mean." But you already saw the signed deviations sum to zero ($\sum(x_i - \bar{x}) = 0$) — positives and negatives cancel exactly, so the average signed deviation is always 0 and tells you nothing. You must kill the signs first. Two ways to kill a sign: absolute value, or squaring. Statistics overwhelmingly squares, and defines the **population variance** as the average *squared* deviation from the mean:

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2.$$

Four reasons it's squared, not absolute — and the last is the deep one:

1. **Differentiability.** $x^2$ is smooth everywhere; $|x|$ has a corner at 0 where its derivative doesn't exist. Every optimization you'll do — fitting models, minimizing loss — wants to take derivatives. Squared deviations are analytically friendly; absolute deviations fight you at zero.
2. **It connects to the mean.** From Section 1c, the squared-deviation world has the mean as its natural center. Variance is the *value* of that minimized squared-error sum (divided by $n$). Variance and mean are the same mathematical family. Absolute deviations would pair with the median instead, fracturing the framework.
3. **It punishes large deviations harder.** A deviation of 4 contributes 16; a deviation of 2 contributes 4. Squaring weights big misses far more than small ones. Sometimes that's what you want (large errors are disproportionately bad), sometimes it's the source of outlier-sensitivity — but it's a deliberate, understood property.
4. **Variances add for independent quantities.** This is the one that actually decides it. If $X$ and $Y$ are independent, $\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$. Absolute deviations have no such clean rule. This additivity is the machinery behind error propagation in your lab reports, behind why the uncertainty of an average shrinks like $1/\sqrt{n}$, and behind the Central Limit Theorem we hit in Module 2. You square because squared spread is the kind of spread that *combines predictably*. Keep this flagged; it pays off enormously soon.

### 2b. Standard deviation, and why we take the root

Variance has a problem: its units are wrong. If your data is in degrees Celsius, the deviations $(x_i - \mu)$ are in °C, and squaring them gives °C². A variance of "1.9 squared-degrees" is dimensionally meaningless to a human — and you, of all people, were trained to take units seriously.

Fix it by taking the square root, recovering the original units. That's the **standard deviation**:

$$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}.$$

Now you have a number in °C that you can read as "the typical distance of a measurement from the mean." That interpretability is the *only* reason standard deviation exists as a separate quantity — it's just the square root of variance, dragged back into the data's own units so a person can reason about it. When you do algebra and proofs you'll usually work with variance (no roots to carry); when you report a result to a human you'll usually quote standard deviation (right units). Same information, two presentations.

---

## 3. The $n$ versus $n-1$ trap (Bessel's correction)

This is where the population/sample distinction from Section 0 stops being philosophy and starts changing your arithmetic — and where a real bug hides in everyone's code.

Look again at the variance formula:

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2.$$

It uses $\mu$, the **true** population mean. But in practice you never have $\mu$. You only have your sample, so you do the obvious thing and substitute the sample mean $\bar{x}$:

$$\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2.$$

This estimator is **biased**. On average, across many samples, it comes out *too small* — it systematically underestimates the true variance. Here is the mechanism, and it follows directly from something you already proved.

In Section 1c you proved $\bar{x}$ is the number that *minimizes* $\sum(x_i - c)^2$. So for any other value of $c$ — including the true mean $\mu$:

$$\sum_{i=1}^{n}(x_i - \bar{x})^2 \;\le\; \sum_{i=1}^{n}(x_i - \mu)^2.$$

The sum of squared deviations measured from the *sample* mean is the smallest it can possibly be. Your sample mean is, by construction, hugging your particular sample more tightly than the true mean does. So squared deviations from $\bar{x}$ are, on average, smaller than squared deviations from $\mu$ — and dividing those too-small sums by $n$ gives you a variance estimate that's too small.

The exact size of the shrinkage turns out to be a factor of $\frac{n-1}{n}$: on average the divide-by-$n$ estimator returns $\frac{n-1}{n}\sigma^2$ instead of $\sigma^2$. (The full algebraic proof needs the variance of the sample mean, which we derive cleanly in Module 2. For now you'll *watch* a simulation hit this exact ratio, which for a hands-on learner is more convincing than the algebra anyway.) The repair is to cancel that factor by dividing by $n-1$ instead of $n$, defining the **sample variance**:

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2.$$

This $s^2$ is **unbiased**: its average over many samples equals $\sigma^2$ exactly.

**Degrees of freedom — why $n-1$ specifically.** Recall the identity $\sum(x_i - \bar{x}) = 0$. Once you've computed $\bar{x}$ from the data, the $n$ deviations are forced to sum to zero — that's one linear constraint binding them. So only $n-1$ of the deviations are free to vary; fix any $n-1$ of them and the last is determined. You "spent" one degree of freedom the moment you estimated the mean from the same data you're now using for the variance. Dividing by the number of *free* pieces of information, $n-1$, rather than the raw count $n$, is what makes the estimate honest. This "lose one degree of freedom per parameter estimated" idea recurs everywhere — in regression you'll divide by $n - p$ for $p$ fitted coefficients, same logic.

**The practical trap, stated bluntly:** the two most common Python libraries disagree on the default.

- **NumPy** `np.var` and `np.std` default to dividing by $n$ (`ddof=0`) — the biased population formula.
- **pandas** `Series.var` and `Series.std` default to dividing by $n-1$ (`ddof=1`) — the unbiased sample formula.

`ddof` means "delta degrees of freedom": the divisor is $n - \text{ddof}$. So `ddof=0` → divide by $n$, `ddof=1` → divide by $n-1$. The same data run through NumPy and pandas with default settings gives *different variances*, and the gap is biggest exactly when you have little data (small $n$) and can least afford to be sloppy. When you have a genuine sample from a larger population — which is almost always — you want `ddof=1`. Set it explicitly in NumPy and don't rely on remembering which default is which.

---

## 4. Hands-on: build it from scratch, then prove the theory with code

Now you run things. The philosophy for the whole tutorial, which I'm stealing from Allen Downey's *Think Stats*: you already know how to program, so use that skill to *make the statistics concrete* — turn integrals into sums, turn theorems into simulations you can watch converge. Below, everything from Sections 1–3 becomes runnable.

Install nothing exotic; you need `numpy`. (`pip install numpy` if it isn't already there.)

### 4a. The quantities from scratch, checked against NumPy

```python
import numpy as np

# Ten "measurements" — pretend these are repeated readings of one quantity.
x = np.array([12.1, 9.8, 11.5, 10.2, 13.0, 8.7, 10.9, 12.4, 9.1, 11.8])
n = x.size

# --- mean, by hand ---
mean_scratch = x.sum() / n

# --- variance, by hand, both conventions ---
deviations = x - mean_scratch          # this is x_i - x̄ for every i at once
sq_dev = deviations ** 2               # squared deviations
var_pop  = sq_dev.sum() / n            # ÷ n      → biased / "population"   (ddof=0)
var_samp = sq_dev.sum() / (n - 1)      # ÷ (n-1)  → unbiased / "sample"     (ddof=1)

# --- standard deviation = root of variance ---
std_samp = np.sqrt(var_samp)

print(f"mean        : {mean_scratch:.4f}   (numpy: {np.mean(x):.4f})")
print(f"var  ddof=0  : {var_pop:.4f}   (numpy: {np.var(x):.4f})")
print(f"var  ddof=1  : {var_samp:.4f}   (numpy: {np.var(x, ddof=1):.4f})")
print(f"std  ddof=1  : {std_samp:.4f}   (numpy: {np.std(x, ddof=1):.4f})")
```

Expected output:

```
mean        : 10.9500   (numpy: 10.9500)
var  ddof=0  : 1.9025   (numpy: 1.9025)
var  ddof=1  : 2.1139   (numpy: 2.1139)
std  ddof=1  : 1.4539   (numpy: 1.4539)
```

**The HOW — what the code is mechanically doing, and why this style.** The line `deviations = x - mean_scratch` is the part to actually understand, because it's the workhorse pattern of the entire NumPy ecosystem:

- `x` is a NumPy array of 10 numbers; `mean_scratch` is a single scalar. Subtracting them does *not* require a loop. NumPy **broadcasts** the scalar: it stretches the single mean value to match the shape of `x` and subtracts it from every element, producing a new 10-element array of deviations in one operation. Likewise `deviations ** 2` squares all ten at once, and `.sum()` adds them in one call.
- This is **vectorization**, and it's why we don't write a Python `for` loop here. The loop version is not just uglier — it's typically 10–100× slower on real data. The reason is concrete: a Python-level loop interprets bytecode and does pointer-chasing object arithmetic on every iteration, whereas `x - mean_scratch` dispatches *once* into a precompiled C routine that runs over a tight, contiguous block of doubles in memory with no per-element Python overhead. As your datasets grow from 10 rows to 10 million, "write it as array operations, never as element loops" stops being style advice and becomes the difference between a query that returns and one that doesn't. You'll feel this hard in the big-data phase.
- The `ddof` argument is the lesson from Section 3 made literal: `np.var(x)` divides by $n$, `np.var(x, ddof=1)` divides by $n-1$, and the two printed numbers (1.9025 vs 2.1139) differ by exactly the factor $\frac{n}{n-1} = \frac{10}{9}$. Check it: $1.9025 \times \frac{10}{9} = 2.1139$. The theory and the floating-point output agree to the digit.

### 4b. Watch the mean fall out of least-squares

Section 1c claimed the mean is the value of $c$ minimizing $\sum(x_i - c)^2$. Don't take it on faith — sweep $c$ across a fine grid, compute the total squared error at each candidate, and find where it bottoms out.

```python
import numpy as np

x = np.array([12.1, 9.8, 11.5, 10.2, 13.0, 8.7, 10.9, 12.4, 9.1, 11.8])

# A dense grid of candidate "single-number summaries" c, spanning the data range.
candidates = np.linspace(x.min(), x.max(), 100_001)

# For each candidate c, the total squared error SSE(c) = Σ (x_i − c)^2.
# x[:, None] reshapes x to a column (10×1); candidates[None, :] is a row (1×100001).
# Subtracting broadcasts them into a full 10×100001 grid of (x_i − c) values;
# we square, then sum down the rows (axis=0) to collapse to one SSE per candidate.
sse = ((x[:, None] - candidates[None, :]) ** 2).sum(axis=0)

c_star = candidates[np.argmin(sse)]    # the c with the smallest SSE
print(f"c that minimizes SSE : {c_star:.4f}")
print(f"the sample mean      : {x.mean():.4f}")
```

Output:

```
c that minimizes SSE : 10.9500
the sample mean      : 10.9500
```

The minimizer of the squared-error curve *is* the mean, to four decimals. The calculus from Section 1c, confirmed by brute force.

**The HOW — the 2D broadcasting trick.** This snippet does something worth dissecting because it's a pattern you'll reuse constantly. `x[:, None]` turns the 10-element vector into a 10×1 column; `candidates[None, :]` turns the 100,001-element vector into a 1×100,001 row. When NumPy subtracts a 10×1 from a 1×100,001, broadcasting expands *both* to 10×100,001 — giving you every combination $(x_i - c_j)$ in a single array, no nested loop over $i$ and $j$. Summing over `axis=0` collapses the 10 data points for each candidate, leaving one SSE value per candidate. You computed a million-plus squared deviations and a hundred-thousand sums with two lines and zero Python loops. That is the NumPy way of thinking: *express the whole computation as operations on arrays whose shapes line up, and let the C layer do the iteration.*

### 4c. Watch Bessel's correction be necessary

This is the centerpiece. Section 3 claimed dividing by $n$ underestimates the true variance by a factor of $\frac{n-1}{n}$, while dividing by $n-1$ is unbiased. We can *prove it empirically*: invent a population with a variance we know exactly, draw a huge number of small samples from it, compute both estimators on each sample, and average. The unbiased one should land on the truth; the biased one should land on $\frac{n-1}{n}$ of the truth.

```python
import numpy as np

rng = np.random.default_rng(42)   # seeded generator → reproducible results

# A population we control: Normal with known mean 5 and known std 2,
# so the TRUE variance is 2^2 = 4. This is the number we're trying to recover.
true_sigma = 2.0
true_var   = true_sigma ** 2      # = 4.0

sample_size   = 5                 # small n exaggerates the bias, making it visible
n_experiments = 200_000           # many repeated samples → averages converge

# Draw 200,000 independent samples, each of size 5, as one 200000×5 array.
samples = rng.normal(loc=5.0, scale=true_sigma, size=(n_experiments, sample_size))

# Compute BOTH variance estimators on every row (each sample), then average
# those estimates across all 200,000 experiments.
avg_var_ddof0 = samples.var(axis=1, ddof=0).mean()   # divide by n
avg_var_ddof1 = samples.var(axis=1, ddof=1).mean()   # divide by n-1

print(f"true population variance        : {true_var:.4f}")
print(f"avg of  ÷n   (ddof=0) estimator : {avg_var_ddof0:.4f}"
      f"   → {avg_var_ddof0/true_var:.4f} of the truth")
print(f"avg of ÷(n-1)(ddof=1) estimator : {avg_var_ddof1:.4f}"
      f"   → {avg_var_ddof1/true_var:.4f} of the truth")
print(f"theory says ÷n undershoots to (n-1)/n = {(sample_size-1)/sample_size:.4f}")
```

Output:

```
true population variance        : 4.0000
avg of  ÷n   (ddof=0) estimator : 3.2054   → 0.8013 of the truth
avg of ÷(n-1)(ddof=1) estimator : 4.0067   → 1.0017 of the truth
```

Stop and appreciate this, because it's the whole module in four numbers. The divide-by-$n$ estimator averages to 3.21 — it recovers **80%** of the true variance of 4. Theory predicted exactly $\frac{n-1}{n} = \frac{4}{5} = 0.80$, and the simulation delivered 0.8013. The divide-by-$(n-1)$ estimator averages to 4.007 — bang on the true 4. The bias is not a rounding artifact or a philosophical nicety; it is a real, predictable, *measurable* distortion, and the $n-1$ divisor is the exact thing that removes it.

**The HOW — why this experiment is valid, and the design choices.** A few deliberate decisions make the demonstration airtight:

- `rng = np.random.default_rng(42)` creates a seeded random generator. The seed makes the run reproducible — you get the same "random" numbers I did, which is what you want for a demonstration. (Use `default_rng()`, the modern NumPy generator, not the legacy `np.random.seed` / `np.random.normal` global-state interface; the modern one is the current recommended API and avoids a class of hard-to-debug global-state bugs.)
- We **chose** the population (`Normal(5, 2)`), so we know the ground-truth variance is exactly 4. You can't measure bias against an unknown truth, so we manufacture a known truth. This is the core trick of testing a statistical method: simulate from a process you fully control, then check whether the method recovers what you put in.
- `sample_size = 5` is small on purpose. The bias factor $\frac{n-1}{n}$ is $\frac{4}{5}=0.8$ at $n=5$ but $\frac{99}{100}=0.99$ at $n=100$ — at large $n$ the bias is real but nearly invisible. Small $n$ makes it jump out. This also tells you *when to care*: the correction matters most with little data.
- `n_experiments = 200_000` is the "watch it converge" lever. Each individual sample's variance estimate is noisy; we're interested in the *average* behavior of the estimator across many samples (its expected value). The Law of Large Numbers — which you'll meet formally in Module 2 — guarantees that averaging over enough experiments makes that empirical average converge to the true expected value. 200,000 is plenty to pin the ratios to three decimals.
- `samples.var(axis=1, ...)` computes the variance *along each row* (`axis=1` = across the 5 columns of one sample), giving 200,000 separate estimates; `.mean()` then averages them. Again: zero Python loops over the experiments. The entire Monte Carlo study is three array operations.

You just ran your first **Monte Carlo simulation** — estimating a property of a process by simulating it many times and averaging. It's one of the most useful tools you'll own, and we'll lean on it constantly to *check* analytical claims and to compute things that are too hard to derive in closed form.

---

## 5. Exercises

Do these in a script or notebook. The point is not to get an answer to compare — the point is that writing the code forces you to confront whether you actually understood the mechanism. Where a question says "predict before you run," genuinely write the prediction down first; the gap between your prediction and the output is where the learning is.

1. **Reproduce and break the mean's fragility.** Take `x = np.array([10, 11, 9, 10, 12, 8, 11])`. Compute its mean and median. Now change the last value from `11` to `1100` (a fat-finger error). Recompute both. Report how far each moved, and explain *in terms of Section 1c* — squared vs. absolute error — why one barely budged and the other lurched.

2. **Verify the zero-sum identity numerically.** For any array you like, compute `(x - x.mean()).sum()`. You should get something microscopically close to 0 but not *exactly* 0 (e.g. `1e-15`). Explain why it isn't exactly zero — this is about floating-point arithmetic, not about the math being wrong — and connect it back to why that identity is the reason variance loses one degree of freedom.

3. **Redo the least-squares sweep for the median.** Adapt the code in 4b, but minimize the sum of *absolute* deviations $\sum|x_i - c|$ instead of squared. Confirm the minimizer matches `np.median(x)`. Then use an **even-length** array and an **odd-length** array and explain the difference you see at the minimum, using the caveat from Section 1c.

4. **Find the $n$ where the bias stops mattering.** Modify the Monte Carlo in 4c into a loop (or vectorized sweep) over `sample_size` in `[2, 5, 10, 30, 100, 1000]`. For each, report the ratio (avg of `ddof=0` estimator) / (true variance). Predict the ratios *before running* using $\frac{n-1}{n}$, then confirm. State the practical rule of thumb you'd draw from the result about when the $n$ vs $n-1$ choice is worth worrying about.

5. **(Stretch — foreshadows Module 2.)** Reuse the simulation machinery but now study the *sample mean* instead of the variance. Draw 200,000 samples of size $n$ from `Normal(5, 2)`, compute each sample's mean, and then compute the **standard deviation of those 200,000 sample means**. Do this for $n \in \{1, 4, 16, 64\}$. You should find the spread of the sample means shrinks as $n$ grows. Predict the relationship (hint: it involves $\sqrt{n}$ and ties directly to the "variances add" property from Section 2a, reason 4), and check your prediction against the numbers. This quantity is the **standard error of the mean**, and it's the entire reason more data gives you more confidence — it's the gateway to inference.

---

## What you now own, and where it goes

You can describe any single variable and, more importantly, you know *why* each descriptor is built the way it is: the mean is the least-squares center, the median is the least-absolute-error center, variance is squared spread (chosen because it's differentiable and additive), standard deviation is variance dragged back into real units, and the $n-1$ divisor is the price of having estimated the mean from the same data.

The thread to pull next is the one Exercise 5 dangles: the sample mean is *itself* a random quantity with its own spread, that spread shrinks like $1/\sqrt{n}$, and as $n$ grows the distribution of the sample mean becomes Normal *regardless of what you sampled from*. That's the **Central Limit Theorem**, and it's the hinge the entire machinery of statistical inference — confidence intervals, hypothesis tests, p-values, the works — turns on. That's Module 2.

When you've worked the exercises and the Bessel simulation makes intuitive sense — not just "the code ran" but "I see why the biased one had to come out low" — tell me and we'll build the CLT from the ground up. If anything here is shaky, say which part and I'll go deeper before we move.
