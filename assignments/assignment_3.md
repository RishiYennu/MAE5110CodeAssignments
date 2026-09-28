# Assignment 3

**PR opening deadline:** Tuesday, September 29th, 11:59PM (midnight)  
**PR completion deadline:** Saturday, October 3rd, 11:59PM (midnight)  
**PR merging deadline:** Sunday, October 4th, 11:59PM (midnight)

This week's assignment is focused on software engineering.

**Goals:**

- Learn to use tests
- Build confidence with `git` and `uv`
- Become familiar with `Marimo` notebooks
- Practice explaining and reviewing changes to somebody else's code

## Peer Coding

For this assignment, instead of working on your own code and getting a peer
review, you will directly work on somebody else's code. First, look up your
assigned partner. Make sure their `main` contains their completed earlier
assignments (ask them to complete it ASAP if needed). You'll bring in the
current assignment yourself. Add their fork as a remote named `A3partner`:

```console
git remote add A3partner <partner-fork-url>
git fetch A3partner
```

As before, `origin` is your fork and `upstream` is the class repo. Your PR will
go from your assignment branch on your fork into `main` on their fork.

### Bring in Assignment 3 with a rebase

Create your assignment branch from the latest class starter code, then replay
its new commits on top of your partner's completed work:

```console
git fetch upstream
git switch -c my_username/assignment_3 upstream/main
git rebase A3partner/main
```

A **merge** joins diverged histories with a merge commit, preserving existing
commits. A **rebase** replays commits on a new base: here, the new class commits
are replayed onto `A3partner/main`. Your partner's commits keep their hashes;
the replayed class commits get new hashes on your assignment branch. The PR will
then contain the class updates and your new work, with your partner's history
intact. See
[Atlassian's Merging vs. Rebasing tutorial](https://www.atlassian.com/git/tutorials/merging-vs-rebasing).

If conflicts occur, resolve the affected files, use `git add <resolved-file>`,
then `git rebase --continue`. Use `git rebase --abort` to return to where you
started if you need to try again.

### Unit testing with pytest

Unit tests make sure you can develop with confidence, by checking that your code
behaves as expected. A standard unit test will run some of your code, and use
`assert` to check that something you expect to be true indeed is. You can
imagine, it's possible to come up with any number of tests that are trivial and
don't actually do very much. Typical things to test are signatures and
_interfaces_ between different parts of your implementation, so that as you add
functionality, new code doesn't break how it interacts with previous
implementations.

There are several packages for unit-testing; we will be using
[pytest](https://docs.pytest.org/en/stable/). Take a look in the `tests/`
folder, and in particular `tests/test_models.py`. You can run this with
`uv run pytest tests/test_models.py`, or run `uv run pytest tests` to run all
Python files that start with `test_` in that folder.

Run all the tests now and read through any failures to understand what is
required to be updated from your partner's model implementations.

Push your assignment branch and open your draft PR:

```console
git push -u origin my_username/assignment_3
```

On GitHub, choose your partner's fork and `main` as the PR's base, and your fork
and `my_username/assignment_3` as its head. Share the draft PR link with your
partner. If the class updates were already present and there is nothing to
compare, make, commit, and push your first assignment change before opening the
PR.

Fix the implementations to satisfy the provided tests.

### Creating your own sanity check

Now implement a sanity check for the pendulum: with 0 damping and no torque, the
energy should remain constant. Implement this as a unit test in a new
`tests/test_pendulum.py`: initialize the environment, make sure torque and
damping are set to zero in the parameter dict, simulate a hundred steps (make
sure _not_ to start at the equilibrium), and assert that the total energy change
across steps stays close to zero (use `np.isclose()`).

What sanity check can you implement for testing that the torque and the damping
are implemented correctly? Implement a test for each of these.

## Marimo notebooks

We'll be using [Marimo](https://marimo.io/) notebooks, and their server
[MoLab](https://molab.marimo.io/notebooks) for BigRedGym. Let's get familiar
with it.

### 1. Explore the annotated compass-gait example

Run these commands from the repository root. First, run the example as a plain
Python script and look at its plots and animation:

```console
uv run scripts/example_compass_gait.py
```

Then read `scripts/example_compass_gait.py`; it is already annotated for
conversion: `# %%` starts a code cell, and `# %% [markdown]` starts a markdown
cell whose text is written as Python comments. Convert it and explore the
notebook:

```console
uv run --group notebooks marimo convert scripts/example_compass_gait.py -o example_compass_gait_notebook.py
uv run --group notebooks marimo edit example_compass_gait_notebook.py
uv run --group notebooks marimo run example_compass_gait_notebook.py
```

Running `marimo edit` lets you develop interactively; `marimo run` displays a
read-only notebook. Compare its cells, plots, and animation with the original
script.

### 2. Annotate the value-iteration example

Start by running this example as a plain Python script too:

```console
uv run scripts/example_value_iteration.py
```

Using the compass-gait script as a guide, add `# %%` cell boundaries to
`scripts/example_value_iteration.py` and a short `# %% [markdown]` introduction
explaining what it does. Choose useful cells for the setup, value iteration,
simulation, and visualization. Keep the file runnable as a plain Python script.

Then convert your annotated script and check that the notebook reproduces its
plots and animation:

```console
uv run --group notebooks marimo convert scripts/example_value_iteration.py -o example_value_iteration_notebook.py
uv run --group notebooks marimo edit example_value_iteration_notebook.py
```

These annotations use [Jupytext](https://jupytext.org/). Keep the raw script as
the source and regenerate the notebook after editing it. Commit your annotated
value-iteration script and its generated notebook.

### UV and optional dependencies

Some tools are useful for particular tasks but aren't needed every time we work
on a project. In [`pyproject.toml`](../pyproject.toml), we put the notebook
tools in a dependency group named `notebooks`:

```toml
[dependency-groups]
notebooks = [
    "marimo>=0.25",
    "jupytext",
]
```

Marimo provides the notebook editor; Jupytext lets its converter read the `# %%`
cell markers in our plain Python script. Keeping these dependencies in an
optional group keeps ordinary development light: our usual `uv sync` setup
includes the main dependencies and the `dev` group (which contains pytest), but
leaves out `notebooks`.

Read these commands in two parts:

- `uv run --group notebooks` prepares the project environment, including the
  packages in the `notebooks` group alongside the usual project dependencies.
  `--group` is an option for uv, and `notebooks` is the name we chose in
  `pyproject.toml`.
- `marimo convert` generates the notebook file; `marimo edit` opens the notebook
  editor. Make shared changes in the raw Python script, then convert again.
  Conversion overwrites the generated notebook, and notebook edits don't sync
  back to the raw script.

Anyone running the scripts without wanting to run notebooks won't have to
install these dependencies.

You may also encounter `uvx`, which is shorthand for running a tool in an
isolated environment. For example, `uvx --with jupytext marimo convert ...`
supplies Marimo and Jupytext for conversion. It does _not_ automatically include
the project's dependencies.

We use `uv run` here so the notebook and the assignment scripts use the same
NumPy and Matplotlib environment. See uv's documentation on
[dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups)
and [running tools](https://docs.astral.sh/uv/guides/tools/) for more detail.

**Optional simplification for the rest of the class:** if you expect to use
notebooks regularly, consider moving `"marimo>=0.25"` and `"jupytext"` into the
default list of `dependencies`. Run `uv sync` to update your environment. You
can then leave out `--group notebooks` in later commands.

## Interactive rebase: tidy your new work

The first rebase brought in the class updates. The purpose was largely to avoid
cluttering the git commit history with merge commits.

**Interactive rebase** lets you choose how to replay commits, including editing
their messages or combining related changes. Here, we'll keep the same base and
tidy only your new, unpushed work.

Before pushing to Github, commit your changes and check that `git status` is
clean. Still on your assignment branch, inspect the commits since your initial
push:

```console
git log --oneline --reverse origin/my_username/assignment_3..HEAD
git rebase -i origin/my_username/assignment_3
```

`origin/my_username/assignment_3` marks your last push, so uploaded history
stays outside the cleanup. The editor lists your new commits oldest first. For
example:

```text
pick a1b2c3d model fixes
pick b2c3d4e Add a missing default parameter
pick c3d4e5f Add annotated notebook example
pick d4e5f6a Explain notebook cell choices
```

You can change `pick` to other actions, such as `reword`, `fixup`, and `squash`,
e.g.:

```text
reword a1b2c3d model fixes
fixup b2c3d4e Add a missing default parameter
pick c3d4e5f Add annotated notebook example
squash d4e5f6a Explain notebook cell choices
```

- `reword` changes a commit message, e.g. to `Make model interfaces consistent`.
- `fixup` combines a correction with the preceding commit, keeping the preceding
  commit's message.
- `pick` keeps a commit as a separate change.
- `squash` combines a commit with the preceding one and opens an editor with
  both messages so you can write one message describing the combined change.

Save and close the editor, then edit the messages when Git asks. See
[Atlassian's rewriting-history tutorial](https://www.atlassian.com/git/tutorials/rewriting-history/)
for more examples.

For your exercise, use **reword**, **fixup**, and **squash** at least once each.
Combine consecutive, related commits; keep unrelated work separate. There is no
required final number of commits. If you don't have enough commits, change a
couple of parameters, and commit those changes, just to go through the motions.

Check the result:

```console
git log --oneline --reverse origin/my_username/assignment_3..HEAD
uv run pytest tests
```

### Merging the PR

Mark your PR ready for review and address feedback. Review the incoming PR and
use
[**Squash and merge**](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/merging-a-pull-request)
into **your own fork's `main`** by the completion deadline. This combines the
whole PR into one commit. Set its title to exactly `Complete assignment 3`.

## Deliverables

You will be graded on the implementation in PR you opened for the implementation
details, but you are responsible for the final step approving the merge for your
own repo. We will check your fork's `main` automatically: the PR must be merged,
the final commit title must match, and all unit tests must pass.
