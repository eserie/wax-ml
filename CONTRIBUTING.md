Contributing to Wax
===================

You can contribute to WAX by asking questions, proposing practical use cases
or by contributing to the code.
You can have a look at our
[Developer Documentation](https://wax-ml.readthedocs.io/en/latest/developer.html).

`WAX` contributing guidelines are a fork of
[Astropy's CONTRIBUTING](https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md)
that we have adapted to `WAX` requirements.

Reporting Issues
----------------

When opening an issue to report a problem, please try to provide a minimal code
example that reproduces the issue along with details of the operating
system and the Python, NumPy, and `WAX` versions you are using.

Contributing
------------

So you are interested in contributing code to the `WAX` Project? Excellent!
We love contributions! `WAX` is open source, built on open source,
and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
[Adrienne Lowe](https://github.com/adriennefriend) for a
[PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted by
Astropy based on its use in the README file for the
[MetPy project](https://github.com/Unidata/MetPy).

### How to Contribute, Best Practices

All contributions to `WAX` are done via [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) from GitHub users'
"forks" (i.e., copies) of the [WAX repository](https://github.com/eserie/wax-ml).

Once you open a pull request (which should be opened against the ``main``
branch, not against any of the other branches), please make sure to
include the following:

- **Code**: the code you are adding, which should follow Astropy's
  [coding guidelines](https://docs.astropy.org/en/latest/development/codeguide.html) as much as possible.

- **Tests**: these are usually tests to ensure code that previously
  failed now works (regression tests), or tests that cover as much as possible
  of the new functionality to make sure it does not break in the future and
  also returns consistent results on all platforms (since we run these tests on
  many platforms/configurations). For more information about how to write
  tests, see our [testing guidelines](https://docs.astropy.org/en/latest/development/testguide.html).

- **Documentation**: `WAX` does not yet have Documentation.

- **Changelog entry**: `WAX` do not yet have Changelog.


Other Tips
----------

- Behind the scenes, we conduct a number of tests or checks with new pull requests.
  This is a technique that is called continuous integration, and we use GitHub Actions
  and CircleCI. To prevent the automated tests from running, you can add ``[ci skip]``
  to your commit message. This is useful if your PR is a work in progress (WIP) and
  you are not yet ready for the tests to run. For example:

      $ git commit -m "WIP widget [ci skip]"

  - If you already made the commit without including this string, you can edit
    your existing commit message by running:

        $ git commit --amend

- Unfortunately, GitHub Actions ignores ``[ci skip]`` for a PR, so we recommend
  you only push your commits to GitHub when you are ready for the CI to run.
  Please do not push a lot of commits for every small WIP changes.

- If your commit makes substantial changes to the documentation but none of
  those changes include code snippets, then you can use ``[ci skip]``,
  which will skip all CI except RTD, where the documentation is built.

- When contributing trivial documentation fixes (i.e., fixes to typos, spelling,
  grammar) that don't contain any special markup and are not associated with
  code changes, please include the string ``[ci skip]`` in your commit
  message.

      $ git commit -m "Fixed typo [ci skip]"

- ``[ci skip]`` and ``[skip ci]`` are the same and can be used interchangeably.

Checklist for Contributed Code
------------------------------

A pull request for a new feature will be reviewed to see if it meets the
following requirements. For any pull request, a `WAX` maintainer can help
to make sure that the pull request meets the requirements for inclusion in the
package.

**Scientific Quality** (when applicable)
  * Is the submission relevant to work on streaming data?
  * Are references included to the origin source for the algorithm?
  * Does the code perform as expected?
  * Has the code been tested against previously existing implementations?

**Code Quality**
  * Are the [coding guidelines](https://docs.astropy.org/en/latest/development/codeguide.html) followed?
  * Is the code compatible with Python >=3.7?
  * Are there dependencies other than the `WAX` core and `WAX` requirements (see requirements.txt).
    * Are additional dependencies handled appropriately?
    * Do functions that require additional dependencies raise an `ImportError`
      if they are not present?

**Testing**
  * Are the [testing guidelines](https://docs.astropy.org/en/latest/development/testguide.html) followed?
  * Are the inputs to the functions sufficiently tested?
  * Are there tests for any exceptions raised?
  * Are there tests for the expected performance?
  * Are the sources for the tests documented?
  * Have tests that require an [optional dependency](https://docs.astropy.org/en/latest/development/testguide.html#tests-requiring-optional-dependencies)
    been marked as such?
  * Does ``make tests`` run without failures?

**Mypy**
  * Does mypy type checking pass? run the command `make mypy` to check.

**Documentation**
  * `WAX` does not yet have Documentation.

**License**
  * Is the `WAX` license included at the top of the file? If not you can use the command `make license`
    at the root of `WAX` project.
  * Are there any conflicts with this code and existing codes?

**WAX requirements**
  * Do all the GitHub Actions and CircleCI tests pass? If not, are they allowed to fail?
    You may run `make act` to test actions locally.
  * If applicable, has an entry been added into the changelog?
  * Can you check out the pull request and repeat the examples and tests?
