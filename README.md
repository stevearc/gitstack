# gitstack

A utility for stacking branch and github pull requests

- [Installation](#installation)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Why gitstack](#why-gitstack)

## Installation

```bash
curl -L https://github.com/stevearc/gitstack/releases/latest/download/gitstack.py -o ~/.local/share/gitstack.py
git config --global alias.stack '!python ~/.local/share/gitstack.py'
```

Now you can use these commands with `git stack`. Feel free to shorten the alias to something quicker to type.

To use pull request features, you'll also need download the [Github CLI](https://cli.github.com/) and log in with `gh auth login`.

## Quick start

```bash
# create a branch and make commits as normal
git stack create --split  # create a new branch for each commit
git stack pr              # create pull requests for each branch
git stack publish         # publish the pull requests
# go to github and add summary, reviewers, test plan, etc

# If changes are requested:
git stack first           # check out the branch for the first PR
# make fixes
git commit -am "fixes"   # add new commits for fixes
git push
# optionally, you can restack the rest of the branches
git stack restack
# optionally, you can force push the restacked branches
git stack push -f -a
```

You can always learn more about the commands with `git stack -h` or `git stack <cmd> -h`.

## Concepts

gitstack is just trying to automate the work that you would have to do yourself. There is no magic, just a lot of rebasing branches onto other branches.
However, it _does_ need to keep track of how the branches are stacked in order to do a proper restack. To do this, **gitstack adds a `prev-branch: <branch>` label to the commit message of the first commit in a branch**. This is the _only_ data that gitstack stores anywhere, and you should feel empowered to edit it by hand if you like.

`git stack create` will turn the current linear history of branches into a stack, overriding any previous stack. If you need to reorder branches, insert new ones, or make other changes, do so with normal git commands and then re-run `git stack create`.

## Why gitstack

I have found all other tools I have tried to be excessively magic. They make use of storing information locally, which makes it hard to sync a stack and work on it from another machine, or they require you to always interact with the stack using their commands. This tool does the minimal amount possible to improve stacking ergonomics, and then gets out of your way. You're still able to use raw git commands like usual.
