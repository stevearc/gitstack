#!/usr/bin/env python
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache, total_ordering
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

log = logging.getLogger(__name__)

__version__ = "dev"
# in github release action, it will set this to the tag name which has a leading v
if __version__.startswith("v"):
    __version__ = __version__[1:]


def print_err(*args):
    pieces = " ".join([str(a) for a in args])
    sys.stderr.write(pieces)
    sys.stderr.write("\n")


def run(*args, **kwargs) -> str:
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    silence = kwargs.pop("silence", False)
    log.debug("run %s", args)
    try:
        stdout = subprocess.run(args, **kwargs).stdout
    except subprocess.CalledProcessError as e:
        if not silence and kwargs.get("capture_output"):
            print_err("Error running: ", " ".join(args))
            if e.stdout is not None:
                print_err(e.stdout.decode("utf-8"))
            if e.stderr is not None:
                print_err(e.stderr.decode("utf-8"))
        if silence or logging.root.getEffectiveLevel() <= logging.INFO:
            raise
        else:
            sys.exit(e.returncode)
    if stdout is None:
        return ""
    else:
        return stdout.decode("utf-8").strip()


def gh(*args, **kwargs) -> str:
    if shutil.which("gh") is None:
        sys.stderr.write("Missing gh executable\n")
        sys.exit(1)

    return run("gh", *args, **kwargs)


def has_gh() -> bool:
    proc = subprocess.run(
        ["gh", "auth", "status"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def git_lines(*args, **kwargs) -> List[str]:
    ret = run("git", *args, **kwargs)
    if ret:
        return ret.splitlines()
    else:
        return []


def exit_if_no_gh():
    if not has_gh():
        sys.stderr.write("gh cli missing or not authenticated\n")
        sys.exit(1)


class Color:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

    @classmethod
    def red(cls, text: str) -> str:
        return cls.RED + text + cls.ENDC

    @classmethod
    def green(cls, text: str) -> str:
        return cls.GREEN + text + cls.ENDC

    @classmethod
    def blue(cls, text: str) -> str:
        return cls.BLUE + text + cls.ENDC

    @classmethod
    def yellow(cls, text: str) -> str:
        return cls.YELLOW + text + cls.ENDC

    @classmethod
    def cyan(cls, text: str) -> str:
        return cls.CYAN + text + cls.ENDC


@total_ordering
@dataclass
class Commit:
    hash: str
    timestamp: int
    relative_time: str
    branches: List[str]
    subject: str
    labeled: bool = False

    def format(self) -> str:
        pieces = [
            Color.yellow(self.hash),
            Color.cyan(self.relative_time),
        ]
        local_branches = [b for b in self.branches if not b.startswith("origin/")]
        if local_branches:
            branches = ", ".join(
                [
                    Color.red(b) + Color.GREEN if b == "HEAD" else b
                    for b in local_branches
                ]
            )
            pieces.append(Color.yellow("(") + Color.green(branches) + Color.yellow(")"))
        pieces.append(self.subject)
        return " ".join(pieces)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Commit):
            raise NotImplementedError
        return self.timestamp < other.timestamp

    def __hash__(self) -> int:
        return hash(self.hash)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Commit):
            raise NotImplementedError
        return self.timestamp == other.timestamp and self.hash == other.hash


class git:
    @staticmethod
    def switch_branch(branch_name: str) -> None:
        run("git", "checkout", branch_name)

    @staticmethod
    def fetch() -> None:
        run("git", "fetch")

    @staticmethod
    def reset(ref: str, hard: bool = False) -> None:
        args = ["git", "reset"]
        if hard:
            args.append("--hard")
        args.append(ref)
        run(*args)

    @staticmethod
    def fast_forward(branch: str) -> None:
        run("git", "merge", "--ff-only", branch)

    @staticmethod
    def commit(message: str) -> None:
        run("git", "commit", "--allow-empty", "-m", message)

    @staticmethod
    def get_current_ref() -> str:
        return run("git", "rev-parse", "@")

    @staticmethod
    def rev_parse(ref: str) -> Optional[str]:
        try:
            return run("git", "rev-parse", ref, silence=True)
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def list_branches(all: bool = False) -> Dict[str, str]:
        """Mapping of branch name to ref"""
        ret = {}
        args = ["branch", "--format=%(refname:short) %(objectname)"]
        if all:
            args.append("-a")
        for b in git_lines(*args):
            if b.startswith("(HEAD detached at"):
                continue
            name, ref = b.split()
            ret[name] = ref
        return ret

    @staticmethod
    def list_remote_branches() -> List[str]:
        return [
            b.strip()[7:]
            for b in git_lines("branch", "-a", "--format=%(refname:short)")
            if b.startswith("origin/")
        ]

    @staticmethod
    def log_detail(ref: str) -> Commit:
        lines = git_lines("log", "-1", ref, "--format=#%h %at %ar (%D) %B")
        return git._parse_log_detail(lines)[0]

    @staticmethod
    def log_range_detail(start: str, end: str) -> List[Commit]:
        lines = git_lines("log", start + ".." + end, "--format=#%h %at %ar (%D) %B")
        return git._parse_log_detail(lines)

    @staticmethod
    def _parse_log_detail(lines: List[str]) -> List[Commit]:
        commits = []
        commit = None
        for line in lines:
            match = re.match(r"^#(\w+) (\d+) (\d+ \w+ ago) \((.*)\) (.*)$", line)
            if match:
                if commit is not None:
                    commits.append(commit)
                rel_time = match[3][:-4]
                if match[4]:
                    branches = match[4].split(", ")
                    for i, branch in enumerate(branches):
                        if branch.startswith("HEAD ->"):
                            branches[i] = branch[8:]
                            branches.insert(0, "HEAD")
                            break
                else:
                    branches = []
                commit = Commit(match[1], int(match[2]), rel_time, branches, match[5])
            elif line.startswith("prev-branch:") and commit is not None:
                commit.labeled = True
        if commit is not None:
            commits.append(commit)
        commits.reverse()
        return commits

    @staticmethod
    def current_branch() -> Optional[str]:
        branch = run("git", "branch", "--show-current")
        if branch:
            return branch
        else:
            return None

    @staticmethod
    def push(branch: str, force: bool = False) -> None:
        git_args = ["push", "-u", "origin", branch]
        if force:
            git_args.insert(1, "--force-with-lease")
        run("git", *git_args, capture_output=False)

    @staticmethod
    def get_containing_branches(ref: str) -> Dict[str, str]:
        """Returns a mapping of branch name to ref"""
        ret = {}
        for line in git_lines(
            "branch", "--contains", ref, "--format=%(objectname),%(refname:short)"
        ):
            ref, name = line.split(",", 1)
            ret[name] = ref
        return ret

    @staticmethod
    def create_branch(branch: str, start: Optional[str] = None) -> None:
        if start is None:
            start = git.get_main_branch()
        run("git", "checkout", "-b", branch, start)

    @staticmethod
    def delete_branch(branch: str, force: bool = False) -> None:
        flag = "-D" if force else "-d"
        run("git", "branch", flag, branch)

    @staticmethod
    def rebase_onto(new_root: str, old_root: str, tip: str) -> None:
        run("git", "rebase", "--onto", new_root, old_root, tip)

    @staticmethod
    def rebase(new_root: str, tip: str) -> None:
        run("git", "rebase", new_root, tip)

    @staticmethod
    def delete_remote_branch(branch: str) -> None:
        run("git", "push", "origin", "--delete", branch)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_main_branch() -> str:
        proc = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.decode("utf-8").split("/")[-1].strip()
        return "master"

    @staticmethod
    def is_main_branch(branch: str) -> bool:
        # HACK to also handle master-passing-tests
        return branch.startswith(git.get_main_branch())

    @staticmethod
    def get_origin_main() -> str:
        return "origin/" + git.get_main_branch()

    @staticmethod
    def exit_if_dirty():
        output = run("git", "status", "--porcelain")
        if output:
            sys.stderr.write("Working directory is dirty\n")
            sys.exit(1)

    @staticmethod
    def get_commit_message(ref: str) -> str:
        return run("git", "log", "-1", "--format=%B", ref)

    @staticmethod
    def refs_between(ref1: str, ref2: str) -> List[str]:
        """
        Exclusive on ref1, inclusive on ref2
        Ordered from oldest to newest
        """
        refs = git_lines("log", ref1 + "..." + ref2, "--format=%H")
        refs.reverse()
        return refs

    @staticmethod
    def merge_base(branch: str, ref2: Optional[str] = None) -> str:
        if ref2 is None:
            ref2 = git.get_origin_main()
        return run("git", "merge-base", branch, ref2)

    @staticmethod
    def get_child_branches(ref: str) -> List[str]:
        return [
            b
            for b in git_lines("branch", "--contains", ref, "--format=%(refname:short)")
            if b != ref
        ]

    @staticmethod
    def label_commit(ref: str, label: "DiffLabel") -> Optional[str]:
        message = git.get_commit_message(ref)
        lines = [
            line
            for line in message.splitlines()
            if not line.startswith("prev-branch:") and not line.startswith("prev-pr:")
        ]
        # Make sure there's a blank line between the message and the labels
        if lines[-1] != "":
            lines.append("")

        if label.prev_branch is not None:
            lines.append("prev-branch: " + label.prev_branch)
        if label.prev_pr is not None:
            lines.append("prev-pr: " + label.prev_pr.as_str())
        new_message = "\n".join(lines)

        if message == new_message:
            return None

        cur_ref = git.current_branch() or git.get_current_ref()
        git.switch_branch(ref)
        run("git", "commit", "--allow-empty", "--amend", "-m", new_message)

        new_ref = git.get_current_ref()
        git.switch_branch(cur_ref)
        return new_ref


@dataclass
class PullRequestLink:
    number: Optional[int]
    link: Optional[str]

    def __init__(self, number: Optional[int], link: Optional[str]):
        # You must specify exactly one of number or link
        assert (number is None) != (link is None)
        self.number = number
        self.link = link

    @classmethod
    def from_num(cls, pr: Optional[int]) -> Optional["PullRequestLink"]:
        if pr is None:
            return None
        return cls(pr, None)

    @classmethod
    def from_str(cls, s: str) -> "PullRequestLink":
        if s.startswith("#"):
            return cls(int(s[1:]), None)
        elif s.isdigit():
            return cls(int(s), None)
        elif re.match(r"^https?://", s):
            return cls(None, s)
        else:
            raise ValueError(
                f"Invalid PR link. Expected PR number or link to another repo: {s}"
            )

    def as_str(self) -> str:
        if self.number is not None:
            return str(self.number)
        else:
            assert self.link is not None
            return self.link


@dataclass
class DiffLabel:
    prev_branch: Optional[str] = None
    prev_pr: Optional[PullRequestLink] = None

    @property
    def is_empty(self) -> bool:
        return self.prev_branch is None and self.prev_pr is None


@dataclass
class Branch:
    name: str
    num_commits: int

    @property
    def root(self) -> str:
        if self.num_commits == 1:
            return self.name
        else:
            return self.name + "~" + str(self.num_commits - 1)

    @property
    def tip(self) -> str:
        return self.name

    def __len__(self) -> int:
        return self.num_commits

    def __iter__(self):
        for i in range(self.num_commits):
            if i == self.num_commits - 1:
                yield self.name
            else:
                yield self.name + "~" + str(self.num_commits - i - 1)

    def __repr__(self) -> str:
        return f"Branch({self.name}, {self.num_commits})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Branch):
            return False
        return self.name == other.name

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


def parse_markdown_table(body: str) -> Tuple[str, str]:
    table = []
    rest = []
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|"):
            table.append(line)
        else:
            rest = lines[i:]
            break
    return "\n".join(table), "\n".join(rest)


def make_markdown_table(data: List[Dict[str, str]], cols: List[str]) -> str:
    max_width = [len(col) for col in cols]
    for row in data:
        for i, col in enumerate(cols):
            max_width[i] = max(max_width[i], len(row[col]))

    lines = [
        "| "
        + " | ".join([col.center(max_width[i]) for i, col in enumerate(cols)])
        + " |",
        "| " + " | ".join([max_width[i] * "-" for i in range(len(cols))]) + " |",
    ]
    for row in data:
        lines.append(
            "| "
            + " | ".join([row[col].ljust(max_width[i]) for i, col in enumerate(cols)])
            + " |",
        )
    return "\n".join(lines)


PR_TITLE_RE = re.compile(r"^(\[\d+/\d+\])?\s*(WIP:)?\s*(.*)$")


class PullRequest:
    def __init__(
        self,
        number: int,
        title: str,
        body: str,
        url: str,
        is_draft: bool,
        repo_name: str,
    ):
        match = PR_TITLE_RE.match(title)
        assert match
        self.number = number
        self.raw_title = title
        self.title = match[3]
        self.raw_body = body
        self.table, self.body = parse_markdown_table(body)
        self.url = url
        self.repo_name = repo_name
        self.is_draft = is_draft

    def __hash__(self) -> int:
        return self.number

    def __eq__(self, other) -> bool:
        if not isinstance(other, PullRequest):
            return False
        return self.number == other.number

    @classmethod
    def from_ref(cls, num_or_branch: Union[str, int]) -> "PullRequest":
        return cls.from_json(
            json.loads(
                gh(
                    "pr",
                    "view",
                    str(num_or_branch),
                    "--json",
                    "title,body,number,url,isDraft,headRepository",
                )
            )
        )

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PullRequest":
        return cls(
            data["number"],
            data["title"],
            data["body"],
            data["url"],
            data["isDraft"],
            data["headRepository"]["name"],
        )

    def parse_prev_prs_from_table(self) -> List["PullRequest"]:
        """Return all of the PR numbers in the list"""
        ret: List["PullRequest"] = []
        for line in self.table.splitlines():
            # Skip lines until we reach the data rows
            if not re.match(r"^\|\s*(\d+)\s*\|", line):
                continue
            pieces = line.split("|")
            if len(pieces) < 3:
                continue
            pr_str = pieces[2].strip()
            # If we have reached the current PR, return what we found so far
            if pr_str.startswith(">"):
                return ret
            elif pr_str.startswith("#"):
                ret.append(PullRequest.from_ref(int(pr_str[1:])))
            else:
                # Parse the URL out of a markdown link
                match = re.match(r"^\[[^\]]*\]\((.*)\)$", line)
                if match:
                    ret.append(PullRequest.from_ref(match[1]))
        return ret

    @staticmethod
    def get_title(index: int, total: int, title: str, is_draft: bool) -> str:
        wip = "WIP: " if is_draft else ""
        if total == 1:
            return f"{wip}{title}"
        else:
            return f"[{index}/{total}] {wip}{title}"

    def set_title(self, index: int, total: int, title: str) -> bool:
        new_title = self.get_title(index, total, title, self.is_draft)
        if new_title != self.raw_title:
            gh("pr", "edit", self.url, "-t", new_title)
            self.title = title
            self.raw_title = new_title
            return True
        else:
            return False

    def set_draft(self, is_draft: bool) -> bool:
        if is_draft == self.is_draft:
            return False
        args = ["pr", "ready", self.url]
        if is_draft:
            args.append("--undo")
        gh(*args)
        self.is_draft = is_draft
        return True

    def set_table(self, stack_prs: List["PullRequest"]) -> bool:
        rows = []
        for i, pr in enumerate(stack_prs):
            title = pr.title
            if pr.is_draft:
                title = "WIP: " + title
            row = {"Title": title, "": str(i + 1)}
            if pr.url == self.url:
                row["PR"] = f">{pr.number}"
            elif pr.repo_name == self.repo_name:
                row["PR"] = f"#{pr.number}"
            else:
                row["PR"] = f"[{pr.repo_name}#{pr.number}]({pr.url})"
            rows.append(row)
        table = make_markdown_table(rows, ["", "PR", "Title"])
        if len(stack_prs) == 1:
            table = ""
        if table != self.table:
            new_body = table + "\n" + self.body
            gh("pr", "edit", self.url, "-b", new_body)
            self.raw_body = new_body
            self.table = table
            return True
        else:
            return False


class Diff:
    """
    An atomic unit in a stack

    These correspond 1-to-1 with branches and with PRs, but they can exist without a
    branch (if the branch has already been merged and deleted) or without a PR (if the
    PR has not been created yet).
    """

    def __init__(
        self,
        branch: Optional[Branch],
        pr: Optional[PullRequest],
        label: DiffLabel,
    ):
        assert branch is not None or pr is not None
        self.branch = branch
        self.pr = pr
        self.label = label

    def get_label_for_diff(self) -> DiffLabel:
        name = None if self.branch is None else self.branch.name
        pr = self.pr.number if self.pr is not None else None
        return DiffLabel(name, PullRequestLink.from_num(pr))

    def add_label(self, label: DiffLabel) -> None:
        if self.branch is None:
            raise RuntimeError("Cannot add label to a diff without a branch")
        children = git.get_child_branches(self.branch.name)
        new_ref = git.label_commit(self.branch.root, label)
        if new_ref is None:
            return
        old_tip = self.branch.tip
        # Rebase the successive commits on top of the new ref
        git.rebase_onto(new_ref, self.branch.root, self.branch.name)

        # Rebase all child branches
        new_root = self.branch.name
        old_root = old_tip
        for child in children:
            old_branch_tip = git.rev_parse(child)
            assert old_branch_tip is not None
            git.rebase_onto(new_root, old_root, child)
            new_root = child
            old_root = old_branch_tip

    @staticmethod
    def parse_labels(ref: str) -> DiffLabel:
        label = DiffLabel()
        for line in git.get_commit_message(ref).splitlines()[1:]:
            if line.startswith("prev-branch:"):
                label.prev_branch = line.split(":")[1].strip()
            elif line.startswith("prev-pr:"):
                label.prev_pr = PullRequestLink.from_str(line.split(":", 1)[1].strip())
        return label

    @classmethod
    def from_branch(cls, branch: str, tips: Dict[str, str]) -> "Diff":
        refs = git.refs_between(git.merge_base(branch), branch)
        # Find the most recent commit that has a label
        for i, ref in enumerate(reversed(refs)):
            label = Diff.parse_labels(ref)
            if i > 0 and ref in tips:
                # We've encountered another branch, so this must be an unlabeled diff
                return cls(
                    Branch(branch, i),
                    pr=None,
                    label=DiffLabel(prev_branch=tips[ref]),
                )
            elif not label.is_empty:
                return cls(Branch(branch, i + 1), pr=None, label=label)
        return cls(Branch(branch, len(refs)), pr=None, label=DiffLabel())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Diff):
            return False
        if self.branch is not None:
            return self.branch == other.branch
        else:
            return self.pr == other.pr

    def __hash__(self) -> int:
        if self.branch is not None:
            return hash(self.branch.name)
        else:
            assert self.pr is not None
            return self.pr.number

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        args = []
        if self.branch is not None:
            args.append(repr(self.branch))
        if self.pr is not None:
            args.append(f"#{self.pr}")
        return f"Diff({', '.join(args)})"


class Stack:
    def __init__(self, diffs: List[Diff]):
        self._diffs = diffs

    def local_diffs(self, before_branch: Optional[str] = None) -> List[Diff]:
        ret = [d for d in self._diffs if d.branch is not None]
        if before_branch is not None:
            while ret[-1].branch is not None and ret[-1].branch.name != before_branch:
                ret.pop()
        return ret

    def branches(self, before_branch: Optional[str] = None) -> List[Branch]:
        return [
            d.branch for d in self.local_diffs(before_branch) if d.branch is not None
        ]

    def hydrate_prs(self, pr_map: Dict[str, PullRequest]) -> None:
        for diff in self._diffs:
            if diff.branch is not None:
                pr = pr_map.get(diff.branch.name)
                if pr is not None:
                    diff.pr = pr

        # gh pr status doesn't show closed PRs, so we need to fetch them separately
        any_new_prs = True
        while any_new_prs:
            any_new_prs = False
            first_diff = self._diffs[0]
            if first_diff.label.prev_pr:
                diff = Diff(
                    None,
                    PullRequest.from_ref(first_diff.label.prev_pr.as_str()),
                    DiffLabel(),
                )
                any_new_prs = True
                self._diffs.insert(0, diff)
            elif first_diff.pr:
                prev_prs = first_diff.pr.parse_prev_prs_from_table()
                prev_diffs = [Diff(None, pr, DiffLabel()) for pr in prev_prs]
                any_new_prs = bool(prev_diffs)
                self._diffs = prev_diffs + self._diffs

    def create_prs(
        self, before_branch: Optional[str] = None, publish: bool = False
    ) -> List[Diff]:
        total = len(self._diffs)
        created = []
        rel = git.get_main_branch()
        repo_root = git.rev_parse("--show-toplevel")
        assert repo_root is not None
        body_file = os.path.join(repo_root, ".github", "PULL_REQUEST_TEMPLATE.md")
        for i, diff in enumerate(self.local_diffs(before_branch)):
            assert diff.branch is not None
            if not diff.pr:
                commit_line = (
                    git.get_commit_message(diff.branch.name).splitlines()[0].strip()
                )
                title = PullRequest.get_title(i + 1, total, commit_line, True)
                args = [
                    "pr",
                    "create",
                    "--head",
                    diff.branch.name,
                    "-B",
                    rel,
                    "-t",
                    title,
                ]
                if not publish:
                    args.append("-d")
                if os.path.exists(body_file):
                    args.extend(["-F", body_file])
                else:
                    args.extend(["-b", ""])
                gh(*args, capture_output=False)
                diff.pr = PullRequest.from_ref(diff.branch.name)
                created.append(diff)
            rel = diff.branch.name
        return created

    def update_prs(self) -> List[Diff]:
        total = len(self._diffs)
        updated = set()

        for i, diff in enumerate(self._diffs):
            pr = diff.pr
            if pr is not None and pr.set_title(i + 1, total, pr.title):
                updated.add(diff)
        pull_requests = [diff.pr for diff in self._diffs if diff.pr is not None]
        for diff in self._diffs:
            pr = diff.pr
            if pr is not None and pr.set_table(pull_requests):
                updated.add(diff)
        return list(updated)

    def get_diff_for_ref(self, ref: Optional[str] = None) -> Optional[Diff]:
        if ref is None:
            ref = "@"
        branches = git.get_containing_branches(ref)
        for diff in self._diffs:
            if diff.branch is not None and diff.branch.name in branches:
                return diff
        return None

    def get_prev_diff(self, diff: Diff) -> Optional[Diff]:
        for i, d in enumerate(self._diffs):
            if d == diff:
                if i > 0:
                    return self._diffs[i - 1]
        return None

    def get_next_diff(self, diff: Diff) -> Optional[Diff]:
        for i, d in enumerate(self._diffs):
            if d == diff:
                if i < len(self._diffs) - 1:
                    return self._diffs[i + 1]
        return None

    def rebase(self, target: Optional[str] = None):
        original_branch = git.current_branch()

        branches = self.branches()

        if target is not None:
            git.rebase(target, branches[0].name)
            if not git.is_main_branch(target):
                diff = next((d for d in self._diffs if d.branch == branches[0]), None)
                assert diff is not None
                diff.add_label(DiffLabel(prev_branch=target))

        prev_branch = branches[0].name
        for branch in branches[1:]:
            if branch.name not in git.get_child_branches(prev_branch):
                git.rebase_onto(prev_branch, branch.root + "^", branch.name)
            prev_branch = branch.name

        if original_branch is not None:
            git.switch_branch(original_branch)

        # delete branches that have been merged
        if target is not None and git.is_main_branch(target):
            cur = git.current_branch()
            git.switch_branch(git.get_main_branch())
            for branch in branches:
                if git.merge_base(branch.name) == git.rev_parse(branch.name):
                    git.delete_branch(branch.name, force=True)
                else:
                    break
            if cur is not None and git.rev_parse(cur) is not None:
                git.switch_branch(cur)

    def split_diff_by_commits(self, diff: Diff) -> None:
        assert diff in self._diffs
        branch = diff.branch
        if branch is None:
            raise RuntimeError("Could not find branch for current diff")

        start_diff_idx = self._diffs.index(diff)
        if len(branch) > 1:
            match = re.match(r"^(.*)\-(\d+)$", branch.name)
            if match:
                base_branch_name = match.group(1)
                base_i = int(match.group(2))
            else:
                base_branch_name = branch.name
                base_i = 1
            refs = git.refs_between(branch.root, branch.name)
            # Delete the existing branch name; it will be replaced by the new numbered branches
            git.switch_branch(branch.root)
            git.delete_branch(branch.name, force=True)

            insert_idx = start_diff_idx

            # Remove the diff we're splitting before adding its replacements
            self._diffs.remove(diff)
            # refs doesn't have the root commit, so we need to add it
            refs.insert(0, "HEAD")
            for i, ref in enumerate(refs):
                new_branch_name = base_branch_name + "-" + str(base_i + i)
                git.create_branch(new_branch_name, ref)
                new_diff = Diff(
                    Branch(new_branch_name, 1),
                    pr=None,
                    label=DiffLabel(),
                )
                self._diffs.insert(insert_idx, new_diff)
                insert_idx += 1

        for i in range(len(branch)):
            diff = self._diffs[start_diff_idx + i]

            prev_diff = self.get_prev_diff(diff)
            if prev_diff is None:
                continue
            prev_branch = prev_diff.branch
            if prev_branch is None:
                continue
            diff.add_label(DiffLabel(prev_branch=prev_branch.name))

    def print_simple(self) -> None:
        pieces = []
        for diff in self._diffs:
            pieces.append(str(diff))
        print(" -> ".join(pieces))

    def print_branches(self) -> None:
        cur = git.current_branch()
        for branch in reversed(self.branches()):
            if branch.name == cur:
                print("*", branch.name)
            else:
                print(" ", branch.name)

    def contains_ref(self, ref: Optional[str] = None) -> bool:
        return self.get_diff_for_ref(ref) is not None


class Repo:
    def __init__(self, stacks: List[Stack]):
        self.stacks = stacks

    def get_stack_for_ref(self, ref: Optional[str] = None) -> Optional[Stack]:
        if ref is None:
            ref = git.get_current_ref()
        for stack in self.stacks:
            if stack.contains_ref(ref):
                return stack
        return None

    def load_prs(self) -> None:
        prs = json.loads(
            gh(
                "pr",
                "status",
                "--json",
                "title,body,number,headRefName,url,isDraft,headRepository",
                silence=True,
            )
        )
        pr_map: Dict[str, PullRequest] = {
            pr["headRefName"]: PullRequest.from_json(pr) for pr in prs["createdBy"]
        }
        for stack in self.stacks:
            stack.hydrate_prs(pr_map)

    @classmethod
    def load(cls) -> "Repo":
        branches = git.list_branches()
        stacks = create_stacks(branches)
        return cls(stacks)


def strip_remote(branch: str) -> str:
    if branch.startswith("origin/"):
        return branch[7:]
    else:
        return branch


def create_stacks(branches: Dict[str, str]) -> List[Stack]:
    diffs: Dict[str, Diff] = {}
    tips = {v: k for k, v in branches.items()}
    # We have to strip the remote prefix off of the keys in diffs and diffs_by_prev
    # because the commit labels themselves do not have the remote prefix. Regardless of
    # the real branch name, we have to always look it up using the representation from
    # the commit labels, which is local-only.
    for branch in branches:
        if strip_remote(branch) != git.get_main_branch():
            diffs[strip_remote(branch)] = Diff.from_branch(branch, tips)
    stacks = []
    diffs_by_prev: Dict[str, Diff] = {}
    diff: Optional[Diff] = None
    for diff in diffs.values():
        if diff.label.prev_branch is not None:
            diffs_by_prev[diff.label.prev_branch] = diff

    while diffs:
        diff = None
        # Find a diff with no previous branch
        for d in diffs.values():
            if d.label.prev_branch not in diffs or d.label.prev_branch is None:
                diff = d
                break
        if diff is None:
            diff = diffs[next(iter(diffs))]
        assert diff.branch is not None
        del diffs[strip_remote(diff.branch.name)]

        diff_list = [diff]
        while strip_remote(diff.branch.name) in diffs_by_prev:
            next_diff = diffs_by_prev.pop(strip_remote(diff.branch.name))
            assert next_diff.branch is not None
            if strip_remote(next_diff.branch.name) in diffs:
                del diffs[strip_remote(next_diff.branch.name)]
                diff_list.append(next_diff)
            else:
                # There is probably a cycle somewhere
                break
            diff = next_diff
            assert diff.branch is not None

        stacks.append(Stack(diff_list))
    return stacks


@dataclass
class BranchInfo:
    name: str
    commits: List[Commit]
    contains: List[str]

    @property
    def tip(self) -> Commit:
        return self.commits[-1]


@total_ordering
class TmpLocalInfo:
    def __init__(
        self,
        stack: Stack,
        merge_base: Commit,
        commits: Dict[str, List[Commit]],
        contains: Dict[str, List[str]],
    ):
        self.stack = stack
        self.merge_base = merge_base
        self.commits = commits
        self.contains = contains

    @property
    def first_commit(self) -> Commit:
        return self.commits[self.stack.branches()[0].name][0]

    @classmethod
    def from_stack(cls, stack: Stack) -> Optional["TmpLocalInfo"]:
        branches = stack.branches()
        if not branches:
            return None
        merge_base = git.merge_base(branches[0].name)
        commits: Dict[str, List[Commit]] = {}
        contains: Dict[str, List[str]] = {}
        for branch in branches:
            commits[branch.name] = git.log_range_detail(
                branch.name + "~" + str(branch.num_commits), branch.name
            )
            contains[branch.name] = list(
                git.get_containing_branches(branch.name).keys()
            )

        mbc = git.log_detail(merge_base)
        return cls(stack, mbc, commits, contains)

    def branches(self) -> Iterator[BranchInfo]:
        for branch in self.stack.branches():
            yield BranchInfo(
                branch.name, self.commits[branch.name], self.contains[branch.name]
            )

    def __bool__(self) -> bool:
        return bool(self.commits[self.stack.branches()[0].name])

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, TmpLocalInfo):
            raise NotImplementedError
        if self.merge_base.timestamp == other.merge_base.timestamp:
            return self.first_commit.timestamp < other.first_commit.timestamp
        return self.merge_base.timestamp < other.merge_base.timestamp

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TmpLocalInfo):
            raise NotImplementedError
        return (
            self.merge_base.timestamp == other.merge_base.timestamp
            and self.first_commit.timestamp == other.first_commit.timestamp
        )


def navigate_stack_relative(count: int):
    repo = Repo.load()
    cur = git.current_branch()
    if cur is None:
        print_err("On detached head")
        sys.exit(1)
    stack = repo.get_stack_for_ref()
    if stack is None:
        print_err("No stack found")
        sys.exit(1)
    branches = [branch.name for branch in stack.branches()]
    idx = branches.index(cur)
    new_idx = max(0, min(len(branches) - 1, idx + count))
    git.switch_branch(branches[new_idx])
    stack.print_branches()


class Command(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def help(self) -> str:
        raise NotImplementedError

    def description(self) -> Optional[str]:
        return self.help

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def invoke(self, args: argparse.Namespace) -> None:
        raise NotImplementedError


class ListCommand(Command):
    @property
    def name(self) -> str:
        return "list"

    @property
    def help(self) -> str:
        return "List all stacks"

    def description(self) -> str:
        return """List all stacks
The commits are displayed as a git log graph, with the following icons:
  * normal commit
  ? branch is stacked, but missing a label (use "git stack create" to add it)
  x needs to be restacked"""

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-c",
            "--commits",
            action="store_true",
            help="Show all commits, not just branch tips",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(show_all_commits=args.commits)

    def run(self, show_all_commits: bool = False) -> None:
        repo = Repo.load()
        local_info: List[TmpLocalInfo] = []
        for stack in repo.stacks:
            info = TmpLocalInfo.from_stack(stack)
            if info:
                local_info.append(info)

        master = git.log_detail(git.get_main_branch())
        local_info.sort()
        merge_bases = {info.merge_base for info in local_info}
        merge_bases.add(master)

        lines = []
        for i, commit in enumerate(sorted(merge_bases)):
            lines.append("* " + commit.format())
            commit_infos = [info for info in local_info if info.merge_base == commit]
            for j, info in enumerate(commit_infos):
                is_last = i == len(merge_bases) - 1 and j == len(commit_infos) - 1
                if is_last:
                    indent = "  "
                    fork = Color.red(" /")
                else:
                    indent = Color.red("| ")
                    fork = Color.red("|/")

                lines.append(fork)
                needs_restack = False
                last_branch = None
                for branch_num, branch in enumerate(info.branches()):
                    labeled = any(c.labeled for c in branch.commits)
                    if last_branch is not None and not needs_restack:
                        needs_restack = branch.name not in last_branch.contains

                    star = "*"
                    if needs_restack:
                        star = "x"
                    elif branch_num > 0 and not labeled:
                        star = "?"

                    if show_all_commits:
                        for commit in branch.commits[:-1]:
                            lines.append(f"{indent}{star} {commit.format()}")
                    lines.append(f"{indent}{star} {branch.tip.format()}")
                    last_branch = branch

        lines.reverse()
        for line in lines:
            print(line)


class CreateCommand(Command):
    @property
    def name(self) -> str:
        return "create"

    @property
    def help(self) -> str:
        return "Create a new stack from the current branch"

    def description(self) -> Optional[str]:
        return """Create a stack from the current linear history of branches.
If any of these branches were previously part of a stack, it will update the stack to
reflect the current state of the branches. You can use this to easily edit a stack
(inserting new branches, deleting branches, reordering branches) using normal git
commits and then re-running this command.

All this command is actually doing is updating the commit messages of the first commit
of each branch."""

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-p",
            "--pull-request",
            metavar="PR",
            help="Stack the first diff on top of this pull request (can be a number, or a link to a PR in another repo)",
            required=False,
        )
        parser.add_argument(
            "-s",
            "--split",
            action="store_true",
            help="Split the current branch into one branch per commit",
            required=False,
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(split=args.split, pull_request=args.pull_request)

    def run(self, split: bool = False, pull_request: Optional[str] = None) -> Stack:
        git.exit_if_dirty()
        repo = Repo.load()

        current_branch = git.current_branch()
        if current_branch is None:
            print_err("On detached head")
            sys.exit(1)

        merge_base = git.merge_base("@")
        if merge_base == git.get_current_ref():
            # TODO maybe provide more guidance? Maybe create a branch if a name is passed in?
            print_err("No branch detected")
            sys.exit(1)

        branches = git.get_containing_branches("@")
        if len(branches) > 1:
            print_err("Multiple branches detected:", ", ".join(branches))
            sys.exit(1)
        # Since we're not on a detached head, this should be impossible
        assert len(branches) == 1

        refs_between = git.refs_between(merge_base, "@")
        refs_to_branch = {
            v: k for k, v in git.get_containing_branches(refs_between[0]).items()
        }

        diffs: List[Diff] = []
        num_commits = 1
        for ref in refs_between:
            branch = refs_to_branch.get(ref)
            if branch is not None:
                diff = Diff(
                    Branch(branch, num_commits),
                    pr=None,
                    label=DiffLabel(),
                )
                diffs.append(diff)
                num_commits = 1
            else:
                num_commits += 1

        if pull_request is not None:
            pr = PullRequestLink.from_str(pull_request)
            diffs[0].add_label(DiffLabel(prev_pr=pr))

        for i, diff in enumerate(diffs):
            if i > 0:
                prev_diff = diffs[i - 1]
                assert prev_diff.branch is not None
                diff.add_label(DiffLabel(prev_branch=prev_diff.branch.name))

        stack = Stack(diffs)
        if split:
            stack.split_diff_by_commits(diffs[-1])
        elif pull_request is None and len(stack.branches()) == 1:
            print_err("Only one branch detected. Did you mean to use --split?")
        return stack


class RestackCommand(Command):
    @property
    def name(self) -> str:
        return "restack"

    @property
    def help(self) -> str:
        return "Rebase stack branches to form a linear history"

    def invoke(self, args: argparse.Namespace) -> None:
        self.run()

    def run(self) -> None:
        repo = Repo.load()
        stack = repo.get_stack_for_ref()
        if stack is None:
            print_err("No stack found")
            sys.exit(1)
        stack.rebase()


class RebaseCommand(Command):
    @property
    def name(self) -> str:
        return "rebase"

    @property
    def help(self) -> str:
        return "Rebase all branches in a stack"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("target", help="Target revision to rebase onto")

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(args.target)

    def run(self, target: str) -> None:
        repo = Repo.load()
        stack = repo.get_stack_for_ref()
        if stack is None:
            print_err("No stack found")
            sys.exit(1)
        stack.rebase(target)


class PushCommand(Command):
    @property
    def name(self) -> str:
        return "push"

    @property
    def help(self) -> str:
        return "Push branches of a stack"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Push all branches, not just the earlier ones",
        )
        parser.add_argument(
            "-f", "--force", action="store_true", help="push with force-with-lease"
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(all_branches=args.all, force=args.force)

    def run(self, all_branches: bool = False, force: bool = False) -> None:
        repo = Repo.load()
        stack = repo.get_stack_for_ref()
        if stack is None:
            print_err("No stack found")
            sys.exit(1)
        cur = git.current_branch()
        before_branch = None if all_branches else cur
        for branch in stack.branches(before_branch):
            git.switch_branch(branch.name)
            git.push(branch.name, force=force)
        if cur is not None:
            git.switch_branch(cur)


class PullRequestCommand(Command):
    @property
    def name(self) -> str:
        return "pr"

    @property
    def help(self) -> str:
        return "Create or update pull requests for a stack"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Create pull requests for all branches, not just the earlier ones",
        )
        parser.add_argument(
            "-p",
            "--publish",
            action="store_true",
            help="Created pull requests will not be in draft mode",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(all_branches=args.all, publish=args.publish)

    def run(self, all_branches: bool = False, publish: bool = False) -> None:
        exit_if_no_gh()
        repo = Repo.load()

        # First push all branches
        stack = repo.get_stack_for_ref()
        if stack is None:
            print_err("No stack found")
            sys.exit(1)
        cur = git.current_branch()
        before_branch = None if all_branches else cur
        for branch in stack.branches(before_branch):
            if git.rev_parse(branch.name) == git.rev_parse("origin/" + branch.name):
                continue
            git.switch_branch(branch.name)
            git.push(branch.name)
        if cur is not None:
            git.switch_branch(cur)

        repo.load_prs()
        before_branch = None if all_branches else git.current_branch()
        created = stack.create_prs(before_branch, publish=publish)
        updated = stack.update_prs()
        for diff in stack._diffs:
            pr = diff.pr
            if pr is not None:
                if diff in created:
                    print("Created  ", pr.url)
                elif diff in updated:
                    print("Updated  ", pr.url)
                else:
                    print("Unchanged", pr.url)


class PublishCommand(Command):
    @property
    def name(self) -> str:
        return "publish"

    @property
    def help(self) -> str:
        return "Publish pull requests (remove draft status)"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Publish pull requests for all branches, not just the earlier ones",
        )
        parser.add_argument(
            "-u",
            "--undo",
            action="store_true",
            help="Convert pull request back to draft mode (current branch only)",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(all_branches=args.all, undo=args.undo)

    def run(self, all_branches: bool = False, undo: bool = False) -> None:
        exit_if_no_gh()
        repo = Repo.load()
        repo.load_prs()
        stack = repo.get_stack_for_ref()
        if stack is None:
            print_err("No stack found")
            sys.exit(1)
        unpublished: Set[Diff] = set()
        published: Set[Diff] = set()
        updated: Set[Diff] = set()
        if undo:
            diff = stack.get_diff_for_ref()
            assert diff is not None
            pr = diff.pr
            if not pr:
                print_err("No pull request found")
                sys.exit(1)
            if pr.set_draft(True):
                unpublished.add(diff)
            updated.update(stack.update_prs())
        else:
            before_branch = None if all_branches else git.current_branch()
            for diff in stack.local_diffs(before_branch):
                pr = diff.pr
                if pr and pr.set_draft(False):
                    published.add(diff)
            updated.update(stack.update_prs())
        for diff in stack._diffs:
            pr = diff.pr
            if pr:
                if diff in published:
                    print("Published  ", pr.url)
                elif diff in unpublished:
                    print("Unpublished", pr.url)
                elif diff in updated:
                    print("Updated    ", pr.url)
                else:
                    print("Unchanged  ", pr.url)


class PrevCommand(Command):
    @property
    def name(self) -> str:
        return "prev"

    @property
    def help(self) -> str:
        return "Check out previous branch in the stack"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "count",
            nargs="?",
            type=int,
            default=1,
            help="Jump backwards this many branches",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(args.count)

    def run(self, count: int = 1) -> None:
        navigate_stack_relative(-1 * count)


class NextCommand(Command):
    @property
    def name(self) -> str:
        return "next"

    @property
    def help(self) -> str:
        return "Check out next branch in the stack"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "count",
            nargs="?",
            type=int,
            default=1,
            help="Jump forwards this many branches",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(args.count)

    def run(self, count: int = 1) -> None:
        navigate_stack_relative(count)


class TipCommand(Command):
    @property
    def name(self) -> str:
        return "tip"

    @property
    def help(self) -> str:
        return "Check out the tip branch in the stack"

    def invoke(self, args: argparse.Namespace) -> None:
        self.run()

    def run(self) -> None:
        navigate_stack_relative(10000)


class FirstCommand(Command):
    @property
    def name(self) -> str:
        return "first"

    @property
    def help(self) -> str:
        return "Check out the first branch in the stack"

    def invoke(self, args: argparse.Namespace) -> None:
        self.run()

    def run(self) -> None:
        navigate_stack_relative(-10000)


class DeleteCommand(Command):
    @property
    def name(self) -> str:
        return "delete"

    @property
    def help(self) -> str:
        return "Remove commit message labels that power the stacking"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Delete labels for all branches in the current stack, not just the current one",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(all_branches=args.all)

    def run(self, all_branches: bool = False) -> None:
        cur = git.current_branch()
        if cur is None:
            print_err("On detached head")
            sys.exit(1)
        repo = Repo.load()
        stack = repo.get_stack_for_ref(cur)
        if stack is None:
            print_err("No stack found")
            sys.exit(1)

        diff: Optional[Diff] = None
        if all_branches:
            for diff in stack.local_diffs():
                diff.add_label(DiffLabel())
        else:
            diff = stack.get_diff_for_ref(cur)
            assert diff is not None
            diff.add_label(DiffLabel())


class PullCommand(Command):
    @property
    def name(self) -> str:
        return "pull"

    @property
    def help(self) -> str:
        return "Fetch all branches from a remote stack"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "branch",
            nargs="?",
            help="Name of the last branch of the stack (default: current branch)",
        )
        parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Force override local branches (you will lose any local work)",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(branch=args.branch, force=args.force)

    def run(self, branch: Optional[str] = None, force: bool = False) -> None:
        if branch is None:
            branch = git.current_branch()
        if branch is None:
            print_err("No branch detected")
            sys.exit(1)
        target = branch

        git.fetch()
        all_branches = git.list_branches(all=True)
        remote_branches = {
            k: v for k, v in all_branches.items() if k.startswith("origin/")
        }
        stacks = create_stacks(remote_branches)
        stack = None
        for candidate in stacks:
            branches = [b.name for b in candidate.branches()]
            if "origin/" + target in branches:
                stack = candidate
                break
        if stack is None:
            print_err("No stack found for branch", target)
            sys.exit(1)

        for remote_branch in stack.branches():
            assert remote_branch.name.startswith("origin/")
            local_branch = remote_branch.name[7:]

            if local_branch not in all_branches:
                git.create_branch(local_branch, "HEAD")
            else:
                git.switch_branch(local_branch)

            if force:
                git.reset(remote_branch.name, hard=True)
            else:
                git.fast_forward(remote_branch.name)

        git.switch_branch(target)


class UpdateCommand(Command):
    @property
    def name(self) -> str:
        return "update"

    @property
    def help(self) -> str:
        return "Update gitstack to the latest version"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "version",
            nargs="?",
            default="latest",
            help="The specific version to update to (default: latest)",
        )

    def invoke(self, args: argparse.Namespace) -> None:
        self.run(version=args.version)

    def run(self, version: str = "latest") -> None:
        if __version__ == "dev":
            msg = "gitstack is currently a development version. Are you sure you want to update? [y/n] "
            if input(msg).lower() != "y":
                return

        if version == "latest":
            contents = urllib.request.urlopen(
                "https://api.github.com/repos/stevearc/gitstack/releases"
            ).read()
            releases = json.loads(contents)
            versions = [r["tag_name"] for r in releases]
            version = versions[0]
            if version[1:] == __version__:
                print("Already up to date")
                return
            if input(f"Update to {version}? [y/n] ").lower() != "y":
                return

        if version == "dev":
            url = (
                "https://raw.githubusercontent.com/stevearc/gitstack/master/gitstack.py"
            )
        else:
            url = f"https://github.com/stevearc/gitstack/releases/download/{version}/gitstack.py"
        (new_file, _) = urllib.request.urlretrieve(url)
        os.unlink(__file__)
        shutil.move(new_file, __file__)
        print(f"Installed version {version}")


def print_examples():
    print(
        """
examples:
    # create a new stack with one branch per commit
    git stack create --split

    # rebase a stack after one PR is merged
    git fetch
    git stack rebase origin/master
    git stack push -f -a      # force push the later branches

    # add more changes to the end of a stack
    git stack tip
    git checkout -b new-branch
    git commit -am "more work"
    git stack create
    git stack pr

    # to stack a branch on top of an already merged pull request
    git checkout -b new-branch
    git commit -am "more work"
    git stack create -p 233    # Stack on top of PR #233
    git stack pr"""
    )


def _setup_logging(args: argparse.Namespace) -> None:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(levelname)s %(asctime)s %(message)s")
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    level = logging.getLevelName(args.log_level.upper())
    logging.root.setLevel(level)


def main() -> None:
    """Stack git branches and pull requests"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["error", "warning", "info", "debug"],
        help="Stderr logging level (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"gitstack version {__version__}",
        help="Print version information and exit",
    )

    subparsers = parser.add_subparsers(dest="cmd", required=False)
    stack_commands: List[Command] = [
        ListCommand(),
        CreateCommand(),
        RestackCommand(),
        RebaseCommand(),
        PrevCommand(),
        NextCommand(),
        TipCommand(),
        FirstCommand(),
        PushCommand(),
        PullRequestCommand(),
        PublishCommand(),
        DeleteCommand(),
        PullCommand(),
        UpdateCommand(),
    ]
    for cmd in stack_commands:
        cmd.add_args(
            subparsers.add_parser(
                cmd.name,
                help=cmd.help,
                description=cmd.description(),
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
        )
    subparsers.add_parser("help", help="Print help information")

    args = parser.parse_args()
    _setup_logging(args)

    for cmd in stack_commands:
        if cmd.name == args.cmd:
            cmd.invoke(args)
            break
    else:
        if args.cmd == "help":
            parser.print_help()
            print_examples()
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
