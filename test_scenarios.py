import argparse
import os
import sys

parent = os.path.dirname(__file__)
sys.path.append(parent)

from gitstack import CreateCommand, RebaseCommand, git, print_err, run


def add_test_file(filename: str) -> None:
    with open(filename, "w") as f:
        f.write("This is a test file\n")
    run("git", "add", filename)


def run_cmd(args: argparse.Namespace) -> None:
    user = os.environ["USER"]
    test_branch_name = f"{user}-TEST"
    if args.test_cmd == "reset":
        git.switch_branch(git.get_main_branch())
        for branch in git.list_branches():
            if branch.startswith(test_branch_name):
                git.delete_branch(branch, True)
        for branch in git.list_remote_branches():
            if branch.startswith(test_branch_name):
                git.delete_remote_branch(branch)
        run("git", "reset", "--hard", git.get_origin_main())
    elif args.test_cmd == "create":
        git.create_branch(test_branch_name)
        add_test_file("test1.txt")
        git.commit("Test commit 1")
        add_test_file("test2.txt")
        git.commit("Test commit 2")
        add_test_file("test3.txt")
        git.commit("Test commit 3")
        CreateCommand().run(split=True)
    elif args.test_cmd == "rebase_cleanup":
        # rebasing a partially merged stack onto master deletes the merged branches
        git.create_branch(test_branch_name)
        git.commit("Test commit 1")
        git.commit("Test commit 2")
        git.commit("Test commit 3")
        stack = CreateCommand().run(split=True)
        git.switch_branch(git.get_main_branch())
        run("git", "merge", test_branch_name + "-1")
        git.switch_branch(stack.branches()[-1].name)
        stack.rebase(git.get_main_branch())
    elif args.test_cmd == "merge_squash":
        git.create_branch(test_branch_name)
        add_test_file("test1.txt")
        git.commit("Test commit 1")
        add_test_file("test2.txt")
        git.commit("Test commit 2")
        add_test_file("test3.txt")
        git.commit("Test commit 3")
        CreateCommand().run(split=True)
        git.switch_branch(test_branch_name + "-1")
        add_test_file("test4.txt")
        git.commit("Test commit 1b")
        git.switch_branch(git.get_main_branch())
        run("git", "merge", "--squash", test_branch_name + "-1")
        git.commit("Merge " + test_branch_name + "-1")
        git.switch_branch(test_branch_name + "-3")
        RebaseCommand().run(git.get_main_branch())

    # elif args.test_cmd == "tidy_stack":
    #     git.create_branch(test_branch_name)
    #     git.commit("Test commit 1")
    #     git.commit("Test commit 2")
    #     git.commit("Test commit 3")
    #     make_stack()
    #     git.switch_branch(test_branch_name + "-2")
    #     git.commit("Fix up a PR")
    #     git.switch_branch(test_branch_name)
    # elif args.test_cmd == "tidy_rebase":
    #     git.create_branch(test_branch_name, git.get_main_branch() + "^")
    #     git.commit("Test commit 1")
    #     git.commit("Test commit 2")
    #     git.commit("Test commit 3")
    #     make_stack()
    #     git.switch_branch(test_branch_name + "-1")
    #     run("git", "rebase", git.get_main_branch())
    #     git.switch_branch(test_branch_name)
    # elif args.test_cmd == "old_stack":
    #     git.create_branch(test_branch_name, git.get_main_branch() + "^")
    #     git.commit("Test commit 1")
    #     git.commit("Test commit 2")
    #     git.commit("Test commit 3")
    #     make_stack()
    # elif args.test_cmd == "incomplete_stack":
    #     git.create_branch(test_branch_name, git.get_main_branch() + "^")
    #     git.commit("Test commit 1")
    #     git.commit("Test commit 2")
    #     make_stack()
    #     git.commit("Test commit 3")
    #     git.commit("Test commit 4")
    else:
        print_err(f"Unknown test command {args.test_cmd}")
        sys.exit(1)


def main() -> None:
    """Main method"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "test_cmd",
        default="tidy_stack",
        choices=[
            "reset",
            "rebase_cleanup",
            "merge_squash",
            "create",
            # "tidy_stack",
            # "tidy_rebase",
            # "old_stack",
            # "incomplete_stack",
        ],
    )
    args = parser.parse_args()
    run_cmd(args)


if __name__ == "__main__":
    main()
