from __future__ import annotations

from pathlib import Path

from git import Repo, cmd


def lsremote(url):
    """
    only one way for
    git ls-remote -t --exit-code --refs {url} | sed -E 's/^[[:xdigit:]]+[[:space:]]+refs\\/tags\\/(.+)/\1/g'
    """
    g = cmd.Git()
    refs = g.ls_remote("-t", "--exit-code", url).split("\n")
    return refs[-1].split("\t")[-1].split("/")[-1]


def test_submodules():
    """
    find tag in current repo and check tag in remote submodule
    """
    repo_path = Path(__file__).resolve().parent.parent
    repo = Repo(repo_path)  # init class Repo
    assert not repo.bare  # checking if repo is bare

    for module in repo.submodules:
        sub_repo = module.module()  # init class Repo for submodule
        assert not sub_repo.bare
        current_tag = next(
            (tag for tag in sub_repo.tags if tag.commit == sub_repo.head.commit), None
        )  # find current tag in local repo

        assert current_tag is not None, f"You don't use tag for '{module.name}'!"

        current_tag = current_tag.name  # git.Tag to str
        remote_tag = lsremote(module.url)  # find last tag in remote module

        assert current_tag == remote_tag, f"Local submodule tag {current_tag} does not match remote {remote_tag}"


if __name__ == "__main__":
    test_submodules()
