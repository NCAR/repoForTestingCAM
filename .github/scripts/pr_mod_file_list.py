#!/usr/bin/env python

"""
Script name:  pr_mod_file_list.py

Goal:  To generate a list of files modified in the associated 
       Github Pull Request (PR), using the PyGithub interface.

Written by:  Jesse Nusbaumer <nusbaume@ucar.edu> - September, 2020
"""

#+++++++++++++++++++++
#Import needed modules
#+++++++++++++++++++++

import sys
import subprocess
import shlex
import argparse

from github import Github

#################
#HELPER FUNCTIONS
#################

#++++++++++++++++++++++++++++++
#Input Argument parser function
#++++++++++++++++++++++++++++++

def parse_arguments():

    """
    Parses command-line input arguments using the argparse
    python module and outputs the final argument object.
    """

    #Create parser object:
    parser = argparse.ArgumentParser(description='Generate list of all files modified by pull request.')

    #Add input arguments to be parsed:
    parser.add_argument('--access_token', metavar='<GITHUB_TOKEN>', action='store', type=str,
                        help="access token used to access GitHub API")

    parser.add_argument('--pr_num', metavar='<PR_NUMBER>', action='store', type=int,
                        help="pull request number")

    parser.add_argument('--trigger_sha', metavar='<GITHUB SHA>', action='store', type=str,
                        help="Commit SHA that triggered the workflow")

    #Parse Argument inputs
    args = parser.parse_args()
    return args

#############
#MAIN PROGRAM
#############

def _main_prog():

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    #++++++++++++
    #Begin script
    #++++++++++++

    print("Checking if issue needs to be closed...")

    #+++++++++++++++++++++++
    #Read in input arguments
    #+++++++++++++++++++++++

    args = parse_arguments()

    #Add argument values to variables:
    token = args.access_token
    pr_num = args.pr_num
    trigger_sha = args.trigger_sha

    #++++++++++++++++++++++++++++++++
    #Log-in to github API using token
    #++++++++++++++++++++++++++++++++

    ghub = Github(token)

    #++++++++++++++++++++
    #Open ESCOMP/CAM repo
    #++++++++++++++++++++

    #Official CAM repo:
    cam_repo = ghub.get_repo("NCAR/repoForTestingCAM")

    #++++++++++++++++++++++++++++++++++++++++++
    #Open Pull Request which triggered workflow
    #++++++++++++++++++++++++++++++++++++++++++

    pull_req = cam_repo.get_pull(pr_num)

    #++++++++++++++++++++++++
    #Extract merge commit SHA
    #++++++++++++++++++++++++

    merge_commit = pull_req.merge_commit_sha

    print(merge_commit)
    print(trigger_sha)

    #+++++++++++++++++++++++++++
    #Gather output from git diff
    #+++++++++++++++++++++++++++

    #Create Git Diff command string:
    git_diff_cmd = "git diff-tree --no-commit-id --name-only -r {}".format(merge_commit)

    #Split command line string into argument list:
    diff_arg_list = shlex.split(git_diff_cmd)

    #Run command using subprocess:
    file_diff_str = subprocess.check_output(diff_arg_list)

    print(file_diff_str)

    print(file_diff_str.split())

#############################################

#Run the main script program:
if __name__ == "__main__":
    _main_prog()
