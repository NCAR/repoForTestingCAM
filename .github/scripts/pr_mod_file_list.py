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

    parser.add_argument('--base_ref', metavar='<GITHUB_BASE_REF>', action='store', type=str,
                        help="pull request target branch")
    
    parser.add_argument('--head_ref', metavar='<GITHUB_HEAD_REF>', action='store', type=str,
                        help="Pull request source branch")

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
    base_ref = args.base_ref
    head_ref = args.head_ref

    print(base_ref)
    print(head_ref)

    #++++++++++++++++++++++++++++++++
    #Log-in to github API using token
    #++++++++++++++++++++++++++++++++

    ghub = Github(token)

    #++++++++++++++++++++
    #Open ESCOMP/CAM repo
    #++++++++++++++++++++

    #Official CAM repo:
    cam_repo = ghub.get_repo("NCAR/repoForTestingCAM")

    print(list(cam_repo.get_branches()))

#############################################

#Run the main script program:
if __name__ == "__main__":
    _main_prog()
