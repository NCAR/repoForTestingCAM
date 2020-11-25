#!/usr/bin/env python

"""
Script name:  pr_mod_file_tests.py

Goal:  To generate a list of files modified in the associated
       Github Pull Request (PR), using the PyGithub interface,
       and then to run tests on those files when appropriate.

Written by:  Jesse Nusbaumer <nusbaume@ucar.edu> - September, 2020
"""

#+++++++++++++++++++++
#Import needed modules
#+++++++++++++++++++++

import sys
import os
import subprocess
import shlex
import argparse

from stat import S_ISREG
from github import Github

#################
#HELPER FUNCTIONS
#################

def _file_is_python(filename):

    """
    Checks whether a given file
    is a python script or
    python source code.
    """

    #Initialize return logical:
    is_python = False

    #Extract status of provided file:
    file_stat = os.stat(filename)

    #Check if it is a "regular" file:
    if S_ISREG(file_stat.st_mode):

        #Next, check if file ends in ".py":
        file_ext = os.path.splitext(filename)[1]

        if file_ext.strip() == ".py":
            #Assume file is python:
            is_python = True
        else:
            #If no ".py" extension exists, then
            #open the file and look for a shabang
            #that contains the word "python".
            with open(filename, "r") as mod_file:
                #Read first line of file:
                first_line = mod_file.readline()

                #Check that first line is a shabang:
                if first_line[0:1] == '#!':
                    #If so, then check that the word
                    #"python" is also present:
                    if first_line.find("python") != -1:
                        #If the word exists, then assume
                        #it is a python file:
                        is_python = True


    #Return file type result:
    return is_python

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

    print("Generating list of modified files...")

    #+++++++++++++++++++++++
    #Read in input arguments
    #+++++++++++++++++++++++

    args = parse_arguments()

    #Add argument values to variables:
    token = args.access_token
    pr_num = args.pr_num

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

    #++++++++++++++++++++++++++++++
    #Extract list of modified files
    #++++++++++++++++++++++++++++++

    #Create empty list to store python files:
    pyfiles = list()

    #Extract Github file objects:
    file_obj_list = pull_req.get_files()

    for file_obj in file_obj_list:

        print(file_obj.filename)
        #Check if it is a python file:
        if _file_is_python(file_obj.filename):
            #If so, then add to python list:
            pyfiles.append(file_obj.filename)

        #CONTINUE HERE!!!  NEED TO TEST
        #"_file_is_python" function!  IF it
        #works then bring in "pylint" script.
        #IF that works, THEN use pylint lists
        #to either generate failue or a successful
        #system exit!
        #GOOD LUCK!!!!!!!!!!!!!!


#############################################

#Run the main script program:
if __name__ == "__main__":
    _main_prog()
