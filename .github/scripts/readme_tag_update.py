#!/usr/bin/env python

"""
Script name:  readme_tag_update.py

Goal:  To determine if a recent push to the "development" branch
       was for a tag, and if so, to update the README.md file on
       the master branch to display the new development tag.


Written by:  Jesse Nusbaumer <nusbaume@ucar.edu> - February, 2020
"""

#+++++++++++++++++++++
#Import needed modules
#+++++++++++++++++++++

#import re
import sys
#import subprocess
#import shlex
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
    parser = argparse.ArgumentParser(description='Close issues and pull requests specified in merged pull request.')

    #Add input arguments to be parsed:
    parser.add_argument('--access_token', metavar='<GITHUB_TOKEN>', action='store', type=str,
                        help="access token used to access GitHub API")

    parser.add_argument('--trigger_sha', metavar='<GITHUB SHA>', action='store', type=str,
                        help="Commit SHA that triggered the workflow")

    #Parse Argument inputs
    args = parser.parse_args()
    return args

#++++++++++++++++++++++++++++++++
#Script message and exit function
#++++++++++++++++++++++++++++++++

def end_script(msg):

    """
    Prints message to screen, and then exits script.
    """
    print("\n{}\n".format(msg))
    print("README tag update script has completed successfully.")
    sys.exit(0)

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

    print("Checking if README file needs to be updated...")

    #+++++++++++++++++++++++
    #Read in input arguments
    #+++++++++++++++++++++++

    args = parse_arguments()

    #Add argument values to variables:
    token = args.access_token
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

    #+++++++++++++++++++++
    #Get list of repo tags
    #+++++++++++++++++++++

    cam_tags = cam_repo.get_tags()

    #+++++++++++++++++++++++++++++++++++++++++++
    #Search for tag with sha that matches commit
    #+++++++++++++++++++++++++++++++++++++++++++

    tag_name = None

    for cam_tag in cam_tags:
        if cam_tag.commit.sha == trigger_sha:
            #If matching tag is found, then extract
            #tag name and tag commit message:
            tag_name = cam_tag.name
            tag_commit = cam_repo.get_commit(trigger_sha)

            #End tag loop:
            break 

    #+++++++++++++++++++++++++++++++++++
    #If no tag matches, then exit script
    #+++++++++++++++++++++++++++++++++++

    if not tag_name:
        endmsg = "No tag was created by this push, so there is nothing to do."
        end_script(endmsg)
    else:
        print("Script found tag name of '{}'".format(tag_name))

    #+++++++++++++++++++++++++++++++++++++++++++++++++
    #Determine which branch contains the tagged commit
    #+++++++++++++++++++++++++++++++++++++++++++++++++

    #Extract tag commit message:
    commit_msg = tag_commit.commit.message

    #CONTINUE HERE!!!!!!!!!!!!!!!

    #++++++++++++++++++++++++++++++++++
    #Upate README file on master branch
    #++++++++++++++++++++++++++++++++++

    print("Script found tag name of '{}'".format(tag_name))

    #++++++++++
    #End script
    #++++++++++

    print("cam_development tag has been successfully updated in README file.")

#############################################

#Run the main script program:
if __name__ == "__main__":
    _main_prog()
