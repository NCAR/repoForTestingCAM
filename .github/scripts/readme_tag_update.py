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

import re
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

    #+++++++++++++++++++++++++++++++
    #Search for associated PR number
    #+++++++++++++++++++++++++++++++

    #Extract tag commit message:
    commit_message = tag_commit.commit.message

    #Compile Pull Request merge text expression:
    pr_merge_pattern = re.compile(r'Merge pull request ')

    #Search for merge text, starting at beginning of message:
    commit_msg_match = pr_merge_pattern.match(commit_message)

    #Check if match exists:
    if commit_msg_match is not None:
        #If it does then pull out text immediately after message:
        post_msg_text = commit_message[commit_msg_match.end():]

        #Split text into individual words:
        post_msg_word_list = post_msg_text.split()

        #Extract first word:
        first_word = post_msg_word_list[0]

        #Print merged pr number to screen:
        print("Merged PR associated with tag: {}".format(first_word))

        try:
            #Try assuming the word is just a number:
            pr_num = int(first_word[1:]) #ignore "#" symbol
        except ValueError:
            #If the conversion fails, then this is likely not a real PR merge, so end the script:
            endmsg = "No Pull Request number was found in the tagged commit message. "
            endmsg += "This is likely a special commit, so the script will not modify README."
            end_script(endmsg)

    else:
        endmsg = "No Pull Request merges were found in the tagged commit message. "
        endmsg += "This is likely a special commit, so the script will not modify README."
        end_script(endmsg)

    #+++++++++++++++++++++++++++++++++++++
    #Check if tagged PR was in fact merged
    #+++++++++++++++++++++++++++++++++++++

    #Extract pull request info:
    merged_pull = cam_repo.get_pull(pr_num)

    #If pull request has not been merged, then exit script:
    if not merged_pull.merged:
        endmsg = "Pull request in commit message was not actually merged, so the script will not close anything."
        end_script(endmsg)

    #++++++++++++++++++++++++++++++++++++++++++
    #Check if PR was for the development branch
    #++++++++++++++++++++++++++++++++++++++++++

    #Extract merged branch from latest Pull request:
    merged_branch = merged_pull.base.ref

    print("Script found branch name of '{}'".format(merged_branch))

    #If PR is not to the development branch, then exit script:
    if merged_branch == "fake_development":
        endmsg = "Tagged PR merged into non-development branch. No further action will thus be taken."
        end_script(endmsg)

    #+++++++++++++++++++++++++++++++++++++++++++++++++
    #Determine which branch contains the tagged commit
    #+++++++++++++++++++++++++++++++++++++++++++++++++

    #CONTINUE HERE!!!!!!!!!!!!!!!

    #IMPLEMENT THE SAME PR NUMBER EXTRACTION AS IS DONE IN THE ISSUE-CLOSING SCRIPT.
    #THEN USE THAT PR TO CHECK IF THE RECEIVING BRANCH WAS THE DEVELOPMENT BRANCH.
    #IF SO THEN BEGIN README MODIFICATIONS.  OTHERWISE EXIT SCRIPT!
    #GOOD LUCK!!!!

    #++++++++++++++++++++++++++++++++++
    #Upate README file on master branch
    #++++++++++++++++++++++++++++++++++

    #++++++++++
    #End script
    #++++++++++

    print("cam_development tag has been successfully updated in README file.")

#############################################

#Run the main script program:
if __name__ == "__main__":
    _main_prog()
