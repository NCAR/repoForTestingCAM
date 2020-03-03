#!/usr/bin/env python
#
#Script name:  branch_PR_issue_closer.py
#
#Goal:  To check if the newly-merged PR's commit message attempted to close an issue.
#       If so, then move the associated project card to the "closed issues" column.
#
#       Also checks if the newly-merged PR is the final PR needed to fix the issue
#       for all related branches.  If so, then the issue is formally closed.
#
#Written by:  Jesse Nusbaumer <nusbaume@ucar.edu> - October, 2019
#
#+++++++++++++++++++++
#Import needed modules
#+++++++++++++++++++++

from github import Github
from github import Issue
from datetime import datetime

import re
import sys
import os
import subprocess
import shlex
import argparse

#################
#HELPER FUNCTIONS
#################

#+++++++++++++++++++++++++++++++++++++++++
#Curl command needed to move project cards
#+++++++++++++++++++++++++++++++++++++++++

def  project_card_move(OA_token, column_id, card_id):

    """
    Currently pyGithub doesn't contain the methods required
    to move project cards from one column to another, so
    the unix curl command must be called directly, which is
    what this function does.

    The specific command-line call made is:

    curl -H "Authorization: token OA_token" -H \
    "Accept: application/vnd.github.inertia-preview+json" \
    -X POST -d '{"position":"top", "column_id":<column_id>}' \
    https://api.github.com/projects/columns/cards/<card_id>/moves

    """

    #create required argument strings from inputs:
    github_OA_header = ''' "Authorization: token {0}" '''.format(OA_token)
    github_url_str   = '''https://api.github.com/projects/columns/cards/{0}/moves'''.format(card_id)
    json_post_inputs = ''' '{"position":"top", "column_id":%i}' ''' %column_id  #format() can't be used due to curly brackets in string.

    #Create curl command line string:
    curl_cmdline = '''curl -H '''+github_OA_header+''' -H "Accept: application/vnd.github.inertia-preview+json" -X POST -d '''+\
                   json_post_inputs+''' '''+github_url_str

    #Split command line string into argument list:
    curl_arg_list = shlex.split(curl_cmdline)

    #Run command using subprocess:
    subprocess.run(curl_arg_list)

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

    parser.add_argument('--pull_num', metavar='<pull request number>', action='store', type=int,
                        help="Number of pull request that has been merged")

    #Parse Argument inputs
    args = parser.parse_args()
    return args

#############
#MAIN PROGRAM
#############

def _main_prog():

    #++++++++++++
    #Begin script
    #++++++++++++

    print("Checking if issue needs to be closed...")

    #+++++++++++++++++++++++
    #Read in input arguments
    #+++++++++++++++++++++++

    args = parse_arguments()

    #Add argument values to variables:
    token  = args.access_token
    pr_num = args.pull_num

    print(token)
    sys.exit(0)

    #++++++++++++++++++++++++++++++++
    #Log-in to github API using token
    #++++++++++++++++++++++++++++++++

    ghub = Github(token)

    #++++++++++++++++++++
    #Open ESCOMP/CAM repo
    #++++++++++++++++++++

    #Official CAM repo:
    #cam_repo = ghub.get_repo("ESCOMP/CAM")

    #Test repo:
    cam_repo = ghub.get_repo("NCAR/repoForTestingCAM")

    #++++++++++++++++++++++++++++++++++++++
    #Create integer list of all open issues:
    #++++++++++++++++++++++++++++++++++++++

    #create new list:
    open_issues = list()

    #Extract list of open issues from repo:
    open_repo_issues = cam_repo.get_issues(state='open')

    #Loop over all open repo issues:
    for issue in open_repo_issues:
        #Add issue number to "open_issues" list:
        open_issues.append(issue.number)

    #+++++++++++++++++++++++++++++++++++++++++++++
    #Create integer list of all open pull requests
    #+++++++++++++++++++++++++++++++++++++++++++++

    #create new list:
    open_pulls = list()

    #Extract list of open PRs from repo:
    open_repo_pulls = cam_repo.get_pulls(state='open')

    #Loop over all open repo issues:
    for pr in open_repo_pulls:
        #Add pr number to "open_pulls" list:
        open_pulls.append(pr.number)

    #+++++++++++++++++++++++++++++++++++++
    #Check that PR has in fact been merged
    #+++++++++++++++++++++++++++++++++++++

    #Extract pull request info:
    merged_pull = cam_repo.get_pull(pr_num)

    #If pull request has not been merged, then exit script:
    if not merged_pull.merged:
        print("\n")
        print("Pull request was not merged, so the script will not close anything.")
        print("\n")
        print("Issue closing check has completed successfully.")
        sys.exit(0)

    #++++++++++++++++++++++++++++++++++++++++
    #Check that PR was not for default branch
    #++++++++++++++++++++++++++++++++++++++++

    #Determine default branch on repo:
    default_branch = cam_repo.default_branch

    #Extract merged branch from latest Pull request:
    merged_branch = merged_pull.base.ref

    #If PR was to default branch, then exit script (as github will handle it automatically):
    if merged_branch == default_branch:
        print("\n")
        print("Pull request ws merged into default repo branch. Thus issue is closed automatically")
        print("\n")
        print("Issue closing check has completed successfully.")
        sys.exit(0)

    #+++++++++++++++++++++++++++++++++++++++++++++++++
    #Check if one of the keywords exists in PR message
    #+++++++++++++++++++++++++++++++++++++++++++++++++

    #Keywords are:
    #close, closes, closed
    #fix, fixes, fixed
    #resolve, resolves, resolved

    #Create regex pattern to find keywords:
    keyword_pattern = re.compile(r'(^|\s)close(\s|s\s|d\s)|(^|\s)fix(\s|es\s|ed\s)|(^|\s)resolve(\s|s\s|d\s)')

    #Extract Pull Request message:
    pr_message = merged_pull.body

    #Make entire message lower-case:
    pr_msg_lower = pr_message.lower()

    #search for at least one keyword:
    if keyword_pattern.search(pr_msg_lower) is not None:
        #If at least one keyword is found, then determine location of every keyword instance:
        word_matches = keyword_pattern.finditer(pr_msg_lower)
    else:
        #If no keyword is found, then exit script (as there is nothing more to do):
        print("\n")
        print("Pull request was merged without using any of the keywords.  Thus there are no issues to close.")
        print("\n")
        print("Issue closing check has completed successfully.")
        sys.exit(0)

   #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   #Extract issue and PR numbers associated with found keywords in merged PR message
   #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #create issue pattern ("the number symbol {#} + a number"),
    #which ends with either a space or a comma:
    issue_pattern = re.compile(r'#[0-9]+(\s|,|$)|.')

    #Create new "close" issues list:
    close_issues = list()

    #Create new "closed" PR list:
    close_pulls = list()

    #Search text right after keywords for possible issue numbers:
    for match in word_matches:

        #create temporary string starting at end of match:
        tmp_msg_str = pr_msg_lower[match.end():]

        #Check if first word matches issue pattern:
        if issue_pattern.match(tmp_msg_str) is not None:

            #If so, then split string into words:
            tmp_word_list = tmp_msg_str.split()

            #Extract first word:
            first_word = tmp_word_list[0]

            #Extract issue number from first word:
            try:
                #First try assuming the string is just a number
                issue_num = int(first_word[1:]) #ignore "#" symbol
            except ValueError:
                #If not, then ignore last letter:
                try:
                    issue_num = int(first_word[1:len(first_word)-1])
                except ValueError:
                    #If ignoring the first and last letter doesn't work,
                    #then the match was likely a false positive,
                    #so set the issue number to one that will never be found:
                    issue_num = -9999

            #Check that number is actually for an issue (as opposed to a PR):
            if issue_num in open_issues:
                #Add issue number to "close issues" list:
                close_issues.append(issue_num)
            elif issue_num in open_pulls:
                #If in fact a PR, then add to PR list:
                close_pulls.append(issue_num)

    #If no issue numbers are present after any of the keywords, then exit script:
    if not close_issues and not close_pulls:
        print("\n")
        print("No issue or PR numbers were found in the merged PR message.  Thus there is nothing to close.")
        print("\n")
        print("Issue closing check has completed successfully.")
        sys.exit(0)

    #++++++++++++++++++++++++++++++++++++++++
    #Extract repo project "To do" card issues
    #++++++++++++++++++++++++++++++++++++++++

    #Intialize project issue number list:
    proj_issues = list()

    #Initalize project issue count list:
    proj_issue_count = list()

    #Initalize issue id to project card id dictionary:
    proj_issue_card_ids = dict()

    #First, Pull-out all projects from repo:
    projects = cam_repo.get_projects()

    #Also determine relevant project and column name for card move:
    if merged_branch  == "cam_development":
        proj_mod_name = "CAM Development branch (cutting edge development)"
        col_mod_name = "closed issues"
    else:
        proj_mod_name = "CAM Public Release for CESM2_1"
        col_mod_name = "Completed issues"

    #Loop over all repo projects:
    for project in projects:

        #Next, pull-out columns from each project:
        proj_columns = project.get_columns()

        #Loop over columns:
        for column in proj_columns:
            #Check if column name is "To do"
            if column.name == "To do":
                #If so, then extract cards:
                cards = column.get_cards()

                #Loop over cards:
                for card in cards:
                    card_content = card.get_content()

                    #Check that card content is an "issue":
                    if isinstance(card_content, Issue.Issue):

                        #Next, check if card issue number matches any of the "close" issue numbers from the PR:
                        if card_content.number in close_issues:

                          #If so, then check if issue number is already in proj_issues:
                          if card_content.number in proj_issues:
                              #If it is already present, then extract index of issue:
                              iss_idx = proj_issues.index(card_content.number)

                              #Add one to project issue counter:
                              proj_issue_count[iss_idx] += 1

                          else:
                              #If not, then append to project issues and counter list:
                              proj_issues.append(card_content.number)
                              proj_issue_count.append(1)

                          #Also add issue id and card id to id dictionary used for card move, if in relevant project:
                          if project.name == proj_mod_name:
                              proj_issue_card_ids.update({card_content.number:card.id})

            #Otherwise, check if column name matches "modified/closed issues" column:
            elif column.name == col_mod_name and project.name == proj_mod_name:
                #If so, then save column id:
                column_id = column.id

    #If no project cards are found that match the issue, then exit script:
    if not proj_issues:
        print("\n")
        print("ERROR:  No project cards matched the issue being closed.")
        print("Either the number in the PR message is wrong, or the project cards have been configured in-correctly.")
        sys.exit(0)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Check if the number of "To-do" project cards matches the total number
    #of merged PRs for each 'close' issue.
    #
    #Then, close all issues for which project cards equals merged PRs
    #
    #If not, then simply move the project card to the relevant project's
    #"closed issues" column.
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #Loop over project issues that have been "closed" by merged PR:
    for issue_num in proj_issues:

        #Determine list index:
        iss_idx = proj_issues.index(issue_num)

        #determine project issue count:
        issue_count = proj_issue_count[iss_idx]

        #if issue count is just one, then close issue:
        if issue_count == 1:
          #Extract github issue object:
          cam_issue = cam_repo.get_issue(number=issue_num)
          #Close issue:
          cam_issue.edit(state='closed')
          print("Issue #{} has been closed.".format(issue_num))
        else:
            #Extract card id from id dictionary:
            card_id = proj_issue_card_ids[issue_num]

            #Then move the card on the relevant project page to the "Modified/Completed Issues" column:
            project_card_move(access_token.strip(), column_id, card_id)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Finally, close all Pull Requests in "close_pulls" list:
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for pull_num in close_pulls:
        #Extract Pull request object:
        cam_pull = cam.repo.get_pull(number=pull_num)

        #Close Pull Request:
        cam_pull.edit(state='closed')
        print("Pull Request #{} has been closed.".format(pull_num))

    #++++++++++
    #End script
    #++++++++++

    print("Issue closing check has completed successfully.")

#############################################

#Run the main script program:
if __name__ == "__main__":
    _main_prog()

