#!/usr/bin/env python
#
#Script name:  ghub_actions_sandbox.py
#
#Goal:  This python script is simply a sandbox to try and test
#       different techniques to be used inside a github actions
#       workflow.
#
#Written by:  Jesse Nusbaumer <nusbaume@ucar.edu> - October, 2020

#+++++++++++++++++++++
#Import needed modules
#+++++++++++++++++++++

from github import Github

import argparse
import sys

#+++++++++++++++++++++++++++++++
#Read in commmand-line arguments
#+++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()
parser.add_argument('--test_env', action='store')
parser.add_argument('--access_token', action='store')
args = parser.parse_args()

#+++++++++++++++++++++++++++
#Print Hello world to screen
#+++++++++++++++++++++++++++

print("Hello World!\n")

print(args.test_env)

#+++++++++++++++++
#Access github API
#+++++++++++++++++

token = args.access_token

ghub = Github(token)

#++++++++++++++++++++
#Open ESCOMP/CAM repo
#++++++++++++++++++++

#Official CAM repo:
#cam_repo = ghub.get_repo("ESCOMP/CAM")

#Test repo:
cam_repo = ghub.get_repo("NCAR/repoForTestingCAM")

#++++++++++++++++++++++++++++++++++++++
#Gather info from most recent merged PR
#++++++++++++++++++++++++++++++++++++++

#Extract all "closed" pull requests, in order of most recently updated first:
closed_pulls = cam_repo.get_pulls(state='closed', sort='updated', direction='desc')

#Loop over closed pull requests:
for pr in closed_pulls:
   #Check that Pull Request was merged:
   if(pr.merged):
     #If so, then pull out PR number and exit loop:
     pr_num = pr.number
     break

print(pr_num)

#++++++++++
#End script
#++++++++++
sys.exit(0)
