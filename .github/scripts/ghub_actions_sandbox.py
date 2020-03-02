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
#parser.add_argument('--token', action='store')
parser.add_argument('--test_env', action='store')
args = parser.parse_args()

#+++++++++++++++++++++++++++
#Print Hello world to screen
#+++++++++++++++++++++++++++

print("Hello World!\n")

print(args.test_env)

#+++++++++++++++++
#Access github API
#+++++++++++++++++

#ghub = Github(token)

#++++++++++++++++++++
#Open ESCOMP/CAM repo
#++++++++++++++++++++

#Official CAM repo:
#cam_repo = ghub.get_repo("ESCOMP/CAM")

#Test repo:
#cam_repo = ghub.get_repo("cacraigucar/forJessetesting")

#++++++++++
#End script
#++++++++++
sys.exit(0)
