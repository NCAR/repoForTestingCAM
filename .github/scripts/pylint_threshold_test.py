#!/usr/bin/env python

"""
Modle name: pylint_threshold_test.py

Purpose:  To test whether the provided list of python
          files pass a pylint check at the specified
          score threshold.

Written by:  Jesse Nusbaumer <nusbaume@ucar.edu> - October, 2020
"""

#+++++++++++++++++++++
#Import needed modules
#+++++++++++++++++++++

import argparse
import io
import pylint.lint as lint

from pylint.reporters.text import TextReporter

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
    parser.add_argument('--python_files', metavar='<comma-separated list>',
                        nargs='+', action='store', type=str,
                        help="list of python files to test")

    parser.add_argument('--rcfile', metavar='<pylintrc file path>', action='store', type=str,
                       help="location of pylintrc file (full path)")

    parser.add_argument('--pylint_level', metavar='<number>', action='store', type=float,
                        required=False, help="pylint score that file(s) must exceed")

    #Parse Argument inputs
    args = parser.parse_args()
    return args

#################
#Main test script
#################

def pylint_check(pyfile_list, rcfile, threshold=10.0):

    """
    Checks if the pylint scores of the provided
    python files are greater than a specified
    threshold.
    """

    #Check if pfylie_list is empty.  If so then exit
    #script, as their are no python files to test:
    if not pyfile_list:
        return

    #Creat empty lists to store pylint output:
    lint_scores = list()
    lint_msgs = list()

    #Create rcfile option string:
    rcstr = '--rcfile={}'.format(rcfile)

    #If files exist, then loop through the list:
    for pyfile in pyfile_list:

        #Create IO object to receive pylint messages:
        pylint_output = io.StringIO()

        #Create pylint reporter object using new IO object:
        pylint_report = TextReporter(pylint_output)

        #Run linter:
        lint_results = lint.Run([rcstr, '--exit-zero', pyfile],
                                reporter=pylint_report, do_exit=False)

        #Extract linter score:
        lint_score = lint_results.linter.stats['global_note']

        #Save pylint output as string:
        lint_msg = pylint_output.getvalue()

        #Close IO object:
        pylint_output.close()

        #Add file score and message to list if
        #below pylint threshold:
        if lint_score < threshold:
            lint_scores.append(lint_score)
            lint_msgs.append(lint_msg)

    for idx, score in enumerate(lint_scores):
        print("File #{}".format(idx))
        print("Pylint score = {}".format(score))
        print("Pylint message:")
        print(lint_msgs[idx])

    #Return plyint lists:
    return lint_score, lint_msgs

####################
#Command-line script
####################

def _pylint_check_commandline():

    """
    Runs the "pylint_check" test using
    command line inputs.
    """

    #Read in command-line arguments:
    args = parse_arguments()

    #Add argument values to variables:
    python_files = args.python_files
    pylintrc = args.rcfile
    pylint_level = args.pylint_level

    #run pylint threshold check:
    if pylint_level:
        scores, msgs = pylint_check(python_files, pylintrc,
                                    threshold=pylint_level)
    else:
        scores, msgs = pylint_check(python_files, pylintrc)

#############################################

#Run main script using provided command line arguments:
if __name__ == "__main__":
    _pylint_check_commandline()

