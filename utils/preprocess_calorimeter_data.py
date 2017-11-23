#!/usr/bin/env python

import wget
import os
import ROOT

# Download ROOT file with toy calorimeter data
filename = "testDataReg.root"
if not os.path.exists(filename):
    wget.download("https://www.hep1.physik.uni-bonn.de/people/homepages/tmva/testDataReg.root")

# Read out events and write to CSV file
output_file = open("toy_calorimeter.csv", "w")
for i in range(13):
    output_file.write("e{} ".format(i))
output_file.write("etruth\n")

file_ = ROOT.TFile(filename)
tree = file_.Get("TreeR")
for event in tree:
    for i in range(13):
        output_file.write("{} ".format(getattr(event, "e{}".format(i))))
    output_file.write("{}\n".format(getattr(event, "etruth".format(i))))
