#!/bin/bash

pandoc -t beamer -s -fmarkdown-implicit_figures --template=template.beamer slides_fermilab_keras_workshop.md -o slides_fermilab_keras_workshop.pdf
