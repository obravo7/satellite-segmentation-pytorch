

## Introduction

The following task seeks to emulate some real world scenario that you might encounter at Incubit. The goal is to create a model that takes an image as input and outputs vectorized segmentations of instances of objects inside the image.

## Problem description

You are given a small slice of satellite data of a city in Japan by a fantasy land surveying company. They want to evaluate the feasibility of using satellite data to augment some downstream tasks. To that end, they need to segment out individual buildings, by category. They also need to return the segmentation in vector format so that they can import it into their CAD software.

They had some of their interns annotate some of the data and would like you to have a go at it. They're not data scientists so the data they provided might not be optimal and their annotations not entirely consistent. But, it is what it is, and you have to make due with it.


## The Data:
The data consists of one single sattelite picture of Tokyo split into a 9x9 grid (81 non-overlapping PNG images total). The naming convention reflects it. You can put it back together if you want to.

In addition, you receive 72 annotation data files containing target labels for individual images. 9 of them, forming the bottom right corner of the image, are kept away for evaluation. So no need to panic if you can't find them. 

The data is picked in a way that will alleviate your model training time, while still being consistent enough to have a good chance of yielding reasonable output on the test data.

## The Labels:
The target data consists of three labels that are of interest in the image:
1. Houses
2. Buildings
3. Sheds/Garages

The labels come from our internal annotation tools. The format doesn't follow any other academic data format, but it's pretty straightforward. 

* The label data is provided in json format, one json annotation file per image, named after the image it represents. 

* there is plenty of verbose metadata in the annotation files, produced by our annotation tools, that is not relevant to the task. Feel free to navigate around it.

* The labels are provided as polygons under an [x,y,x,y,x,y....] format. Once the sequence is finished, the last point connects to the first point. There is no distinction between clockwise and conterclockwise.

* One polygon defines the perimeter of a building unit.


## The Task

Your mission is to create a model that can take an image as input and output the individual vector polygon detections of each of the individual buildings according to their class in [x,y,x,y,x,y,....] format.

You are free to use any pretrained models, any external data sources, any programming language, DL framework, or other resources that you see fit. 

What we'd like to see:
* The code, preferably in github form
* A report describing your problem analysis, approach, results, conclusions, hurdles, ideas. It can be in pdf, readme markdown, ipynb form, a mix thereof, or whatever you feel is good at conveying information.


## The output

We would like your code to be able to output the results in the following json format:

```
{'filename':file_name, 
{'labels': [{'name': label_name, 'annotations': [{'id':some_unique_integer_id, 'segmentation':[x,y,x,y,x,y....]}
                                             ....] }
        ....]
}
```

with .... standing for an indeterminate number of elements of the same type in the json structure

## The Evaluation

What we're looking for is
- Your analysis and understanding of the provided problem, challenge, data and results
- Quality and creativity of the solution
- Thought process and ability to convey it to us in words, tables, plots and/or visualizations
- Readability and usability of the code

## The Timeline

There is no hard deadline for the task but you should not spend more than 2 weeks of occasionally working on it, fitting it around your schedule. Less if you have more free time and put in more concerted effort.


