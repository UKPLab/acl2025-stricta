## STRICTA DATASET

The provided files under data are just mock files to show the structure of the dataset. Download the complete dataset
from [tudatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4614).

>> For detailed field descriptions of the files check the README of the dataset.

## Dataset Structure
The standard structure looks as follows:

```
raw                                             # the raw data incl. annotations and papers
├── main_study
│   ├── paper_{x}
│   │   ├── media
│   │   ├── paper.itg.json
│   │   ├── annotations_in_out.json
│   │   └── ...
│   ├── annotations_in_out.json
│   ├── meta.json
│   ├── workflow.json
│   ├── extraction_annotations.csv
│   ├── inference_annotations.csv
│   └── ...
├── student_seminar
│   ├── ...
│   └── ...

annotations_language_corrected                  # the annotations with language error correction (typos, capitalization, ...)
├── main_study
│   └── ...
├── student_seminar
│   └── ...

```

## Raw vs. Annotations Language Corrected

The raw data contains the data as provided by the annotators; however, some textual answers contain typos, small
grammar mistakes, etc. The annotations language corrected contains the same data, but with these errors corrected. If
you use the default data loading pipelines, you should generally opt for the language corrected version; if you want
to replicate the results in the paper you should choose the raw version. Additionally, the raw data contains
the raw paper xml files plus a visualization of selected spans in the paper.

## Main vs. Student Seminar

Following the data collection, we have two subsets (junior and senior). The main_study folder refers to the senior dataset
and the student_seminar to the junior dataset. Choose either or both for your experiments. For replication, you should
choose both.

## Paper Directories

* Each paper directory contains a directory "media" which contains the figures of the paper. 
* The paper.itg.json is a structured representation of the paper following the ITG format
* annotations_in_out.json lists the inputs and outputs to each step of the workflow execution for each paper
* meta.json lists meta-information on the paper including authors, links etc.

## workflow.json
The workflow.json file contains the workflow definition for the paper. It is a JSON representation of the workflow.

# meta.json
The meta.json file contains meta-information on the dataset.

# extraction_annotations.csv
The extraction_annotations.csv file contains the annotations for the extraction steps.

# inference_annotations.csv
The inference_annotations.csv file contains the annotations for the inference steps.