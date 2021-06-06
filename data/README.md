# Training data

## IAM Dataset

To train with dataset you must put the files in this structure:

```
|-- data
    |-- words.txt
    |-- words
        |-- a01-000u-00-00.png
        |-- a01-000u-00-01.png
        |-- a01-000u-00-02.png
        |-- ...
```

Where `words.txt` is the file with labels provided in the dataset, and words folder contains all word images in the dataset.

## Custom data for training

Custom data must follow the structure specified above. All images should be put in words folder. Custom words.txt should contain each file name (without extension) in first column and corresponding label in the last column (with possible other columns in between). Columns should be separated by a space. Example from the IAM words.txt file:

```
a01-000u-00-01 ok 154 507 766 213 48 NN MOVE
```
