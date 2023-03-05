# PMTProcessor
## Hi, welcome to PMTProcessor!
PMTProcessor is a software which provides support for processing and analysing data from SNO+ experiment.

To use this software, simply open your terminal window and run the below command:

```
git clone https://github.com/SunPeike/Bad_PMT_Detection.git
```

once downloaded, you can type
```
cd Bad_PMT_Detection
```
in the command line, then type
```
ls
```
then you can have a look of what is included in this folder.

There are several functions which help you to read, clean and visualize the data. According to SNO+ data disclosure regulations, I am not allowed to provide the experimental data, which means you need to use your own data sets. The easiest way of import your data is type the below codes in your terminal:
```
jupyter notebook
```
then you can click "example.ipynb", where I left you a block to fill in your folder path. My local path of data folder is:
```
folderpath = "/Users/peikesun/Documents/Year3/project/inputs"
```
you should replace it with your own path
