
# Outfit recommender system

This repository is the implementation of an outfit recommender system based on user input images.

## Files description

app.py -> The main file of the application responsible for handling user requests and generating recommendations.

preprocessing.ipynb -> included results of data analysis, histograms, clustering and evaluation of various feature extractor neural networks

embedding_resnet50_resnet18.pt -> extracted feature embeddings from product images by concatenation of ResNet50 and ResNet18 feature vectors

presentation.(pdf/pptx) -> prepared presentation

sample_inputs -> a folder containing the prepared input samples to test the system

## requirements

`python3.8`
`numpy==1.24.3`
`pandas==2.0.3`
`dash==2.16.1`
`dash-core-components==2.0.0`
`dash-html-components==2.0.0`
`Pillow==10.1.0`
`plotly==5.20.0`
`scikit-learn==1.3.2`
`scipy==1.10.1`
`torch==2.1.2`
`torchvision==0.16.2`



## run in python

1- Install requirements:
``` pip install -r requirements.txt ```

2- Run the app: ``` python3.8 app.py```

## run in Docker

1- building images:
```docker build . -t outfit-rs```

2- running the container:
```docker run -p 8080:8080 outfit-rs ```
Access the application at http://127.0.0.1:8080/.