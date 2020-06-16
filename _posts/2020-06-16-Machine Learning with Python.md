---
layout: post
title: Machine Learning with Python
author: Sangram
categories: [ Computer Science, Python]
---


Open Source Fast Scalable Machine Learning Platform For Smarter Applications: Deep Learning, Gradient Boosting & XGBoost, Random Forest, Generalized Linear Modeling (Logistic Regression, Elastic Net), K-Means, PCA, Stacked Ensembles, Automatic Machine Learning (AutoML), etc. http://h2o.ai
 
Metrics - Machine learning evaluation metrics. Machine learning evaluation metrics, implemented in Python, R, Haskell, and MATLAB / Octave https://github.com/benhamner/Metrics
 
Numenta Platform for Intelligent Computing is an implementation of Hierarchical Temporal Memory (HTM), a theory of intelligence based strictly on the neuroscience of the neocortex. http://numenta.org/
 
scikit-learn - The most popular Python library for Machine Learning. Simple and efficient tools for predictive data analysis. Accessible to everybody, and reusable in various contexts. Built on NumPy, SciPy, and matplotlib. Open source, commercially usable http://scikit-learn.org/
 
Vowpal Wabbit is a machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning. https://vowpalwabbit.org
 
Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library, for Python, R, Java, Scala, C++ and more. Runs on single machine, Hadoop, Spark, Flink and DataFlow https://xgboost.ai/
 
Spark ML - Apache Spark's scalable Machine Learning library. http://spark.apache.org/docs/latest/ml-guide.html

  - [Python](#python)
      - [Computer Vision](#computer-vision-5)
      - [Natural Language Processing](#natural-language-processing-10)
      - [General-Purpose Machine Learning](#general-purpose-machine-learning-21)
      - [Data Analysis / Data Visualization](#data-analysis--data-visualization-9)
      - [Misc Scripts / iPython Notebooks / Codebases](#misc-scripts--ipython-notebooks--codebases)
      - [Neural Networks](#neural-networks)
      - [Kaggle Competition Source Code](#kaggle-competition-source-code)
      - [Reinforcement Learning](#reinforcement-learning-1)


<a name="python-cv"></a>

#### Computer Vision

* [Scikit-Image](https://github.com/scikit-image/scikit-image) - A collection of algorithms for image processing in Python.
* [SimpleCV](http://simplecv.org/) - An open source computer vision framework that gives access to several high-powered computer vision libraries, such as OpenCV. Written on Python and runs on Mac, Windows, and Ubuntu Linux.
* [Vigranumpy](https://github.com/ukoethe/vigra) - Python bindings for the VIGRA C++ computer vision library.
* [OpenFace](https://cmusatyalab.github.io/openface/) - Free and open source face recognition with deep neural networks.
* [PCV](https://github.com/jesolem/PCV) - Open source Python module for computer vision. **[Deprecated]**
* [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library that recognize and manipulate faces from Python or from the command line.
* [dockerface](https://github.com/natanielruiz/dockerface) - Easy to install and use deep learning Faster R-CNN face detection for images and video in a docker container.
* [Detectron](https://github.com/facebookresearch/Detectron) - FAIR's software system that implements state-of-the-art object detection algorithms, including Mask R-CNN. It is written in Python and powered by the Caffe2 deep learning framework. **[Deprecated]**
* [detectron2](https://github.com/facebookresearch/detectron2) - FAIR's next-generation research platform for object detection and segmentation. It is a ground-up rewrite of the previous version, Detectron, and is powered by the PyTorch deep learning framework. 
* [albumentations](https://github.com/albu/albumentations) - А fast and framework agnostic image augmentation library that implements a diverse set of augmentation techniques. Supports classification, segmentation, detection out of the box. Was used to win a number of Deep Learning competitions at Kaggle, Topcoder and those that were a part of the CVPR workshops.
* [pytessarct](https://github.com/madmaze/pytesseract) - Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and "read" the text embedded in images.Python-tesseract is a wrapper for [Google's Tesseract-OCR Engine](https://github.com/tesseract-ocr/tesseract)>.
* [imutils](https://github.com/jrosebr1/imutils) - A library containg Convenience functions to make basic image processing operations such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and Python.
* [PyTorchCV](https://github.com/donnyyou/PyTorchCV) - A PyTorch-Based Framework for Deep Learning in Computer Vision.
* [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt) - A PyTorch implementation of Justin Johnson's neural-style (neural style transfer).
* [Detecto](https://github.com/alankbi/detecto) - Train and run a computer vision model with 5-10 lines of code.
* [neural-dream](https://github.com/ProGamerGov/neural-dream) - A PyTorch implementation of DeepDream.
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - A real-time multi-person keypoint detection library for body, face, hands, and foot estimation
* [Deep High-Resolution-Net](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) - A Pytorch implementation of CVPR2019 paper "Deep High-Resolution Representation Learning for Human Pose Estimation"

<a name="python-nlp"></a>
#### Natural Language Processing

* [pkuseg-python](https://github.com/lancopku/pkuseg-python) - A better version of Jieba, developed by Peking University.
* [NLTK](https://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [Pattern](http://www.clips.ua.ac.be/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
* [Quepy](https://github.com/machinalis/quepy) - A python framework to transform natural language questions to queries in a database query language.
* [TextBlob](http://textblob.readthedocs.io/en/dev/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
* [YAlign](https://github.com/machinalis/yalign) - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora. **[Deprecated]**
* [jieba](https://github.com/fxsjy/jieba#jieba-1) - Chinese Words Segmentation Utilities.
* [SnowNLP](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
* [spammy](https://github.com/tasdikrahman/spammy) - A library for email Spam filtering built on top of nltk
* [loso](https://github.com/fangpenlin/loso) - Another Chinese segmentation library. **[Deprecated]**
* [genius](https://github.com/duanhongyi/genius) - A Chinese segment base on Conditional Random Field.
* [KoNLPy](http://konlpy.org) - A Python package for Korean natural language processing.
* [nut](https://github.com/pprett/nut) - Natural language Understanding Toolkit. **[Deprecated]**
* [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
* [BLLIP Parser](https://pypi.org/project/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser). **[Deprecated]**
* [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](https://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
* [PySS3](https://github.com/sergioburdisso/pyss3) - Python package that implements a novel white-box machine learning model for text classification, called SS3. Since SS3 has the ability to visually explain its rationale, this package also comes with easy-to-use interactive visualizations tools ([online demos](http://tworld.io/ss3/)).
* [python-ucto](https://github.com/proycon/python-ucto) - Python binding to ucto (a unicode-aware rule-based tokenizer for various languages).
* [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
* [python-zpar](https://github.com/EducationalTestingService/python-zpar) - Python bindings for [ZPar](https://github.com/frcchang/zpar), a statistical part-of-speech-tagger, constiuency parser, and dependency parser for English.
* [colibri-core](https://github.com/proycon/colibri-core) - Python binding to C++ library for extracting and working with with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP with Python and Cython.
* [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) - Python interface for converting Penn Treebank trees to Stanford Dependencies.
* [Distance](https://github.com/doukremt/distance) - Levenshtein and Hamming distance computation. **[Deprecated]**
* [Fuzzy Wuzzy](https://github.com/seatgeek/fuzzywuzzy) - Fuzzy String Matching in Python.
* [jellyfish](https://github.com/jamesturk/jellyfish) - a python library for doing approximate and phonetic matching of strings.
* [editdistance](https://pypi.org/project/editdistance/) - fast implementation of edit distance.
* [textacy](https://github.com/chartbeat-labs/textacy) - higher-level NLP built on Spacy.
* [stanford-corenlp-python](https://github.com/dasmith/stanford-corenlp-python) - Python wrapper for [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) **[Deprecated]**
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkit.
* [rasa_nlu](https://github.com/RasaHQ/rasa_nlu) - turn natural language into structured data.
* [yase](https://github.com/PPACI/yase) - Transcode sentence (or other sequence) to list of word vector .
* [Polyglot](https://github.com/aboSamoor/polyglot) - Multilingual text (NLP) processing toolkit.
* [DrQA](https://github.com/facebookresearch/DrQA) - Reading Wikipedia to answer open-domain questions.
* [Dedupe](https://github.com/dedupeio/dedupe) - A python library for accurate and scalable fuzzy matching, record deduplication and entity-resolution.
* [Snips NLU](https://github.com/snipsco/snips-nlu) - Natural Language Understanding library for intent classification and entity extraction
* [NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER) - Named-entity recognition using neural networks providing state-of-the-art-results
* [DeepPavlov](https://github.com/deepmipt/DeepPavlov/) - conversational AI library with many pretrained Russian NLP models.
* [BigARTM](https://github.com/bigartm/bigartm) - topic modelling platform.
