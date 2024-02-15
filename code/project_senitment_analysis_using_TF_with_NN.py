{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ic4_occAAiAT"
   },
   "source": [
    "##### Copyright 2019 The TensorFlow Authors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "cellView": "form",
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:24.336322Z",
     "iopub.status.busy": "2023-12-07T03:08:24.336072Z",
     "iopub.status.idle": "2023-12-07T03:08:24.340041Z",
     "shell.execute_reply": "2023-12-07T03:08:24.339379Z"
    },
    "id": "ioaprt5q5US7"
   },
   "outputs": [],
   "source": [
    "#@title Licensed under the Apache License, Version 2.0 (the \"License\");\n",
    "# you may not use this file except in compliance with the License.\n",
    "# You may obtain a copy of the License at\n",
    "#\n",
    "# https://www.apache.org/licenses/LICENSE-2.0\n",
    "#\n",
    "# Unless required by applicable law or agreed to in writing, software\n",
    "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
    "# See the License for the specific language governing permissions and\n",
    "# limitations under the License."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "cellView": "form",
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:24.343087Z",
     "iopub.status.busy": "2023-12-07T03:08:24.342827Z",
     "iopub.status.idle": "2023-12-07T03:08:24.346243Z",
     "shell.execute_reply": "2023-12-07T03:08:24.345664Z"
    },
    "id": "yCl0eTNH5RS3"
   },
   "outputs": [],
   "source": [
    "#@title MIT License\n",
    "#\n",
    "# Copyright (c) 2017 François Chollet\n",
    "#\n",
    "# Permission is hereby granted, free of charge, to any person obtaining a\n",
    "# copy of this software and associated documentation files (the \"Software\"),\n",
    "# to deal in the Software without restriction, including without limitation\n",
    "# the rights to use, copy, modify, merge, publish, distribute, sublicense,\n",
    "# and/or sell copies of the Software, and to permit persons to whom the\n",
    "# Software is furnished to do so, subject to the following conditions:\n",
    "#\n",
    "# The above copyright notice and this permission notice shall be included in\n",
    "# all copies or substantial portions of the Software.\n",
    "#\n",
    "# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n",
    "# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n",
    "# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL\n",
    "# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n",
    "# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING\n",
    "# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER\n",
    "# DEALINGS IN THE SOFTWARE."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ItXfxkxvosLH"
   },
   "source": [
    "# Basic text classification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hKY4XMc9o8iB"
   },
   "source": [
    "<table class=\"tfo-notebook-buttons\" align=\"left\">\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://www.tensorflow.org/tutorials/keras/text_classification\"><img src=\"https://www.tensorflow.org/images/tf_logo_32px.png\" />View on TensorFlow.org</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/text_classification.ipynb\"><img src=\"https://www.tensorflow.org/images/colab_logo_32px.png\" />Run in Google Colab</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/text_classification.ipynb\"><img src=\"https://www.tensorflow.org/images/GitHub-Mark-32px.png\" />View source on GitHub</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a href=\"https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/keras/text_classification.ipynb\"><img src=\"https://www.tensorflow.org/images/download_logo_32px.png\" />Download notebook</a>\n",
    "  </td>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Eg62Pmz3o83v"
   },
   "source": [
    "This tutorial demonstrates text classification starting from plain text files stored on disk. You'll train a binary classifier to perform sentiment analysis on an IMDB dataset. At the end of the notebook, there is an exercise for you to try, in which you'll train a multi-class classifier to predict the tag for a programming question on Stack Overflow.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:24.349826Z",
     "iopub.status.busy": "2023-12-07T03:08:24.349379Z",
     "iopub.status.idle": "2023-12-07T03:08:26.973546Z",
     "shell.execute_reply": "2023-12-07T03:08:26.972825Z"
    },
    "id": "8RZOuS9LWQvv"
   },
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'matplotlib'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mmatplotlib\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpyplot\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mplt\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mos\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mre\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'matplotlib'"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import re\n",
    "import shutil\n",
    "import string\n",
    "import tensorflow as tf\n",
    "\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras import losses\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:26.978120Z",
     "iopub.status.busy": "2023-12-07T03:08:26.977441Z",
     "iopub.status.idle": "2023-12-07T03:08:26.981502Z",
     "shell.execute_reply": "2023-12-07T03:08:26.980893Z"
    },
    "id": "6-tTFS04dChr"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.15.0\n"
     ]
    }
   ],
   "source": [
    "print(tf.__version__)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NBTI1bi8qdFV"
   },
   "source": [
    "## Sentiment analysis\n",
    "\n",
    "This notebook trains a sentiment analysis model to classify movie reviews as *positive* or *negative*, based on the text of the review. This is an example of *binary*—or two-class—classification, an important and widely applicable kind of machine learning problem.\n",
    "\n",
    "You'll use the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/). These are split into 25,000 reviews for training and 25,000 reviews for testing. The training and testing sets are *balanced*, meaning they contain an equal number of positive and negative reviews.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "iAsKG535pHep"
   },
   "source": [
    "### Download and explore the IMDB dataset\n",
    "\n",
    "Let's download and extract the dataset, then explore the directory structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:26.984681Z",
     "iopub.status.busy": "2023-12-07T03:08:26.984455Z",
     "iopub.status.idle": "2023-12-07T03:08:51.950665Z",
     "shell.execute_reply": "2023-12-07T03:08:51.949900Z"
    },
    "id": "k7ZYnuajVlFN"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "    8192/84125825 [..............................] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "   40960/84125825 [..............................] - ETA: 3:14"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "   90112/84125825 [..............................] - ETA: 2:52"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  139264/84125825 [..............................] - ETA: 2:22"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  221184/84125825 [..............................] - ETA: 2:03"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  303104/84125825 [..............................] - ETA: 1:44"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  434176/84125825 [..............................] - ETA: 1:30"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  548864/84125825 [..............................] - ETA: 1:19"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  745472/84125825 [..............................] - ETA: 1:08"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  925696/84125825 [..............................] - ETA: 59s "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 1204224/84125825 [..............................] - ETA: 51s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 1449984/84125825 [..............................] - ETA: 45s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 1843200/84125825 [..............................] - ETA: 39s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 2170880/84125825 [..............................] - ETA: 35s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 2678784/84125825 [..............................] - ETA: 31s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 3104768/84125825 [>.............................] - ETA: 28s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 3776512/84125825 [>.............................] - ETA: 24s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 4235264/84125825 [>.............................] - ETA: 23s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 4890624/84125825 [>.............................] - ETA: 20s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 5857280/84125825 [=>............................] - ETA: 17s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 6496256/84125825 [=>............................] - ETA: 16s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 7364608/84125825 [=>............................] - ETA: 15s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 8577024/84125825 [==>...........................] - ETA: 13s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 9363456/84125825 [==>...........................] - ETA: 12s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "10444800/84125825 [==>...........................] - ETA: 11s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "11886592/84125825 [===>..........................] - ETA: 10s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "13099008/84125825 [===>..........................] - ETA: 9s "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "14704640/84125825 [====>.........................] - ETA: 8s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "16130048/84125825 [====>.........................] - ETA: 8s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "17915904/84125825 [=====>........................] - ETA: 7s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "19128320/84125825 [=====>........................] - ETA: 7s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "20881408/84125825 [======>.......................] - ETA: 6s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "22937600/84125825 [=======>......................] - ETA: 5s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "24420352/84125825 [=======>......................] - ETA: 5s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "25632768/84125825 [========>.....................] - ETA: 5s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "27992064/84125825 [========>.....................] - ETA: 4s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "28991488/84125825 [=========>....................] - ETA: 4s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "30302208/84125825 [=========>....................] - ETA: 4s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "31899648/84125825 [==========>...................] - ETA: 4s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "32907264/84125825 [==========>...................] - ETA: 4s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "34267136/84125825 [===========>..................] - ETA: 3s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "35905536/84125825 [===========>..................] - ETA: 3s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "36921344/84125825 [============>.................] - ETA: 3s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "38297600/84125825 [============>.................] - ETA: 3s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "39829504/84125825 [=============>................] - ETA: 3s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "41033728/84125825 [=============>................] - ETA: 3s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "42459136/84125825 [==============>...............] - ETA: 3s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "43925504/84125825 [==============>...............] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "45228032/84125825 [===============>..............] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "46309376/84125825 [===============>..............] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "47767552/84125825 [================>.............] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "49389568/84125825 [================>.............] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "50618368/84125825 [=================>............] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "52109312/84125825 [=================>............] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "53764096/84125825 [==================>...........] - ETA: 2s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "55025664/84125825 [==================>...........] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "56532992/84125825 [===================>..........] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "58204160/84125825 [===================>..........] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "59531264/84125825 [====================>.........] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "61054976/84125825 [====================>.........] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "62693376/84125825 [=====================>........] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "64102400/84125825 [=====================>........] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "65642496/84125825 [======================>.......] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "67330048/84125825 [=======================>......] - ETA: 1s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "68739072/84125825 [=======================>......] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "70328320/84125825 [========================>.....] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "72146944/84125825 [========================>.....] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "73441280/84125825 [=========================>....] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "74637312/84125825 [=========================>....] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "76226560/84125825 [==========================>...] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "77799424/84125825 [==========================>...] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "79388672/84125825 [===========================>..] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "80601088/84125825 [===========================>..] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "82223104/84125825 [============================>.] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "83992576/84125825 [============================>.] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "84125825/84125825 [==============================] - 5s 0us/step\n"
     ]
    }
   ],
   "source": [
    "url = \"https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\"\n",
    "\n",
    "dataset = tf.keras.utils.get_file(\"aclImdb_v1\", url,\n",
    "                                    untar=True, cache_dir='.',\n",
    "                                    cache_subdir='')\n",
    "\n",
    "dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:51.954835Z",
     "iopub.status.busy": "2023-12-07T03:08:51.954547Z",
     "iopub.status.idle": "2023-12-07T03:08:51.961580Z",
     "shell.execute_reply": "2023-12-07T03:08:51.960911Z"
    },
    "id": "355CfOvsV1pl"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['imdbEr.txt', 'train', 'test', 'imdb.vocab', 'README']"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.listdir(dataset_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:51.964630Z",
     "iopub.status.busy": "2023-12-07T03:08:51.964379Z",
     "iopub.status.idle": "2023-12-07T03:08:51.969183Z",
     "shell.execute_reply": "2023-12-07T03:08:51.968577Z"
    },
    "id": "7ASND15oXpF1"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['unsup',\n",
       " 'neg',\n",
       " 'unsupBow.feat',\n",
       " 'urls_pos.txt',\n",
       " 'pos',\n",
       " 'urls_neg.txt',\n",
       " 'urls_unsup.txt',\n",
       " 'labeledBow.feat']"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_dir = os.path.join(dataset_dir, 'train')\n",
    "os.listdir(train_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ysMNMI1CWDFD"
   },
   "source": [
    "The `aclImdb/train/pos` and `aclImdb/train/neg` directories contain many text files, each of which is a single movie review. Let's take a look at one of them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:51.972523Z",
     "iopub.status.busy": "2023-12-07T03:08:51.972265Z",
     "iopub.status.idle": "2023-12-07T03:08:51.976373Z",
     "shell.execute_reply": "2023-12-07T03:08:51.975699Z"
    },
    "id": "R7g8hFvzWLIZ"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rachel Griffiths writes and directs this award winning short film. A heartwarming story about coping with grief and cherishing the memory of those we've loved and lost. Although, only 15 minutes long, Griffiths manages to capture so much emotion and truth onto film in the short space of time. Bud Tingwell gives a touching performance as Will, a widower struggling to cope with his wife's death. Will is confronted by the harsh reality of loneliness and helplessness as he proceeds to take care of Ruth's pet cow, Tulip. The film displays the grief and responsibility one feels for those they have loved and lost. Good cinematography, great direction, and superbly acted. It will bring tears to all those who have lost a loved one, and survived.\n"
     ]
    }
   ],
   "source": [
    "sample_file = os.path.join(train_dir, 'pos/1181_9.txt')\n",
    "with open(sample_file) as f:\n",
    "  print(f.read())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Mk20TEm6ZRFP"
   },
   "source": [
    "### Load the dataset\n",
    "\n",
    "Next, you will load the data off disk and prepare it into a format suitable for training. To do so, you will use the helpful [text_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory) utility, which expects a directory structure as follows.\n",
    "\n",
    "```\n",
    "main_directory/\n",
    "...class_a/\n",
    "......a_text_1.txt\n",
    "......a_text_2.txt\n",
    "...class_b/\n",
    "......b_text_1.txt\n",
    "......b_text_2.txt\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nQauv38Lnok3"
   },
   "source": [
    "To prepare a dataset for binary classification, you will need two folders on disk, corresponding to `class_a` and `class_b`. These will be the positive and negative movie reviews, which can be found in  `aclImdb/train/pos` and `aclImdb/train/neg`. As the IMDB dataset contains additional folders, you will remove them before using this utility."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:51.979763Z",
     "iopub.status.busy": "2023-12-07T03:08:51.979510Z",
     "iopub.status.idle": "2023-12-07T03:08:52.854248Z",
     "shell.execute_reply": "2023-12-07T03:08:52.853422Z"
    },
    "id": "VhejsClzaWfl"
   },
   "outputs": [],
   "source": [
    "remove_dir = os.path.join(train_dir, 'unsup')\n",
    "shutil.rmtree(remove_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "95kkUdRoaeMw"
   },
   "source": [
    "Next, you will use the `text_dataset_from_directory` utility to create a labeled `tf.data.Dataset`. [tf.data](https://www.tensorflow.org/guide/data) is a powerful collection of tools for working with data. \n",
    "\n",
    "When running a machine learning experiment, it is a best practice to divide your dataset into three splits: [train](https://developers.google.com/machine-learning/glossary#training_set), [validation](https://developers.google.com/machine-learning/glossary#validation_set), and [test](https://developers.google.com/machine-learning/glossary#test-set). \n",
    "\n",
    "The IMDB dataset has already been divided into train and test, but it lacks a validation set. Let's create a validation set using an 80:20 split of the training data by using the `validation_split` argument below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:52.858753Z",
     "iopub.status.busy": "2023-12-07T03:08:52.858195Z",
     "iopub.status.idle": "2023-12-07T03:08:57.346376Z",
     "shell.execute_reply": "2023-12-07T03:08:57.345613Z"
    },
    "id": "nOrK-MTYaw3C"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 25000 files belonging to 2 classes.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using 20000 files for training.\n"
     ]
    }
   ],
   "source": [
    "batch_size = 32\n",
    "seed = 42\n",
    "\n",
    "raw_train_ds = tf.keras.utils.text_dataset_from_directory(\n",
    "    'aclImdb/train', \n",
    "    batch_size=batch_size, \n",
    "    validation_split=0.2, \n",
    "    subset='training', \n",
    "    seed=seed)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5Y33oxOUpYkh"
   },
   "source": [
    "As you can see above, there are 25,000 examples in the training folder, of which you will use 80% (or 20,000) for training. As you will see in a moment, you can train a model by passing a dataset directly to `model.fit`. If you're new to `tf.data`, you can also iterate over the dataset and print out a few examples as follows."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:57.350625Z",
     "iopub.status.busy": "2023-12-07T03:08:57.350332Z",
     "iopub.status.idle": "2023-12-07T03:08:57.387941Z",
     "shell.execute_reply": "2023-12-07T03:08:57.387160Z"
    },
    "id": "51wNaPPApk1K"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Review b'\"Pandemonium\" is a horror movie spoof that comes off more stupid than funny. Believe me when I tell you, I love comedies. Especially comedy spoofs. \"Airplane\", \"The Naked Gun\" trilogy, \"Blazing Saddles\", \"High Anxiety\", and \"Spaceballs\" are some of my favorite comedies that spoof a particular genre. \"Pandemonium\" is not up there with those films. Most of the scenes in this movie had me sitting there in stunned silence because the movie wasn\\'t all that funny. There are a few laughs in the film, but when you watch a comedy, you expect to laugh a lot more than a few times and that\\'s all this film has going for it. Geez, \"Scream\" had more laughs than this film and that was more of a horror film. How bizarre is that?<br /><br />*1/2 (out of four)'\n",
      "Label 0\n",
      "Review b\"David Mamet is a very interesting and a very un-equal director. His first movie 'House of Games' was the one I liked best, and it set a series of films with characters whose perspective of life changes as they get into complicated situations, and so does the perspective of the viewer.<br /><br />So is 'Homicide' which from the title tries to set the mind of the viewer to the usual crime drama. The principal characters are two cops, one Jewish and one Irish who deal with a racially charged area. The murder of an old Jewish shop owner who proves to be an ancient veteran of the Israeli Independence war triggers the Jewish identity in the mind and heart of the Jewish detective.<br /><br />This is were the flaws of the film are the more obvious. The process of awakening is theatrical and hard to believe, the group of Jewish militants is operatic, and the way the detective eventually walks to the final violent confrontation is pathetic. The end of the film itself is Mamet-like smart, but disappoints from a human emotional perspective.<br /><br />Joe Mantegna and William Macy give strong performances, but the flaws of the story are too evident to be easily compensated.\"\n",
      "Label 0\n",
      "Review b'Great documentary about the lives of NY firefighters during the worst terrorist attack of all time.. That reason alone is why this should be a must see collectors item.. What shocked me was not only the attacks, but the\"High Fat Diet\" and physical appearance of some of these firefighters. I think a lot of Doctors would agree with me that,in the physical shape they were in, some of these firefighters would NOT of made it to the 79th floor carrying over 60 lbs of gear. Having said that i now have a greater respect for firefighters and i realize becoming a firefighter is a life altering job. The French have a history of making great documentary\\'s and that is what this is, a Great Documentary.....'\n",
      "Label 1\n"
     ]
    }
   ],
   "source": [
    "for text_batch, label_batch in raw_train_ds.take(1):\n",
    "  for i in range(3):\n",
    "    print(\"Review\", text_batch.numpy()[i])\n",
    "    print(\"Label\", label_batch.numpy()[i])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JWq1SUIrp1a-"
   },
   "source": [
    "Notice the reviews contain raw text (with punctuation and occasional HTML tags like `<br/>`). You will show how to handle these in the following section. \n",
    "\n",
    "The labels are 0 or 1. To see which of these correspond to positive and negative movie reviews, you can check the `class_names` property on the dataset.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:57.391637Z",
     "iopub.status.busy": "2023-12-07T03:08:57.391048Z",
     "iopub.status.idle": "2023-12-07T03:08:57.395218Z",
     "shell.execute_reply": "2023-12-07T03:08:57.394506Z"
    },
    "id": "MlICTG8spyO2"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Label 0 corresponds to neg\n",
      "Label 1 corresponds to pos\n"
     ]
    }
   ],
   "source": [
    "print(\"Label 0 corresponds to\", raw_train_ds.class_names[0])\n",
    "print(\"Label 1 corresponds to\", raw_train_ds.class_names[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pbdO39vYqdJr"
   },
   "source": [
    "Next, you will create a validation and test dataset. You will use the remaining 5,000 reviews from the training set for validation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SzxazN8Hq1pF"
   },
   "source": [
    "Note:  When using the `validation_split` and `subset` arguments, make sure to either specify a random seed, or to pass `shuffle=False`, so that the validation and training splits have no overlap."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:57.398670Z",
     "iopub.status.busy": "2023-12-07T03:08:57.398166Z",
     "iopub.status.idle": "2023-12-07T03:08:58.601934Z",
     "shell.execute_reply": "2023-12-07T03:08:58.601226Z"
    },
    "id": "JsMwwhOoqjKF"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 25000 files belonging to 2 classes.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using 5000 files for validation.\n"
     ]
    }
   ],
   "source": [
    "raw_val_ds = tf.keras.utils.text_dataset_from_directory(\n",
    "    'aclImdb/train', \n",
    "    batch_size=batch_size, \n",
    "    validation_split=0.2, \n",
    "    subset='validation', \n",
    "    seed=seed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:58.605300Z",
     "iopub.status.busy": "2023-12-07T03:08:58.605033Z",
     "iopub.status.idle": "2023-12-07T03:08:59.808028Z",
     "shell.execute_reply": "2023-12-07T03:08:59.807286Z"
    },
    "id": "rdSr0Nt3q_ns"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 25000 files belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "raw_test_ds = tf.keras.utils.text_dataset_from_directory(\n",
    "    'aclImdb/test', \n",
    "    batch_size=batch_size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "qJmTiO0IYAjm"
   },
   "source": [
    "### Prepare the dataset for training\n",
    "\n",
    "Next, you will standardize, tokenize, and vectorize the data using the helpful `tf.keras.layers.TextVectorization` layer. \n",
    "\n",
    "Standardization refers to preprocessing the text, typically to remove punctuation or HTML elements to simplify the dataset. Tokenization refers to splitting strings into tokens (for example, splitting a sentence into individual words, by splitting on whitespace). Vectorization refers to converting tokens into numbers so they can be fed into a neural network. All of these tasks can be accomplished with this layer.\n",
    "\n",
    "As you saw above, the reviews contain various HTML tags like `<br />`. These tags will not be removed by the default standardizer in the `TextVectorization` layer (which converts text to lowercase and strips punctuation by default, but doesn't strip HTML). You will write a custom standardization function to remove the HTML."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZVcHl-SLrH-u"
   },
   "source": [
    "Note: To prevent [training-testing skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew) (also known as training-serving skew), it is important to preprocess the data identically at train and test time. To facilitate this, the `TextVectorization` layer can be included directly inside your model, as shown later in this tutorial."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:59.811900Z",
     "iopub.status.busy": "2023-12-07T03:08:59.811625Z",
     "iopub.status.idle": "2023-12-07T03:08:59.815676Z",
     "shell.execute_reply": "2023-12-07T03:08:59.815053Z"
    },
    "id": "SDRI_s_tX1Hk"
   },
   "outputs": [],
   "source": [
    "def custom_standardization(input_data):\n",
    "  lowercase = tf.strings.lower(input_data)\n",
    "  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')\n",
    "  return tf.strings.regex_replace(stripped_html,\n",
    "                                  '[%s]' % re.escape(string.punctuation),\n",
    "                                  '')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d2d3Aw8dsUux"
   },
   "source": [
    "Next, you will create a `TextVectorization` layer. You will use this layer to standardize, tokenize, and vectorize our data. You set the `output_mode` to `int` to create unique integer indices for each token.\n",
    "\n",
    "Note that you're using the default split function, and the custom standardization function you defined above. You'll also define some constants for the model, like an explicit maximum `sequence_length`, which will cause the layer to pad or truncate sequences to exactly `sequence_length` values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:59.818926Z",
     "iopub.status.busy": "2023-12-07T03:08:59.818479Z",
     "iopub.status.idle": "2023-12-07T03:08:59.833160Z",
     "shell.execute_reply": "2023-12-07T03:08:59.832490Z"
    },
    "id": "-c76RvSzsMnX"
   },
   "outputs": [],
   "source": [
    "max_features = 10000\n",
    "sequence_length = 250\n",
    "\n",
    "vectorize_layer = layers.TextVectorization(\n",
    "    standardize=custom_standardization,\n",
    "    max_tokens=max_features,\n",
    "    output_mode='int',\n",
    "    output_sequence_length=sequence_length)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vlFOpfF6scT6"
   },
   "source": [
    "Next, you will call `adapt` to fit the state of the preprocessing layer to the dataset. This will cause the model to build an index of strings to integers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lAhdjK7AtroA"
   },
   "source": [
    "Note: It's important to only use your training data when calling adapt (using the test set would leak information)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:08:59.836975Z",
     "iopub.status.busy": "2023-12-07T03:08:59.836472Z",
     "iopub.status.idle": "2023-12-07T03:09:02.200795Z",
     "shell.execute_reply": "2023-12-07T03:09:02.200003Z"
    },
    "id": "GH4_2ZGJsa_X"
   },
   "outputs": [],
   "source": [
    "# Make a text-only dataset (without labels), then call adapt\n",
    "train_text = raw_train_ds.map(lambda x, y: x)\n",
    "vectorize_layer.adapt(train_text)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SHQVEFzNt-K_"
   },
   "source": [
    "Let's create a function to see the result of using this layer to preprocess some data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.205405Z",
     "iopub.status.busy": "2023-12-07T03:09:02.204721Z",
     "iopub.status.idle": "2023-12-07T03:09:02.208610Z",
     "shell.execute_reply": "2023-12-07T03:09:02.207965Z"
    },
    "id": "SCIg_T50wOCU"
   },
   "outputs": [],
   "source": [
    "def vectorize_text(text, label):\n",
    "  text = tf.expand_dims(text, -1)\n",
    "  return vectorize_layer(text), label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.212138Z",
     "iopub.status.busy": "2023-12-07T03:09:02.211644Z",
     "iopub.status.idle": "2023-12-07T03:09:02.280477Z",
     "shell.execute_reply": "2023-12-07T03:09:02.279755Z"
    },
    "id": "XULcm6B3xQIO"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Review tf.Tensor(b'Great movie - especially the music - Etta James - \"At Last\". This speaks volumes when you have finally found that special someone.', shape=(), dtype=string)\n",
      "Label neg\n",
      "Vectorized review (<tf.Tensor: shape=(1, 250), dtype=int64, numpy=\n",
      "array([[  86,   17,  260,    2,  222,    1,  571,   31,  229,   11, 2418,\n",
      "           1,   51,   22,   25,  404,  251,   12,  306,  282,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
      "           0,    0,    0,    0,    0,    0,    0,    0]])>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)\n"
     ]
    }
   ],
   "source": [
    "# retrieve a batch (of 32 reviews and labels) from the dataset\n",
    "text_batch, label_batch = next(iter(raw_train_ds))\n",
    "first_review, first_label = text_batch[0], label_batch[0]\n",
    "print(\"Review\", first_review)\n",
    "print(\"Label\", raw_train_ds.class_names[first_label])\n",
    "print(\"Vectorized review\", vectorize_text(first_review, first_label))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6u5EX0hxyNZT"
   },
   "source": [
    "As you can see above, each token has been replaced by an integer. You can lookup the token (string) that each integer corresponds to by calling `.get_vocabulary()` on the layer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.283779Z",
     "iopub.status.busy": "2023-12-07T03:09:02.283512Z",
     "iopub.status.idle": "2023-12-07T03:09:02.331164Z",
     "shell.execute_reply": "2023-12-07T03:09:02.330474Z"
    },
    "id": "kRq9hTQzhVhW"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1287 --->  silent\n",
      " 313 --->  night\n",
      "Vocabulary size: 10000\n"
     ]
    }
   ],
   "source": [
    "print(\"1287 ---> \",vectorize_layer.get_vocabulary()[1287])\n",
    "print(\" 313 ---> \",vectorize_layer.get_vocabulary()[313])\n",
    "print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "XD2H6utRydGv"
   },
   "source": [
    "You are nearly ready to train your model. As a final preprocessing step, you will apply the TextVectorization layer you created earlier to the train, validation, and test dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.334804Z",
     "iopub.status.busy": "2023-12-07T03:09:02.334307Z",
     "iopub.status.idle": "2023-12-07T03:09:02.464572Z",
     "shell.execute_reply": "2023-12-07T03:09:02.463958Z"
    },
    "id": "2zhmpeViI1iG"
   },
   "outputs": [],
   "source": [
    "train_ds = raw_train_ds.map(vectorize_text)\n",
    "val_ds = raw_val_ds.map(vectorize_text)\n",
    "test_ds = raw_test_ds.map(vectorize_text)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YsVQyPMizjuO"
   },
   "source": [
    "### Configure the dataset for performance\n",
    "\n",
    "These are two important methods you should use when loading data to make sure that I/O does not become blocking.\n",
    "\n",
    "`.cache()` keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache, which is more efficient to read than many small files.\n",
    "\n",
    "`.prefetch()` overlaps data preprocessing and model execution while training. \n",
    "\n",
    "You can learn more about both methods, as well as how to cache data to disk in the [data performance guide](https://www.tensorflow.org/guide/data_performance)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.468326Z",
     "iopub.status.busy": "2023-12-07T03:09:02.467858Z",
     "iopub.status.idle": "2023-12-07T03:09:02.479461Z",
     "shell.execute_reply": "2023-12-07T03:09:02.478759Z"
    },
    "id": "wMcs_H7izm5m"
   },
   "outputs": [],
   "source": [
    "AUTOTUNE = tf.data.AUTOTUNE\n",
    "\n",
    "train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)\n",
    "val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)\n",
    "test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LLC02j2g-llC"
   },
   "source": [
    "### Create the model\n",
    "\n",
    "It's time to create your neural network:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.483125Z",
     "iopub.status.busy": "2023-12-07T03:09:02.482553Z",
     "iopub.status.idle": "2023-12-07T03:09:02.485755Z",
     "shell.execute_reply": "2023-12-07T03:09:02.485184Z"
    },
    "id": "dkQP6in8yUBR"
   },
   "outputs": [],
   "source": [
    "embedding_dim = 16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.488911Z",
     "iopub.status.busy": "2023-12-07T03:09:02.488433Z",
     "iopub.status.idle": "2023-12-07T03:09:02.550473Z",
     "shell.execute_reply": "2023-12-07T03:09:02.549824Z"
    },
    "id": "xpKOoWgu-llD"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Layer (type)                Output Shape              Param #   \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=================================================================\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " embedding (Embedding)       (None, None, 16)          160000    \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                                 \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " dropout (Dropout)           (None, None, 16)          0         \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                                 \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " global_average_pooling1d (  (None, 16)                0         \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " GlobalAveragePooling1D)                                         \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                                 \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " dropout_1 (Dropout)         (None, 16)                0         \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                                 \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " dense (Dense)               (None, 1)                 17        \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                                                 \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=================================================================\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total params: 160017 (625.07 KB)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Trainable params: 160017 (625.07 KB)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Non-trainable params: 0 (0.00 Byte)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model = tf.keras.Sequential([\n",
    "  layers.Embedding(max_features, embedding_dim),\n",
    "  layers.Dropout(0.2),\n",
    "  layers.GlobalAveragePooling1D(),\n",
    "  layers.Dropout(0.2),\n",
    "  layers.Dense(1)])\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6PbKQ6mucuKL"
   },
   "source": [
    "The layers are stacked sequentially to build the classifier:\n",
    "\n",
    "1. The first layer is an `Embedding` layer. This layer takes the integer-encoded reviews and looks up an embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: `(batch, sequence, embedding)`.  To learn more about embeddings, check out the [Word embeddings](https://www.tensorflow.org/text/guide/word_embeddings) tutorial.\n",
    "2. Next, a `GlobalAveragePooling1D` layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.\n",
    "3. The last layer is densely connected with a single output node."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "L4EqVWg4-llM"
   },
   "source": [
    "### Loss function and optimizer\n",
    "\n",
    "A model needs a loss function and an optimizer for training. Since this is a binary classification problem and the model outputs a probability (a single-unit layer with a sigmoid activation), you'll use `losses.BinaryCrossentropy` loss function.\n",
    "\n",
    "Now, configure the model to use an optimizer and a loss function:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.559586Z",
     "iopub.status.busy": "2023-12-07T03:09:02.558876Z",
     "iopub.status.idle": "2023-12-07T03:09:02.576310Z",
     "shell.execute_reply": "2023-12-07T03:09:02.575731Z"
    },
    "id": "Mr0GP-cQ-llN"
   },
   "outputs": [],
   "source": [
    "model.compile(loss=losses.BinaryCrossentropy(from_logits=True),\n",
    "              optimizer='adam',\n",
    "              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "35jv_fzP-llU"
   },
   "source": [
    "### Train the model\n",
    "\n",
    "You will train the model by passing the `dataset` object to the fit method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:09:02.580185Z",
     "iopub.status.busy": "2023-12-07T03:09:02.579585Z",
     "iopub.status.idle": "2023-12-07T03:10:06.859914Z",
     "shell.execute_reply": "2023-12-07T03:10:06.858775Z"
    },
    "id": "tXSGrjWZ-llW"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
      "I0000 00:00:1701918544.534151   83357 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 23:03 - loss: 0.6921 - binary_accuracy: 0.4375"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  2/625 [..............................] - ETA: 1:36 - loss: 0.6933 - binary_accuracy: 0.3906 "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  3/625 [..............................] - ETA: 1:38 - loss: 0.6923 - binary_accuracy: 0.4688"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  4/625 [..............................] - ETA: 1:36 - loss: 0.6923 - binary_accuracy: 0.4844"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  5/625 [..............................] - ETA: 1:36 - loss: 0.6919 - binary_accuracy: 0.5000"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  6/625 [..............................] - ETA: 1:35 - loss: 0.6918 - binary_accuracy: 0.5052"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  7/625 [..............................] - ETA: 1:35 - loss: 0.6926 - binary_accuracy: 0.4866"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  8/625 [..............................] - ETA: 1:34 - loss: 0.6930 - binary_accuracy: 0.4805"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "  9/625 [..............................] - ETA: 1:34 - loss: 0.6928 - binary_accuracy: 0.4861"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 10/625 [..............................] - ETA: 1:34 - loss: 0.6935 - binary_accuracy: 0.4844"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 11/625 [..............................] - ETA: 1:33 - loss: 0.6932 - binary_accuracy: 0.4972"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 12/625 [..............................] - ETA: 1:33 - loss: 0.6932 - binary_accuracy: 0.5000"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 13/625 [..............................] - ETA: 1:33 - loss: 0.6929 - binary_accuracy: 0.4976"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 14/625 [..............................] - ETA: 1:33 - loss: 0.6932 - binary_accuracy: 0.4933"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 15/625 [..............................] - ETA: 1:33 - loss: 0.6933 - binary_accuracy: 0.4938"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 16/625 [..............................] - ETA: 1:33 - loss: 0.6929 - binary_accuracy: 0.4980"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 17/625 [..............................] - ETA: 1:32 - loss: 0.6931 - binary_accuracy: 0.4982"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1:32 - loss: 0.6931 - binary_accuracy: 0.5000"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 19/625 [..............................] - ETA: 1:32 - loss: 0.6933 - binary_accuracy: 0.4951"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 21/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4940"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 22/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4972"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 23/625 [>.............................] - ETA: 1:27 - loss: 0.6930 - binary_accuracy: 0.5027"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 24/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4987"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 25/625 [>.............................] - ETA: 1:27 - loss: 0.6930 - binary_accuracy: 0.5000"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 26/625 [>.............................] - ETA: 1:27 - loss: 0.6929 - binary_accuracy: 0.5036"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 27/625 [>.............................] - ETA: 1:27 - loss: 0.6932 - binary_accuracy: 0.4942"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 28/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4944"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 29/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4968"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 30/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4979"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 31/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4940"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 32/625 [>.............................] - ETA: 1:27 - loss: 0.6931 - binary_accuracy: 0.4941"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 33/625 [>.............................] - ETA: 1:27 - loss: 0.6930 - binary_accuracy: 0.4972"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 35/625 [>.............................] - ETA: 1:24 - loss: 0.6929 - binary_accuracy: 0.4964"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 36/625 [>.............................] - ETA: 1:24 - loss: 0.6928 - binary_accuracy: 0.4991"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 37/625 [>.............................] - ETA: 1:24 - loss: 0.6928 - binary_accuracy: 0.5000"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 38/625 [>.............................] - ETA: 1:24 - loss: 0.6928 - binary_accuracy: 0.4992"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 39/625 [>.............................] - ETA: 1:24 - loss: 0.6927 - binary_accuracy: 0.5008"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 40/625 [>.............................] - ETA: 1:24 - loss: 0.6926 - binary_accuracy: 0.5055"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 41/625 [>.............................] - ETA: 1:24 - loss: 0.6926 - binary_accuracy: 0.5076"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 42/625 [=>............................] - ETA: 1:24 - loss: 0.6925 - binary_accuracy: 0.5104"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 43/625 [=>............................] - ETA: 1:24 - loss: 0.6925 - binary_accuracy: 0.5138"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 44/625 [=>............................] - ETA: 1:24 - loss: 0.6924 - binary_accuracy: 0.5185"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 46/625 [=>............................] - ETA: 1:22 - loss: 0.6923 - binary_accuracy: 0.5251"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 47/625 [=>............................] - ETA: 1:22 - loss: 0.6923 - binary_accuracy: 0.5279"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 48/625 [=>............................] - ETA: 1:22 - loss: 0.6922 - binary_accuracy: 0.5326"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 49/625 [=>............................] - ETA: 1:22 - loss: 0.6921 - binary_accuracy: 0.5357"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 50/625 [=>............................] - ETA: 1:22 - loss: 0.6921 - binary_accuracy: 0.5350"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 51/625 [=>............................] - ETA: 1:22 - loss: 0.6921 - binary_accuracy: 0.5355"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 52/625 [=>............................] - ETA: 1:22 - loss: 0.6921 - binary_accuracy: 0.5367"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 54/625 [=>............................] - ETA: 1:20 - loss: 0.6920 - binary_accuracy: 0.5411"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 55/625 [=>............................] - ETA: 1:20 - loss: 0.6919 - binary_accuracy: 0.5409"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 56/625 [=>............................] - ETA: 1:20 - loss: 0.6920 - binary_accuracy: 0.5402"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 57/625 [=>............................] - ETA: 1:20 - loss: 0.6919 - binary_accuracy: 0.5395"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 58/625 [=>............................] - ETA: 1:20 - loss: 0.6919 - binary_accuracy: 0.5399"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 59/625 [=>............................] - ETA: 1:20 - loss: 0.6918 - binary_accuracy: 0.5392"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 60/625 [=>............................] - ETA: 1:20 - loss: 0.6917 - binary_accuracy: 0.5406"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 61/625 [=>............................] - ETA: 1:20 - loss: 0.6917 - binary_accuracy: 0.5400"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 62/625 [=>............................] - ETA: 1:20 - loss: 0.6916 - binary_accuracy: 0.5423"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 63/625 [==>...........................] - ETA: 1:20 - loss: 0.6915 - binary_accuracy: 0.5422"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 64/625 [==>...........................] - ETA: 1:20 - loss: 0.6915 - binary_accuracy: 0.5410"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 65/625 [==>...........................] - ETA: 1:20 - loss: 0.6915 - binary_accuracy: 0.5413"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 68/625 [==>...........................] - ETA: 1:17 - loss: 0.6915 - binary_accuracy: 0.5381"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 70/625 [==>...........................] - ETA: 1:16 - loss: 0.6915 - binary_accuracy: 0.5371"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 71/625 [==>...........................] - ETA: 1:16 - loss: 0.6914 - binary_accuracy: 0.5370"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 73/625 [==>...........................] - ETA: 1:14 - loss: 0.6914 - binary_accuracy: 0.5355"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 74/625 [==>...........................] - ETA: 1:14 - loss: 0.6913 - binary_accuracy: 0.5363"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 76/625 [==>...........................] - ETA: 1:13 - loss: 0.6911 - binary_accuracy: 0.5382"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 77/625 [==>...........................] - ETA: 1:13 - loss: 0.6911 - binary_accuracy: 0.5373"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 78/625 [==>...........................] - ETA: 1:13 - loss: 0.6912 - binary_accuracy: 0.5349"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 79/625 [==>...........................] - ETA: 1:13 - loss: 0.6911 - binary_accuracy: 0.5360"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 80/625 [==>...........................] - ETA: 1:13 - loss: 0.6910 - binary_accuracy: 0.5363"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 83/625 [==>...........................] - ETA: 1:11 - loss: 0.6911 - binary_accuracy: 0.5339"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 84/625 [===>..........................] - ETA: 1:11 - loss: 0.6910 - binary_accuracy: 0.5342"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 85/625 [===>..........................] - ETA: 1:11 - loss: 0.6911 - binary_accuracy: 0.5335"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 86/625 [===>..........................] - ETA: 1:11 - loss: 0.6910 - binary_accuracy: 0.5342"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 88/625 [===>..........................] - ETA: 1:10 - loss: 0.6909 - binary_accuracy: 0.5334"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 89/625 [===>..........................] - ETA: 1:10 - loss: 0.6911 - binary_accuracy: 0.5309"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 90/625 [===>..........................] - ETA: 1:10 - loss: 0.6910 - binary_accuracy: 0.5306"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 91/625 [===>..........................] - ETA: 1:10 - loss: 0.6910 - binary_accuracy: 0.5302"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 93/625 [===>..........................] - ETA: 1:09 - loss: 0.6911 - binary_accuracy: 0.5282"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 96/625 [===>..........................] - ETA: 1:08 - loss: 0.6911 - binary_accuracy: 0.5267"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 98/625 [===>..........................] - ETA: 1:07 - loss: 0.6910 - binary_accuracy: 0.5281"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "101/625 [===>..........................] - ETA: 1:05 - loss: 0.6911 - binary_accuracy: 0.5251"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "102/625 [===>..........................] - ETA: 1:05 - loss: 0.6911 - binary_accuracy: 0.5245"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "104/625 [===>..........................] - ETA: 1:05 - loss: 0.6910 - binary_accuracy: 0.5249"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "105/625 [====>.........................] - ETA: 1:05 - loss: 0.6910 - binary_accuracy: 0.5247"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "107/625 [====>.........................] - ETA: 1:04 - loss: 0.6910 - binary_accuracy: 0.5254"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "108/625 [====>.........................] - ETA: 1:04 - loss: 0.6909 - binary_accuracy: 0.5272"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "109/625 [====>.........................] - ETA: 1:04 - loss: 0.6909 - binary_accuracy: 0.5278"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "110/625 [====>.........................] - ETA: 1:04 - loss: 0.6909 - binary_accuracy: 0.5287"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "111/625 [====>.........................] - ETA: 1:04 - loss: 0.6909 - binary_accuracy: 0.5290"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "112/625 [====>.........................] - ETA: 1:04 - loss: 0.6908 - binary_accuracy: 0.5296"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "113/625 [====>.........................] - ETA: 1:04 - loss: 0.6909 - binary_accuracy: 0.5285"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "115/625 [====>.........................] - ETA: 1:03 - loss: 0.6908 - binary_accuracy: 0.5291"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "117/625 [====>.........................] - ETA: 1:03 - loss: 0.6908 - binary_accuracy: 0.5291"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "119/625 [====>.........................] - ETA: 1:02 - loss: 0.6908 - binary_accuracy: 0.5305"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "120/625 [====>.........................] - ETA: 1:02 - loss: 0.6907 - binary_accuracy: 0.5320"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "121/625 [====>.........................] - ETA: 1:02 - loss: 0.6907 - binary_accuracy: 0.5325"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "122/625 [====>.........................] - ETA: 1:02 - loss: 0.6907 - binary_accuracy: 0.5351"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "124/625 [====>.........................] - ETA: 1:01 - loss: 0.6906 - binary_accuracy: 0.5373"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "125/625 [=====>........................] - ETA: 1:01 - loss: 0.6906 - binary_accuracy: 0.5380"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "127/625 [=====>........................] - ETA: 1:01 - loss: 0.6905 - binary_accuracy: 0.5408"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "130/625 [=====>........................] - ETA: 1:00 - loss: 0.6905 - binary_accuracy: 0.5445"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "131/625 [=====>........................] - ETA: 1:00 - loss: 0.6904 - binary_accuracy: 0.5465"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "134/625 [=====>........................] - ETA: 59s - loss: 0.6904 - binary_accuracy: 0.5492 "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "136/625 [=====>........................] - ETA: 58s - loss: 0.6903 - binary_accuracy: 0.5519"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "137/625 [=====>........................] - ETA: 58s - loss: 0.6903 - binary_accuracy: 0.5536"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "139/625 [=====>........................] - ETA: 57s - loss: 0.6903 - binary_accuracy: 0.5540"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "142/625 [=====>........................] - ETA: 56s - loss: 0.6902 - binary_accuracy: 0.5577"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "144/625 [=====>........................] - ETA: 56s - loss: 0.6901 - binary_accuracy: 0.5592"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "145/625 [=====>........................] - ETA: 56s - loss: 0.6901 - binary_accuracy: 0.5601"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "146/625 [======>.......................] - ETA: 56s - loss: 0.6901 - binary_accuracy: 0.5610"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "147/625 [======>.......................] - ETA: 56s - loss: 0.6901 - binary_accuracy: 0.5614"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "148/625 [======>.......................] - ETA: 56s - loss: 0.6900 - binary_accuracy: 0.5629"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "149/625 [======>.......................] - ETA: 56s - loss: 0.6900 - binary_accuracy: 0.5635"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "151/625 [======>.......................] - ETA: 55s - loss: 0.6900 - binary_accuracy: 0.5650"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "152/625 [======>.......................] - ETA: 55s - loss: 0.6899 - binary_accuracy: 0.5666"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "160/625 [======>.......................] - ETA: 52s - loss: 0.6897 - binary_accuracy: 0.5738"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "162/625 [======>.......................] - ETA: 52s - loss: 0.6896 - binary_accuracy: 0.5764"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "164/625 [======>.......................] - ETA: 51s - loss: 0.6896 - binary_accuracy: 0.5776"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "165/625 [======>.......................] - ETA: 51s - loss: 0.6895 - binary_accuracy: 0.5778"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "166/625 [======>.......................] - ETA: 51s - loss: 0.6895 - binary_accuracy: 0.5796"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "167/625 [=======>......................] - ETA: 51s - loss: 0.6894 - binary_accuracy: 0.5812"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "169/625 [=======>......................] - ETA: 51s - loss: 0.6894 - binary_accuracy: 0.5827"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "170/625 [=======>......................] - ETA: 51s - loss: 0.6893 - binary_accuracy: 0.5836"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "171/625 [=======>......................] - ETA: 51s - loss: 0.6893 - binary_accuracy: 0.5842"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "172/625 [=======>......................] - ETA: 51s - loss: 0.6893 - binary_accuracy: 0.5847"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "174/625 [=======>......................] - ETA: 51s - loss: 0.6892 - binary_accuracy: 0.5857"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "175/625 [=======>......................] - ETA: 51s - loss: 0.6892 - binary_accuracy: 0.5859"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "179/625 [=======>......................] - ETA: 49s - loss: 0.6891 - binary_accuracy: 0.5873"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "180/625 [=======>......................] - ETA: 49s - loss: 0.6891 - binary_accuracy: 0.5868"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "181/625 [=======>......................] - ETA: 49s - loss: 0.6891 - binary_accuracy: 0.5865"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "183/625 [=======>......................] - ETA: 49s - loss: 0.6890 - binary_accuracy: 0.5876"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "185/625 [=======>......................] - ETA: 49s - loss: 0.6889 - binary_accuracy: 0.5873"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "188/625 [========>.....................] - ETA: 48s - loss: 0.6888 - binary_accuracy: 0.5883"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "189/625 [========>.....................] - ETA: 48s - loss: 0.6888 - binary_accuracy: 0.5890"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "190/625 [========>.....................] - ETA: 48s - loss: 0.6887 - binary_accuracy: 0.5891"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "191/625 [========>.....................] - ETA: 48s - loss: 0.6887 - binary_accuracy: 0.5898"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "192/625 [========>.....................] - ETA: 48s - loss: 0.6887 - binary_accuracy: 0.5892"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "194/625 [========>.....................] - ETA: 47s - loss: 0.6887 - binary_accuracy: 0.5892"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "195/625 [========>.....................] - ETA: 47s - loss: 0.6887 - binary_accuracy: 0.5888"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "196/625 [========>.....................] - ETA: 47s - loss: 0.6886 - binary_accuracy: 0.5886"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "197/625 [========>.....................] - ETA: 47s - loss: 0.6886 - binary_accuracy: 0.5879"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "201/625 [========>.....................] - ETA: 46s - loss: 0.6885 - binary_accuracy: 0.5885"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "202/625 [========>.....................] - ETA: 46s - loss: 0.6885 - binary_accuracy: 0.5882"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "204/625 [========>.....................] - ETA: 46s - loss: 0.6885 - binary_accuracy: 0.5873"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "206/625 [========>.....................] - ETA: 46s - loss: 0.6885 - binary_accuracy: 0.5880"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "207/625 [========>.....................] - ETA: 46s - loss: 0.6884 - binary_accuracy: 0.5885"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "211/625 [=========>....................] - ETA: 45s - loss: 0.6882 - binary_accuracy: 0.5903"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "213/625 [=========>....................] - ETA: 44s - loss: 0.6882 - binary_accuracy: 0.5911"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "214/625 [=========>....................] - ETA: 44s - loss: 0.6882 - binary_accuracy: 0.5921"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "215/625 [=========>....................] - ETA: 44s - loss: 0.6881 - binary_accuracy: 0.5929"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "217/625 [=========>....................] - ETA: 44s - loss: 0.6880 - binary_accuracy: 0.5943"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "218/625 [=========>....................] - ETA: 44s - loss: 0.6880 - binary_accuracy: 0.5949"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "219/625 [=========>....................] - ETA: 44s - loss: 0.6879 - binary_accuracy: 0.5959"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "220/625 [=========>....................] - ETA: 44s - loss: 0.6879 - binary_accuracy: 0.5960"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "221/625 [=========>....................] - ETA: 44s - loss: 0.6879 - binary_accuracy: 0.5969"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "224/625 [=========>....................] - ETA: 43s - loss: 0.6878 - binary_accuracy: 0.5977"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "225/625 [=========>....................] - ETA: 43s - loss: 0.6877 - binary_accuracy: 0.5986"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "226/625 [=========>....................] - ETA: 43s - loss: 0.6876 - binary_accuracy: 0.5993"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "228/625 [=========>....................] - ETA: 43s - loss: 0.6876 - binary_accuracy: 0.5999"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "233/625 [==========>...................] - ETA: 42s - loss: 0.6874 - binary_accuracy: 0.6034"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "235/625 [==========>...................] - ETA: 41s - loss: 0.6873 - binary_accuracy: 0.6043"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "237/625 [==========>...................] - ETA: 41s - loss: 0.6873 - binary_accuracy: 0.6056"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "238/625 [==========>...................] - ETA: 41s - loss: 0.6872 - binary_accuracy: 0.6065"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "240/625 [==========>...................] - ETA: 41s - loss: 0.6871 - binary_accuracy: 0.6070"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "241/625 [==========>...................] - ETA: 41s - loss: 0.6870 - binary_accuracy: 0.6076"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "244/625 [==========>...................] - ETA: 40s - loss: 0.6869 - binary_accuracy: 0.6095"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "245/625 [==========>...................] - ETA: 40s - loss: 0.6868 - binary_accuracy: 0.6105"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "247/625 [==========>...................] - ETA: 40s - loss: 0.6868 - binary_accuracy: 0.6112"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "249/625 [==========>...................] - ETA: 39s - loss: 0.6867 - binary_accuracy: 0.6124"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "251/625 [===========>..................] - ETA: 39s - loss: 0.6866 - binary_accuracy: 0.6134"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "253/625 [===========>..................] - ETA: 39s - loss: 0.6866 - binary_accuracy: 0.6139"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "254/625 [===========>..................] - ETA: 39s - loss: 0.6865 - binary_accuracy: 0.6142"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "256/625 [===========>..................] - ETA: 39s - loss: 0.6864 - binary_accuracy: 0.6146"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "258/625 [===========>..................] - ETA: 38s - loss: 0.6863 - binary_accuracy: 0.6158"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "259/625 [===========>..................] - ETA: 38s - loss: 0.6863 - binary_accuracy: 0.6163"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "261/625 [===========>..................] - ETA: 38s - loss: 0.6862 - binary_accuracy: 0.6173"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "262/625 [===========>..................] - ETA: 38s - loss: 0.6861 - binary_accuracy: 0.6183"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "264/625 [===========>..................] - ETA: 38s - loss: 0.6860 - binary_accuracy: 0.6190"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "266/625 [===========>..................] - ETA: 37s - loss: 0.6859 - binary_accuracy: 0.6194"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "267/625 [===========>..................] - ETA: 37s - loss: 0.6859 - binary_accuracy: 0.6195"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "268/625 [===========>..................] - ETA: 37s - loss: 0.6858 - binary_accuracy: 0.6195"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "269/625 [===========>..................] - ETA: 37s - loss: 0.6858 - binary_accuracy: 0.6202"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "270/625 [===========>..................] - ETA: 37s - loss: 0.6858 - binary_accuracy: 0.6198"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "271/625 [============>.................] - ETA: 37s - loss: 0.6857 - binary_accuracy: 0.6206"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "275/625 [============>.................] - ETA: 36s - loss: 0.6855 - binary_accuracy: 0.6227"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "277/625 [============>.................] - ETA: 36s - loss: 0.6855 - binary_accuracy: 0.6226"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "280/625 [============>.................] - ETA: 36s - loss: 0.6853 - binary_accuracy: 0.6240"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "281/625 [============>.................] - ETA: 36s - loss: 0.6853 - binary_accuracy: 0.6240"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "284/625 [============>.................] - ETA: 35s - loss: 0.6851 - binary_accuracy: 0.6250"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "288/625 [============>.................] - ETA: 34s - loss: 0.6850 - binary_accuracy: 0.6262"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "291/625 [============>.................] - ETA: 34s - loss: 0.6849 - binary_accuracy: 0.6265"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "293/625 [=============>................] - ETA: 34s - loss: 0.6849 - binary_accuracy: 0.6267"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "301/625 [=============>................] - ETA: 32s - loss: 0.6845 - binary_accuracy: 0.6301"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "302/625 [=============>................] - ETA: 32s - loss: 0.6845 - binary_accuracy: 0.6307"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "306/625 [=============>................] - ETA: 31s - loss: 0.6843 - binary_accuracy: 0.6315"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "307/625 [=============>................] - ETA: 31s - loss: 0.6842 - binary_accuracy: 0.6321"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "308/625 [=============>................] - ETA: 31s - loss: 0.6842 - binary_accuracy: 0.6327"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "311/625 [=============>................] - ETA: 31s - loss: 0.6840 - binary_accuracy: 0.6334"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "315/625 [==============>...............] - ETA: 30s - loss: 0.6839 - binary_accuracy: 0.6337"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "321/625 [==============>...............] - ETA: 29s - loss: 0.6837 - binary_accuracy: 0.6352"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "327/625 [==============>...............] - ETA: 28s - loss: 0.6834 - binary_accuracy: 0.6368"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "329/625 [==============>...............] - ETA: 28s - loss: 0.6833 - binary_accuracy: 0.6374"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "330/625 [==============>...............] - ETA: 28s - loss: 0.6832 - binary_accuracy: 0.6380"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "333/625 [==============>...............] - ETA: 28s - loss: 0.6831 - binary_accuracy: 0.6394"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "338/625 [===============>..............] - ETA: 27s - loss: 0.6828 - binary_accuracy: 0.6405"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "339/625 [===============>..............] - ETA: 27s - loss: 0.6828 - binary_accuracy: 0.6412"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "343/625 [===============>..............] - ETA: 26s - loss: 0.6826 - binary_accuracy: 0.6424"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "348/625 [===============>..............] - ETA: 25s - loss: 0.6824 - binary_accuracy: 0.6432"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "349/625 [===============>..............] - ETA: 25s - loss: 0.6823 - binary_accuracy: 0.6433"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "354/625 [===============>..............] - ETA: 25s - loss: 0.6822 - binary_accuracy: 0.6442"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "360/625 [================>.............] - ETA: 24s - loss: 0.6818 - binary_accuracy: 0.6467"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "369/625 [================>.............] - ETA: 23s - loss: 0.6813 - binary_accuracy: 0.6489"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "370/625 [================>.............] - ETA: 23s - loss: 0.6813 - binary_accuracy: 0.6491"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "374/625 [================>.............] - ETA: 22s - loss: 0.6811 - binary_accuracy: 0.6502"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "378/625 [=================>............] - ETA: 22s - loss: 0.6808 - binary_accuracy: 0.6517"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "380/625 [=================>............] - ETA: 21s - loss: 0.6807 - binary_accuracy: 0.6528"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "390/625 [=================>............] - ETA: 20s - loss: 0.6800 - binary_accuracy: 0.6558"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "391/625 [=================>............] - ETA: 20s - loss: 0.6800 - binary_accuracy: 0.6563"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "392/625 [=================>............] - ETA: 20s - loss: 0.6799 - binary_accuracy: 0.6567"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "394/625 [=================>............] - ETA: 20s - loss: 0.6797 - binary_accuracy: 0.6574"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "401/625 [==================>...........] - ETA: 19s - loss: 0.6794 - binary_accuracy: 0.6584"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "403/625 [==================>...........] - ETA: 19s - loss: 0.6792 - binary_accuracy: 0.6590"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "407/625 [==================>...........] - ETA: 18s - loss: 0.6789 - binary_accuracy: 0.6604"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "409/625 [==================>...........] - ETA: 18s - loss: 0.6788 - binary_accuracy: 0.6609"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "410/625 [==================>...........] - ETA: 18s - loss: 0.6787 - binary_accuracy: 0.6614"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "418/625 [===================>..........] - ETA: 17s - loss: 0.6782 - binary_accuracy: 0.6634"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "421/625 [===================>..........] - ETA: 17s - loss: 0.6780 - binary_accuracy: 0.6642"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "427/625 [===================>..........] - ETA: 16s - loss: 0.6776 - binary_accuracy: 0.6660"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "437/625 [===================>..........] - ETA: 15s - loss: 0.6771 - binary_accuracy: 0.6674"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "445/625 [====================>.........] - ETA: 14s - loss: 0.6765 - binary_accuracy: 0.6695"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "446/625 [====================>.........] - ETA: 14s - loss: 0.6765 - binary_accuracy: 0.6697"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "447/625 [====================>.........] - ETA: 14s - loss: 0.6764 - binary_accuracy: 0.6697"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "451/625 [====================>.........] - ETA: 14s - loss: 0.6761 - binary_accuracy: 0.6703"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "454/625 [====================>.........] - ETA: 13s - loss: 0.6759 - binary_accuracy: 0.6710"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "456/625 [====================>.........] - ETA: 13s - loss: 0.6758 - binary_accuracy: 0.6713"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "464/625 [=====================>........] - ETA: 12s - loss: 0.6754 - binary_accuracy: 0.6724"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "465/625 [=====================>........] - ETA: 12s - loss: 0.6752 - binary_accuracy: 0.6730"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "470/625 [=====================>........] - ETA: 12s - loss: 0.6750 - binary_accuracy: 0.6735"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "474/625 [=====================>........] - ETA: 12s - loss: 0.6748 - binary_accuracy: 0.6738"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "488/625 [======================>.......] - ETA: 10s - loss: 0.6737 - binary_accuracy: 0.6765"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "490/625 [======================>.......] - ETA: 10s - loss: 0.6735 - binary_accuracy: 0.6770"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "497/625 [======================>.......] - ETA: 9s - loss: 0.6730 - binary_accuracy: 0.6784 "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "499/625 [======================>.......] - ETA: 9s - loss: 0.6729 - binary_accuracy: 0.6787"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "509/625 [=======================>......] - ETA: 8s - loss: 0.6721 - binary_accuracy: 0.6804"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "513/625 [=======================>......] - ETA: 8s - loss: 0.6719 - binary_accuracy: 0.6804"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "521/625 [========================>.....] - ETA: 7s - loss: 0.6713 - binary_accuracy: 0.6823"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "524/625 [========================>.....] - ETA: 7s - loss: 0.6712 - binary_accuracy: 0.6821"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "526/625 [========================>.....] - ETA: 7s - loss: 0.6710 - binary_accuracy: 0.6827"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "530/625 [========================>.....] - ETA: 7s - loss: 0.6707 - binary_accuracy: 0.6834"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "533/625 [========================>.....] - ETA: 6s - loss: 0.6705 - binary_accuracy: 0.6835"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "535/625 [========================>.....] - ETA: 6s - loss: 0.6704 - binary_accuracy: 0.6835"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "538/625 [========================>.....] - ETA: 6s - loss: 0.6703 - binary_accuracy: 0.6840"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "541/625 [========================>.....] - ETA: 6s - loss: 0.6701 - binary_accuracy: 0.6846"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "544/625 [=========================>....] - ETA: 5s - loss: 0.6699 - binary_accuracy: 0.6850"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "546/625 [=========================>....] - ETA: 5s - loss: 0.6697 - binary_accuracy: 0.6853"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "547/625 [=========================>....] - ETA: 5s - loss: 0.6696 - binary_accuracy: 0.6858"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "560/625 [=========================>....] - ETA: 4s - loss: 0.6686 - binary_accuracy: 0.6872"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "564/625 [==========================>...] - ETA: 4s - loss: 0.6684 - binary_accuracy: 0.6876"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "572/625 [==========================>...] - ETA: 3s - loss: 0.6677 - binary_accuracy: 0.6892"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "577/625 [==========================>...] - ETA: 3s - loss: 0.6675 - binary_accuracy: 0.6895"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "582/625 [==========================>...] - ETA: 3s - loss: 0.6671 - binary_accuracy: 0.6899"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "591/625 [===========================>..] - ETA: 2s - loss: 0.6664 - binary_accuracy: 0.6913"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "600/625 [===========================>..] - ETA: 1s - loss: 0.6656 - binary_accuracy: 0.6930"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "602/625 [===========================>..] - ETA: 1s - loss: 0.6654 - binary_accuracy: 0.6934"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "608/625 [============================>.] - ETA: 1s - loss: 0.6649 - binary_accuracy: 0.6942"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "609/625 [============================>.] - ETA: 1s - loss: 0.6648 - binary_accuracy: 0.6943"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "615/625 [============================>.] - ETA: 0s - loss: 0.6643 - binary_accuracy: 0.6951"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "619/625 [============================>.] - ETA: 0s - loss: 0.6641 - binary_accuracy: 0.6953"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 45s 69ms/step - loss: 0.6636 - binary_accuracy: 0.6962 - val_loss: 0.6149 - val_binary_accuracy: 0.7746\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.5712 - binary_accuracy: 0.8750"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.6111 - binary_accuracy: 0.7726"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 35/625 [>.............................] - ETA: 1s - loss: 0.6096 - binary_accuracy: 0.7634"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 53/625 [=>............................] - ETA: 1s - loss: 0.6059 - binary_accuracy: 0.7677"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 70/625 [==>...........................] - ETA: 1s - loss: 0.6063 - binary_accuracy: 0.7683"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 87/625 [===>..........................] - ETA: 1s - loss: 0.6026 - binary_accuracy: 0.7723"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "104/625 [===>..........................] - ETA: 1s - loss: 0.6010 - binary_accuracy: 0.7725"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "121/625 [====>.........................] - ETA: 1s - loss: 0.6002 - binary_accuracy: 0.7738"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "138/625 [=====>........................] - ETA: 1s - loss: 0.5983 - binary_accuracy: 0.7758"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "155/625 [======>.......................] - ETA: 1s - loss: 0.5972 - binary_accuracy: 0.7756"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "172/625 [=======>......................] - ETA: 1s - loss: 0.5953 - binary_accuracy: 0.7773"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "189/625 [========>.....................] - ETA: 1s - loss: 0.5933 - binary_accuracy: 0.7781"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "206/625 [========>.....................] - ETA: 1s - loss: 0.5924 - binary_accuracy: 0.7767"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "223/625 [=========>....................] - ETA: 1s - loss: 0.5894 - binary_accuracy: 0.7793"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "240/625 [==========>...................] - ETA: 1s - loss: 0.5877 - binary_accuracy: 0.7803"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "257/625 [===========>..................] - ETA: 1s - loss: 0.5856 - binary_accuracy: 0.7811"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "274/625 [============>.................] - ETA: 1s - loss: 0.5828 - binary_accuracy: 0.7847"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "291/625 [============>.................] - ETA: 1s - loss: 0.5817 - binary_accuracy: 0.7846"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "308/625 [=============>................] - ETA: 0s - loss: 0.5802 - binary_accuracy: 0.7845"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "326/625 [==============>...............] - ETA: 0s - loss: 0.5791 - binary_accuracy: 0.7844"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "343/625 [===============>..............] - ETA: 0s - loss: 0.5775 - binary_accuracy: 0.7853"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "360/625 [================>.............] - ETA: 0s - loss: 0.5758 - binary_accuracy: 0.7865"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "377/625 [=================>............] - ETA: 0s - loss: 0.5740 - binary_accuracy: 0.7881"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "394/625 [=================>............] - ETA: 0s - loss: 0.5714 - binary_accuracy: 0.7902"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "412/625 [==================>...........] - ETA: 0s - loss: 0.5694 - binary_accuracy: 0.7911"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "429/625 [===================>..........] - ETA: 0s - loss: 0.5672 - binary_accuracy: 0.7925"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "446/625 [====================>.........] - ETA: 0s - loss: 0.5656 - binary_accuracy: 0.7932"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "464/625 [=====================>........] - ETA: 0s - loss: 0.5638 - binary_accuracy: 0.7938"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "481/625 [======================>.......] - ETA: 0s - loss: 0.5624 - binary_accuracy: 0.7948"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "499/625 [======================>.......] - ETA: 0s - loss: 0.5602 - binary_accuracy: 0.7967"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "517/625 [=======================>......] - ETA: 0s - loss: 0.5585 - binary_accuracy: 0.7979"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "535/625 [========================>.....] - ETA: 0s - loss: 0.5568 - binary_accuracy: 0.7982"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "552/625 [=========================>....] - ETA: 0s - loss: 0.5552 - binary_accuracy: 0.7994"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "570/625 [==========================>...] - ETA: 0s - loss: 0.5536 - binary_accuracy: 0.8003"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "588/625 [===========================>..] - ETA: 0s - loss: 0.5523 - binary_accuracy: 0.8005"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "605/625 [============================>.] - ETA: 0s - loss: 0.5506 - binary_accuracy: 0.8018"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "623/625 [============================>.] - ETA: 0s - loss: 0.5492 - binary_accuracy: 0.8018"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.5489 - binary_accuracy: 0.8019 - val_loss: 0.4983 - val_binary_accuracy: 0.8228\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.4384 - binary_accuracy: 0.8750"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 19/625 [..............................] - ETA: 1s - loss: 0.4872 - binary_accuracy: 0.8289"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 36/625 [>.............................] - ETA: 1s - loss: 0.4855 - binary_accuracy: 0.8247"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 54/625 [=>............................] - ETA: 1s - loss: 0.4836 - binary_accuracy: 0.8287"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 71/625 [==>...........................] - ETA: 1s - loss: 0.4845 - binary_accuracy: 0.8279"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 89/625 [===>..........................] - ETA: 1s - loss: 0.4811 - binary_accuracy: 0.8297"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "106/625 [====>.........................] - ETA: 1s - loss: 0.4803 - binary_accuracy: 0.8293"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "124/625 [====>.........................] - ETA: 1s - loss: 0.4787 - binary_accuracy: 0.8286"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "142/625 [=====>........................] - ETA: 1s - loss: 0.4778 - binary_accuracy: 0.8290"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "160/625 [======>.......................] - ETA: 1s - loss: 0.4790 - binary_accuracy: 0.8268"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "178/625 [=======>......................] - ETA: 1s - loss: 0.4765 - binary_accuracy: 0.8288"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "196/625 [========>.....................] - ETA: 1s - loss: 0.4766 - binary_accuracy: 0.8273"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "214/625 [=========>....................] - ETA: 1s - loss: 0.4750 - binary_accuracy: 0.8296"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "231/625 [==========>...................] - ETA: 1s - loss: 0.4727 - binary_accuracy: 0.8312"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "249/625 [==========>...................] - ETA: 1s - loss: 0.4707 - binary_accuracy: 0.8316"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "267/625 [===========>..................] - ETA: 1s - loss: 0.4682 - binary_accuracy: 0.8333"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "285/625 [============>.................] - ETA: 0s - loss: 0.4670 - binary_accuracy: 0.8339"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "303/625 [=============>................] - ETA: 0s - loss: 0.4670 - binary_accuracy: 0.8331"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "321/625 [==============>...............] - ETA: 0s - loss: 0.4666 - binary_accuracy: 0.8324"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "339/625 [===============>..............] - ETA: 0s - loss: 0.4653 - binary_accuracy: 0.8341"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "357/625 [================>.............] - ETA: 0s - loss: 0.4640 - binary_accuracy: 0.8345"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "374/625 [================>.............] - ETA: 0s - loss: 0.4623 - binary_accuracy: 0.8363"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "392/625 [=================>............] - ETA: 0s - loss: 0.4602 - binary_accuracy: 0.8375"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "410/625 [==================>...........] - ETA: 0s - loss: 0.4580 - binary_accuracy: 0.8393"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "428/625 [===================>..........] - ETA: 0s - loss: 0.4562 - binary_accuracy: 0.8402"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "446/625 [====================>.........] - ETA: 0s - loss: 0.4548 - binary_accuracy: 0.8407"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "463/625 [=====================>........] - ETA: 0s - loss: 0.4537 - binary_accuracy: 0.8414"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "480/625 [======================>.......] - ETA: 0s - loss: 0.4534 - binary_accuracy: 0.8417"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "498/625 [======================>.......] - ETA: 0s - loss: 0.4517 - binary_accuracy: 0.8427"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "516/625 [=======================>......] - ETA: 0s - loss: 0.4501 - binary_accuracy: 0.8439"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "534/625 [========================>.....] - ETA: 0s - loss: 0.4492 - binary_accuracy: 0.8440"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "552/625 [=========================>....] - ETA: 0s - loss: 0.4483 - binary_accuracy: 0.8441"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "570/625 [==========================>...] - ETA: 0s - loss: 0.4472 - binary_accuracy: 0.8445"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "588/625 [===========================>..] - ETA: 0s - loss: 0.4467 - binary_accuracy: 0.8444"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "606/625 [============================>.] - ETA: 0s - loss: 0.4458 - binary_accuracy: 0.8450"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "624/625 [============================>.] - ETA: 0s - loss: 0.4450 - binary_accuracy: 0.8450"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.4449 - binary_accuracy: 0.8450 - val_loss: 0.4199 - val_binary_accuracy: 0.8468\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.3701 - binary_accuracy: 0.9062"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.3997 - binary_accuracy: 0.8663"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 35/625 [>.............................] - ETA: 1s - loss: 0.4029 - binary_accuracy: 0.8554"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 52/625 [=>............................] - ETA: 1s - loss: 0.4006 - binary_accuracy: 0.8582"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 69/625 [==>...........................] - ETA: 1s - loss: 0.4020 - binary_accuracy: 0.8591"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 85/625 [===>..........................] - ETA: 1s - loss: 0.3984 - binary_accuracy: 0.8603"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "102/625 [===>..........................] - ETA: 1s - loss: 0.4004 - binary_accuracy: 0.8585"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "119/625 [====>.........................] - ETA: 1s - loss: 0.3995 - binary_accuracy: 0.8585"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "137/625 [=====>........................] - ETA: 1s - loss: 0.3973 - binary_accuracy: 0.8586"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "155/625 [======>.......................] - ETA: 1s - loss: 0.3993 - binary_accuracy: 0.8565"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "172/625 [=======>......................] - ETA: 1s - loss: 0.3975 - binary_accuracy: 0.8579"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "189/625 [========>.....................] - ETA: 1s - loss: 0.3959 - binary_accuracy: 0.8588"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "207/625 [========>.....................] - ETA: 1s - loss: 0.3964 - binary_accuracy: 0.8582"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "224/625 [=========>....................] - ETA: 1s - loss: 0.3955 - binary_accuracy: 0.8587"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "242/625 [==========>...................] - ETA: 1s - loss: 0.3947 - binary_accuracy: 0.8580"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "259/625 [===========>..................] - ETA: 1s - loss: 0.3930 - binary_accuracy: 0.8599"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "276/625 [============>.................] - ETA: 1s - loss: 0.3909 - binary_accuracy: 0.8624"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "293/625 [=============>................] - ETA: 0s - loss: 0.3926 - binary_accuracy: 0.8604"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "310/625 [=============>................] - ETA: 0s - loss: 0.3921 - binary_accuracy: 0.8603"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "327/625 [==============>...............] - ETA: 0s - loss: 0.3919 - binary_accuracy: 0.8610"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "343/625 [===============>..............] - ETA: 0s - loss: 0.3909 - binary_accuracy: 0.8606"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "360/625 [================>.............] - ETA: 0s - loss: 0.3900 - binary_accuracy: 0.8615"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "377/625 [=================>............] - ETA: 0s - loss: 0.3890 - binary_accuracy: 0.8616"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "394/625 [=================>............] - ETA: 0s - loss: 0.3869 - binary_accuracy: 0.8632"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "410/625 [==================>...........] - ETA: 0s - loss: 0.3858 - binary_accuracy: 0.8640"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "426/625 [===================>..........] - ETA: 0s - loss: 0.3847 - binary_accuracy: 0.8647"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "443/625 [====================>.........] - ETA: 0s - loss: 0.3834 - binary_accuracy: 0.8651"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "460/625 [=====================>........] - ETA: 0s - loss: 0.3829 - binary_accuracy: 0.8657"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "477/625 [=====================>........] - ETA: 0s - loss: 0.3831 - binary_accuracy: 0.8656"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "494/625 [======================>.......] - ETA: 0s - loss: 0.3815 - binary_accuracy: 0.8667"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "511/625 [=======================>......] - ETA: 0s - loss: 0.3804 - binary_accuracy: 0.8671"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "528/625 [========================>.....] - ETA: 0s - loss: 0.3795 - binary_accuracy: 0.8670"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "545/625 [=========================>....] - ETA: 0s - loss: 0.3799 - binary_accuracy: 0.8668"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "562/625 [=========================>....] - ETA: 0s - loss: 0.3787 - binary_accuracy: 0.8675"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "580/625 [==========================>...] - ETA: 0s - loss: 0.3788 - binary_accuracy: 0.8670"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "598/625 [===========================>..] - ETA: 0s - loss: 0.3782 - binary_accuracy: 0.8674"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "616/625 [============================>.] - ETA: 0s - loss: 0.3781 - binary_accuracy: 0.8670"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.3780 - binary_accuracy: 0.8670 - val_loss: 0.3733 - val_binary_accuracy: 0.8608\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.3485 - binary_accuracy: 0.9062"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.3464 - binary_accuracy: 0.8802"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 36/625 [>.............................] - ETA: 1s - loss: 0.3490 - binary_accuracy: 0.8724"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 54/625 [=>............................] - ETA: 1s - loss: 0.3480 - binary_accuracy: 0.8738"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 71/625 [==>...........................] - ETA: 1s - loss: 0.3497 - binary_accuracy: 0.8737"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 89/625 [===>..........................] - ETA: 1s - loss: 0.3469 - binary_accuracy: 0.8743"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "107/625 [====>.........................] - ETA: 1s - loss: 0.3488 - binary_accuracy: 0.8718"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "125/625 [=====>........................] - ETA: 1s - loss: 0.3479 - binary_accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "143/625 [=====>........................] - ETA: 1s - loss: 0.3472 - binary_accuracy: 0.8711"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "161/625 [======>.......................] - ETA: 1s - loss: 0.3497 - binary_accuracy: 0.8696"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "179/625 [=======>......................] - ETA: 1s - loss: 0.3467 - binary_accuracy: 0.8724"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "197/625 [========>.....................] - ETA: 1s - loss: 0.3485 - binary_accuracy: 0.8707"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "215/625 [=========>....................] - ETA: 1s - loss: 0.3477 - binary_accuracy: 0.8725"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "232/625 [==========>...................] - ETA: 1s - loss: 0.3471 - binary_accuracy: 0.8728"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "249/625 [==========>...................] - ETA: 1s - loss: 0.3457 - binary_accuracy: 0.8724"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "266/625 [===========>..................] - ETA: 1s - loss: 0.3440 - binary_accuracy: 0.8746"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "283/625 [============>.................] - ETA: 0s - loss: 0.3434 - binary_accuracy: 0.8746"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "300/625 [=============>................] - ETA: 0s - loss: 0.3455 - binary_accuracy: 0.8730"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "318/625 [==============>...............] - ETA: 0s - loss: 0.3453 - binary_accuracy: 0.8729"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "335/625 [===============>..............] - ETA: 0s - loss: 0.3447 - binary_accuracy: 0.8729"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "352/625 [===============>..............] - ETA: 0s - loss: 0.3444 - binary_accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "369/625 [================>.............] - ETA: 0s - loss: 0.3433 - binary_accuracy: 0.8733"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "387/625 [=================>............] - ETA: 0s - loss: 0.3427 - binary_accuracy: 0.8740"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "405/625 [==================>...........] - ETA: 0s - loss: 0.3410 - binary_accuracy: 0.8749"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "423/625 [===================>..........] - ETA: 0s - loss: 0.3398 - binary_accuracy: 0.8760"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "440/625 [====================>.........] - ETA: 0s - loss: 0.3386 - binary_accuracy: 0.8764"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "458/625 [====================>.........] - ETA: 0s - loss: 0.3385 - binary_accuracy: 0.8766"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "476/625 [=====================>........] - ETA: 0s - loss: 0.3388 - binary_accuracy: 0.8768"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "493/625 [======================>.......] - ETA: 0s - loss: 0.3370 - binary_accuracy: 0.8782"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "510/625 [=======================>......] - ETA: 0s - loss: 0.3359 - binary_accuracy: 0.8789"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "528/625 [========================>.....] - ETA: 0s - loss: 0.3357 - binary_accuracy: 0.8786"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "545/625 [=========================>....] - ETA: 0s - loss: 0.3362 - binary_accuracy: 0.8781"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "563/625 [==========================>...] - ETA: 0s - loss: 0.3351 - binary_accuracy: 0.8786"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "580/625 [==========================>...] - ETA: 0s - loss: 0.3353 - binary_accuracy: 0.8782"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "598/625 [===========================>..] - ETA: 0s - loss: 0.3352 - binary_accuracy: 0.8785"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "615/625 [============================>.] - ETA: 0s - loss: 0.3352 - binary_accuracy: 0.8787"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.3351 - binary_accuracy: 0.8784 - val_loss: 0.3444 - val_binary_accuracy: 0.8672\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 6/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.3013 - binary_accuracy: 0.9062"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.3093 - binary_accuracy: 0.9010"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 35/625 [>.............................] - ETA: 1s - loss: 0.3125 - binary_accuracy: 0.8884"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 52/625 [=>............................] - ETA: 1s - loss: 0.3131 - binary_accuracy: 0.8870"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 69/625 [==>...........................] - ETA: 1s - loss: 0.3144 - binary_accuracy: 0.8877"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 87/625 [===>..........................] - ETA: 1s - loss: 0.3128 - binary_accuracy: 0.8869"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "104/625 [===>..........................] - ETA: 1s - loss: 0.3164 - binary_accuracy: 0.8846"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "121/625 [====>.........................] - ETA: 1s - loss: 0.3157 - binary_accuracy: 0.8856"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "139/625 [=====>........................] - ETA: 1s - loss: 0.3131 - binary_accuracy: 0.8858"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "156/625 [======>.......................] - ETA: 1s - loss: 0.3162 - binary_accuracy: 0.8836"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "173/625 [=======>......................] - ETA: 1s - loss: 0.3139 - binary_accuracy: 0.8851"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "190/625 [========>.....................] - ETA: 1s - loss: 0.3128 - binary_accuracy: 0.8854"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "207/625 [========>.....................] - ETA: 1s - loss: 0.3141 - binary_accuracy: 0.8845"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "224/625 [=========>....................] - ETA: 1s - loss: 0.3136 - binary_accuracy: 0.8846"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "241/625 [==========>...................] - ETA: 1s - loss: 0.3138 - binary_accuracy: 0.8842"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "259/625 [===========>..................] - ETA: 1s - loss: 0.3123 - binary_accuracy: 0.8859"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "277/625 [============>.................] - ETA: 1s - loss: 0.3103 - binary_accuracy: 0.8875"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "294/625 [=============>................] - ETA: 0s - loss: 0.3125 - binary_accuracy: 0.8858"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "311/625 [=============>................] - ETA: 0s - loss: 0.3123 - binary_accuracy: 0.8859"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "328/625 [==============>...............] - ETA: 0s - loss: 0.3119 - binary_accuracy: 0.8858"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "345/625 [===============>..............] - ETA: 0s - loss: 0.3114 - binary_accuracy: 0.8858"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "362/625 [================>.............] - ETA: 0s - loss: 0.3104 - binary_accuracy: 0.8860"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "379/625 [=================>............] - ETA: 0s - loss: 0.3094 - binary_accuracy: 0.8861"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "396/625 [==================>...........] - ETA: 0s - loss: 0.3085 - binary_accuracy: 0.8865"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "414/625 [==================>...........] - ETA: 0s - loss: 0.3075 - binary_accuracy: 0.8870"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "432/625 [===================>..........] - ETA: 0s - loss: 0.3060 - binary_accuracy: 0.8877"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "450/625 [====================>.........] - ETA: 0s - loss: 0.3058 - binary_accuracy: 0.8874"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "468/625 [=====================>........] - ETA: 0s - loss: 0.3048 - binary_accuracy: 0.8884"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "486/625 [======================>.......] - ETA: 0s - loss: 0.3055 - binary_accuracy: 0.8883"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "504/625 [=======================>......] - ETA: 0s - loss: 0.3043 - binary_accuracy: 0.8888"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "522/625 [========================>.....] - ETA: 0s - loss: 0.3035 - binary_accuracy: 0.8891"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "539/625 [========================>.....] - ETA: 0s - loss: 0.3039 - binary_accuracy: 0.8886"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "557/625 [=========================>....] - ETA: 0s - loss: 0.3035 - binary_accuracy: 0.8890"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "575/625 [==========================>...] - ETA: 0s - loss: 0.3034 - binary_accuracy: 0.8892"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "592/625 [===========================>..] - ETA: 0s - loss: 0.3040 - binary_accuracy: 0.8892"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "610/625 [============================>.] - ETA: 0s - loss: 0.3047 - binary_accuracy: 0.8890"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.3042 - binary_accuracy: 0.8891 - val_loss: 0.3255 - val_binary_accuracy: 0.8724\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 7/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.2705 - binary_accuracy: 0.9062"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.2781 - binary_accuracy: 0.9149"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 36/625 [>.............................] - ETA: 1s - loss: 0.2822 - binary_accuracy: 0.8984"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 54/625 [=>............................] - ETA: 1s - loss: 0.2841 - binary_accuracy: 0.8993"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 72/625 [==>...........................] - ETA: 1s - loss: 0.2865 - binary_accuracy: 0.8984"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 90/625 [===>..........................] - ETA: 1s - loss: 0.2853 - binary_accuracy: 0.8990"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "108/625 [====>.........................] - ETA: 1s - loss: 0.2890 - binary_accuracy: 0.8955"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "126/625 [=====>........................] - ETA: 1s - loss: 0.2882 - binary_accuracy: 0.8963"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "144/625 [=====>........................] - ETA: 1s - loss: 0.2879 - binary_accuracy: 0.8952"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "161/625 [======>.......................] - ETA: 1s - loss: 0.2904 - binary_accuracy: 0.8942"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "178/625 [=======>......................] - ETA: 1s - loss: 0.2882 - binary_accuracy: 0.8947"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "196/625 [========>.....................] - ETA: 1s - loss: 0.2892 - binary_accuracy: 0.8935"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "214/625 [=========>....................] - ETA: 1s - loss: 0.2881 - binary_accuracy: 0.8952"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "231/625 [==========>...................] - ETA: 1s - loss: 0.2879 - binary_accuracy: 0.8946"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "249/625 [==========>...................] - ETA: 1s - loss: 0.2870 - binary_accuracy: 0.8945"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "267/625 [===========>..................] - ETA: 1s - loss: 0.2860 - binary_accuracy: 0.8967"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "285/625 [============>.................] - ETA: 0s - loss: 0.2864 - binary_accuracy: 0.8969"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "302/625 [=============>................] - ETA: 0s - loss: 0.2877 - binary_accuracy: 0.8954"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "319/625 [==============>...............] - ETA: 0s - loss: 0.2879 - binary_accuracy: 0.8954"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "336/625 [===============>..............] - ETA: 0s - loss: 0.2871 - binary_accuracy: 0.8952"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "353/625 [===============>..............] - ETA: 0s - loss: 0.2865 - binary_accuracy: 0.8950"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "371/625 [================>.............] - ETA: 0s - loss: 0.2853 - binary_accuracy: 0.8956"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "388/625 [=================>............] - ETA: 0s - loss: 0.2844 - binary_accuracy: 0.8960"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "406/625 [==================>...........] - ETA: 0s - loss: 0.2831 - binary_accuracy: 0.8967"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "424/625 [===================>..........] - ETA: 0s - loss: 0.2824 - binary_accuracy: 0.8973"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "442/625 [====================>.........] - ETA: 0s - loss: 0.2809 - binary_accuracy: 0.8979"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "460/625 [=====================>........] - ETA: 0s - loss: 0.2810 - binary_accuracy: 0.8978"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "478/625 [=====================>........] - ETA: 0s - loss: 0.2816 - binary_accuracy: 0.8977"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "496/625 [======================>.......] - ETA: 0s - loss: 0.2802 - binary_accuracy: 0.8982"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "514/625 [=======================>......] - ETA: 0s - loss: 0.2790 - binary_accuracy: 0.8988"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "532/625 [========================>.....] - ETA: 0s - loss: 0.2793 - binary_accuracy: 0.8981"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "549/625 [=========================>....] - ETA: 0s - loss: 0.2792 - binary_accuracy: 0.8985"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "566/625 [==========================>...] - ETA: 0s - loss: 0.2789 - binary_accuracy: 0.8988"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "584/625 [===========================>..] - ETA: 0s - loss: 0.2797 - binary_accuracy: 0.8986"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "601/625 [===========================>..] - ETA: 0s - loss: 0.2800 - binary_accuracy: 0.8985"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "618/625 [============================>.] - ETA: 0s - loss: 0.2806 - binary_accuracy: 0.8980"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.2800 - binary_accuracy: 0.8985 - val_loss: 0.3123 - val_binary_accuracy: 0.8740\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 8/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.2993 - binary_accuracy: 0.9062"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.2619 - binary_accuracy: 0.9132"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 36/625 [>.............................] - ETA: 1s - loss: 0.2665 - binary_accuracy: 0.9054"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 54/625 [=>............................] - ETA: 1s - loss: 0.2650 - binary_accuracy: 0.9057"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 71/625 [==>...........................] - ETA: 1s - loss: 0.2674 - binary_accuracy: 0.9045"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 88/625 [===>..........................] - ETA: 1s - loss: 0.2649 - binary_accuracy: 0.9052"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "106/625 [====>.........................] - ETA: 1s - loss: 0.2696 - binary_accuracy: 0.9021"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "123/625 [====>.........................] - ETA: 1s - loss: 0.2678 - binary_accuracy: 0.9042"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "140/625 [=====>........................] - ETA: 1s - loss: 0.2660 - binary_accuracy: 0.9025"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "157/625 [======>.......................] - ETA: 1s - loss: 0.2688 - binary_accuracy: 0.9013"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "174/625 [=======>......................] - ETA: 1s - loss: 0.2666 - binary_accuracy: 0.9023"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "191/625 [========>.....................] - ETA: 1s - loss: 0.2659 - binary_accuracy: 0.9023"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "209/625 [=========>....................] - ETA: 1s - loss: 0.2675 - binary_accuracy: 0.9018"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "227/625 [=========>....................] - ETA: 1s - loss: 0.2683 - binary_accuracy: 0.9014"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "245/625 [==========>...................] - ETA: 1s - loss: 0.2679 - binary_accuracy: 0.9010"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "262/625 [===========>..................] - ETA: 1s - loss: 0.2672 - binary_accuracy: 0.9017"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "279/625 [============>.................] - ETA: 1s - loss: 0.2653 - binary_accuracy: 0.9032"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "296/625 [=============>................] - ETA: 0s - loss: 0.2678 - binary_accuracy: 0.9018"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "313/625 [==============>...............] - ETA: 0s - loss: 0.2678 - binary_accuracy: 0.9017"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "330/625 [==============>...............] - ETA: 0s - loss: 0.2673 - binary_accuracy: 0.9013"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "347/625 [===============>..............] - ETA: 0s - loss: 0.2667 - binary_accuracy: 0.9012"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "364/625 [================>.............] - ETA: 0s - loss: 0.2656 - binary_accuracy: 0.9023"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "381/625 [=================>............] - ETA: 0s - loss: 0.2651 - binary_accuracy: 0.9027"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "398/625 [==================>...........] - ETA: 0s - loss: 0.2644 - binary_accuracy: 0.9023"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "415/625 [==================>...........] - ETA: 0s - loss: 0.2637 - binary_accuracy: 0.9032"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "432/625 [===================>..........] - ETA: 0s - loss: 0.2626 - binary_accuracy: 0.9034"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "449/625 [====================>.........] - ETA: 0s - loss: 0.2623 - binary_accuracy: 0.9033"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "466/625 [=====================>........] - ETA: 0s - loss: 0.2617 - binary_accuracy: 0.9036"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "483/625 [======================>.......] - ETA: 0s - loss: 0.2625 - binary_accuracy: 0.9033"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "500/625 [=======================>......] - ETA: 0s - loss: 0.2611 - binary_accuracy: 0.9042"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "517/625 [=======================>......] - ETA: 0s - loss: 0.2605 - binary_accuracy: 0.9044"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "534/625 [========================>.....] - ETA: 0s - loss: 0.2608 - binary_accuracy: 0.9040"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "551/625 [=========================>....] - ETA: 0s - loss: 0.2609 - binary_accuracy: 0.9042"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "568/625 [==========================>...] - ETA: 0s - loss: 0.2604 - binary_accuracy: 0.9044"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "585/625 [===========================>..] - ETA: 0s - loss: 0.2612 - binary_accuracy: 0.9042"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "602/625 [===========================>..] - ETA: 0s - loss: 0.2618 - binary_accuracy: 0.9039"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "619/625 [============================>.] - ETA: 0s - loss: 0.2623 - binary_accuracy: 0.9034"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.2619 - binary_accuracy: 0.9036 - val_loss: 0.3028 - val_binary_accuracy: 0.8758\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.2288 - binary_accuracy: 0.9062"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.2484 - binary_accuracy: 0.9219"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 35/625 [>.............................] - ETA: 1s - loss: 0.2469 - binary_accuracy: 0.9161"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 52/625 [=>............................] - ETA: 1s - loss: 0.2461 - binary_accuracy: 0.9153"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 69/625 [==>...........................] - ETA: 1s - loss: 0.2471 - binary_accuracy: 0.9139"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 86/625 [===>..........................] - ETA: 1s - loss: 0.2454 - binary_accuracy: 0.9135"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "103/625 [===>..........................] - ETA: 1s - loss: 0.2510 - binary_accuracy: 0.9108"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "120/625 [====>.........................] - ETA: 1s - loss: 0.2505 - binary_accuracy: 0.9109"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "136/625 [=====>........................] - ETA: 1s - loss: 0.2485 - binary_accuracy: 0.9092"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "154/625 [======>.......................] - ETA: 1s - loss: 0.2513 - binary_accuracy: 0.9075"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "171/625 [=======>......................] - ETA: 1s - loss: 0.2493 - binary_accuracy: 0.9090"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "188/625 [========>.....................] - ETA: 1s - loss: 0.2481 - binary_accuracy: 0.9097"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "205/625 [========>.....................] - ETA: 1s - loss: 0.2501 - binary_accuracy: 0.9095"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "222/625 [=========>....................] - ETA: 1s - loss: 0.2490 - binary_accuracy: 0.9099"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "239/625 [==========>...................] - ETA: 1s - loss: 0.2503 - binary_accuracy: 0.9089"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "256/625 [===========>..................] - ETA: 1s - loss: 0.2498 - binary_accuracy: 0.9091"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "274/625 [============>.................] - ETA: 1s - loss: 0.2479 - binary_accuracy: 0.9113"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "291/625 [============>.................] - ETA: 1s - loss: 0.2504 - binary_accuracy: 0.9100"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "308/625 [=============>................] - ETA: 0s - loss: 0.2506 - binary_accuracy: 0.9098"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "326/625 [==============>...............] - ETA: 0s - loss: 0.2496 - binary_accuracy: 0.9096"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "343/625 [===============>..............] - ETA: 0s - loss: 0.2484 - binary_accuracy: 0.9099"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "361/625 [================>.............] - ETA: 0s - loss: 0.2482 - binary_accuracy: 0.9095"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "378/625 [=================>............] - ETA: 0s - loss: 0.2478 - binary_accuracy: 0.9096"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "395/625 [=================>............] - ETA: 0s - loss: 0.2468 - binary_accuracy: 0.9097"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "412/625 [==================>...........] - ETA: 0s - loss: 0.2465 - binary_accuracy: 0.9103"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "429/625 [===================>..........] - ETA: 0s - loss: 0.2455 - binary_accuracy: 0.9104"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "446/625 [====================>.........] - ETA: 0s - loss: 0.2446 - binary_accuracy: 0.9105"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "463/625 [=====================>........] - ETA: 0s - loss: 0.2445 - binary_accuracy: 0.9108"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "480/625 [======================>.......] - ETA: 0s - loss: 0.2452 - binary_accuracy: 0.9102"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "497/625 [======================>.......] - ETA: 0s - loss: 0.2442 - binary_accuracy: 0.9104"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "514/625 [=======================>......] - ETA: 0s - loss: 0.2429 - binary_accuracy: 0.9113"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "532/625 [========================>.....] - ETA: 0s - loss: 0.2434 - binary_accuracy: 0.9108"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "549/625 [=========================>....] - ETA: 0s - loss: 0.2432 - binary_accuracy: 0.9107"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "566/625 [==========================>...] - ETA: 0s - loss: 0.2430 - binary_accuracy: 0.9112"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "583/625 [==========================>...] - ETA: 0s - loss: 0.2442 - binary_accuracy: 0.9109"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "600/625 [===========================>..] - ETA: 0s - loss: 0.2449 - binary_accuracy: 0.9104"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "617/625 [============================>.] - ETA: 0s - loss: 0.2454 - binary_accuracy: 0.9099"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.2449 - binary_accuracy: 0.9103 - val_loss: 0.2961 - val_binary_accuracy: 0.8780\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 10/10\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/625 [..............................] - ETA: 2s - loss: 0.2152 - binary_accuracy: 0.9375"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 18/625 [..............................] - ETA: 1s - loss: 0.2293 - binary_accuracy: 0.9219"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 35/625 [>.............................] - ETA: 1s - loss: 0.2287 - binary_accuracy: 0.9205"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 52/625 [=>............................] - ETA: 1s - loss: 0.2273 - binary_accuracy: 0.9207"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 69/625 [==>...........................] - ETA: 1s - loss: 0.2289 - binary_accuracy: 0.9207"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 86/625 [===>..........................] - ETA: 1s - loss: 0.2286 - binary_accuracy: 0.9186"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "103/625 [===>..........................] - ETA: 1s - loss: 0.2359 - binary_accuracy: 0.9166"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "120/625 [====>.........................] - ETA: 1s - loss: 0.2343 - binary_accuracy: 0.9174"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "137/625 [=====>........................] - ETA: 1s - loss: 0.2315 - binary_accuracy: 0.9158"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "154/625 [======>.......................] - ETA: 1s - loss: 0.2346 - binary_accuracy: 0.9144"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "171/625 [=======>......................] - ETA: 1s - loss: 0.2331 - binary_accuracy: 0.9147"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "188/625 [========>.....................] - ETA: 1s - loss: 0.2316 - binary_accuracy: 0.9151"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "205/625 [========>.....................] - ETA: 1s - loss: 0.2331 - binary_accuracy: 0.9155"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "223/625 [=========>....................] - ETA: 1s - loss: 0.2327 - binary_accuracy: 0.9159"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "241/625 [==========>...................] - ETA: 1s - loss: 0.2344 - binary_accuracy: 0.9149"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "259/625 [===========>..................] - ETA: 1s - loss: 0.2337 - binary_accuracy: 0.9154"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "277/625 [============>.................] - ETA: 1s - loss: 0.2322 - binary_accuracy: 0.9172"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "295/625 [=============>................] - ETA: 0s - loss: 0.2346 - binary_accuracy: 0.9162"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "312/625 [=============>................] - ETA: 0s - loss: 0.2348 - binary_accuracy: 0.9164"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "329/625 [==============>...............] - ETA: 0s - loss: 0.2342 - binary_accuracy: 0.9162"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "347/625 [===============>..............] - ETA: 0s - loss: 0.2335 - binary_accuracy: 0.9156"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "364/625 [================>.............] - ETA: 0s - loss: 0.2324 - binary_accuracy: 0.9165"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "382/625 [=================>............] - ETA: 0s - loss: 0.2320 - binary_accuracy: 0.9162"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "399/625 [==================>...........] - ETA: 0s - loss: 0.2317 - binary_accuracy: 0.9161"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "417/625 [===================>..........] - ETA: 0s - loss: 0.2310 - binary_accuracy: 0.9164"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "434/625 [===================>..........] - ETA: 0s - loss: 0.2301 - binary_accuracy: 0.9166"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "451/625 [====================>.........] - ETA: 0s - loss: 0.2297 - binary_accuracy: 0.9167"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "468/625 [=====================>........] - ETA: 0s - loss: 0.2289 - binary_accuracy: 0.9171"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "486/625 [======================>.......] - ETA: 0s - loss: 0.2299 - binary_accuracy: 0.9167"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "504/625 [=======================>......] - ETA: 0s - loss: 0.2288 - binary_accuracy: 0.9172"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "522/625 [========================>.....] - ETA: 0s - loss: 0.2280 - binary_accuracy: 0.9176"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "540/625 [========================>.....] - ETA: 0s - loss: 0.2284 - binary_accuracy: 0.9170"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "557/625 [=========================>....] - ETA: 0s - loss: 0.2282 - binary_accuracy: 0.9173"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "574/625 [==========================>...] - ETA: 0s - loss: 0.2287 - binary_accuracy: 0.9172"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "591/625 [===========================>..] - ETA: 0s - loss: 0.2296 - binary_accuracy: 0.9167"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "608/625 [============================>.] - ETA: 0s - loss: 0.2306 - binary_accuracy: 0.9162"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - ETA: 0s - loss: 0.2300 - binary_accuracy: 0.9164"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "625/625 [==============================] - 2s 3ms/step - loss: 0.2300 - binary_accuracy: 0.9164 - val_loss: 0.2913 - val_binary_accuracy: 0.8804\n"
     ]
    }
   ],
   "source": [
    "epochs = 10\n",
    "history = model.fit(\n",
    "    train_ds,\n",
    "    validation_data=val_ds,\n",
    "    epochs=epochs)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9EEGuDVuzb5r"
   },
   "source": [
    "### Evaluate the model\n",
    "\n",
    "Let's see how the model performs. Two values will be returned. Loss (a number which represents our error, lower values are better), and accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:10:06.864242Z",
     "iopub.status.busy": "2023-12-07T03:10:06.863673Z",
     "iopub.status.idle": "2023-12-07T03:10:08.515772Z",
     "shell.execute_reply": "2023-12-07T03:10:08.514951Z"
    },
    "id": "zOMKywn4zReN"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/782 [..............................] - ETA: 1:02 - loss: 0.2927 - binary_accuracy: 0.8125"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 27/782 [>.............................] - ETA: 1s - loss: 0.3137 - binary_accuracy: 0.8785  "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 52/782 [>.............................] - ETA: 1s - loss: 0.3088 - binary_accuracy: 0.8756"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 78/782 [=>............................] - ETA: 1s - loss: 0.3170 - binary_accuracy: 0.8754"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "104/782 [==>...........................] - ETA: 1s - loss: 0.3159 - binary_accuracy: 0.8720"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "130/782 [===>..........................] - ETA: 1s - loss: 0.3229 - binary_accuracy: 0.8649"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "156/782 [====>.........................] - ETA: 1s - loss: 0.3263 - binary_accuracy: 0.8628"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "182/782 [=====>........................] - ETA: 1s - loss: 0.3245 - binary_accuracy: 0.8630"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "208/782 [======>.......................] - ETA: 1s - loss: 0.3229 - binary_accuracy: 0.8645"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "233/782 [=======>......................] - ETA: 1s - loss: 0.3200 - binary_accuracy: 0.8667"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "258/782 [========>.....................] - ETA: 1s - loss: 0.3193 - binary_accuracy: 0.8676"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "283/782 [=========>....................] - ETA: 0s - loss: 0.3153 - binary_accuracy: 0.8693"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "307/782 [==========>...................] - ETA: 0s - loss: 0.3141 - binary_accuracy: 0.8702"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "332/782 [===========>..................] - ETA: 0s - loss: 0.3145 - binary_accuracy: 0.8702"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "357/782 [============>.................] - ETA: 0s - loss: 0.3124 - binary_accuracy: 0.8709"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "382/782 [=============>................] - ETA: 0s - loss: 0.3134 - binary_accuracy: 0.8703"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "408/782 [==============>...............] - ETA: 0s - loss: 0.3123 - binary_accuracy: 0.8712"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "433/782 [===============>..............] - ETA: 0s - loss: 0.3117 - binary_accuracy: 0.8715"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "459/782 [================>.............] - ETA: 0s - loss: 0.3099 - binary_accuracy: 0.8731"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "484/782 [=================>............] - ETA: 0s - loss: 0.3106 - binary_accuracy: 0.8724"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "510/782 [==================>...........] - ETA: 0s - loss: 0.3105 - binary_accuracy: 0.8728"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "536/782 [===================>..........] - ETA: 0s - loss: 0.3099 - binary_accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "562/782 [====================>.........] - ETA: 0s - loss: 0.3098 - binary_accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "588/782 [=====================>........] - ETA: 0s - loss: 0.3086 - binary_accuracy: 0.8738"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "614/782 [======================>.......] - ETA: 0s - loss: 0.3096 - binary_accuracy: 0.8733"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "640/782 [=======================>......] - ETA: 0s - loss: 0.3099 - binary_accuracy: 0.8729"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "666/782 [========================>.....] - ETA: 0s - loss: 0.3110 - binary_accuracy: 0.8724"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "692/782 [=========================>....] - ETA: 0s - loss: 0.3101 - binary_accuracy: 0.8731"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "718/782 [==========================>...] - ETA: 0s - loss: 0.3093 - binary_accuracy: 0.8732"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "743/782 [===========================>..] - ETA: 0s - loss: 0.3088 - binary_accuracy: 0.8738"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "769/782 [============================>.] - ETA: 0s - loss: 0.3098 - binary_accuracy: 0.8736"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "782/782 [==============================] - 2s 2ms/step - loss: 0.3098 - binary_accuracy: 0.8737\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loss:  0.30984994769096375\n",
      "Accuracy:  0.8737199902534485\n"
     ]
    }
   ],
   "source": [
    "loss, accuracy = model.evaluate(test_ds)\n",
    "\n",
    "print(\"Loss: \", loss)\n",
    "print(\"Accuracy: \", accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "z1iEXVTR0Z2t"
   },
   "source": [
    "This fairly naive approach achieves an accuracy of about 86%."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ldbQqCw2Xc1W"
   },
   "source": [
    "### Create a plot of accuracy and loss over time\n",
    "\n",
    "`model.fit()` returns a `History` object that contains a dictionary with everything that happened during training:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:10:08.519469Z",
     "iopub.status.busy": "2023-12-07T03:10:08.519186Z",
     "iopub.status.idle": "2023-12-07T03:10:08.523994Z",
     "shell.execute_reply": "2023-12-07T03:10:08.523313Z"
    },
    "id": "-YcvZsdvWfDf"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history_dict = history.history\n",
    "history_dict.keys()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1_CH32qJXruI"
   },
   "source": [
    "There are four entries: one for each monitored metric during training and validation. You can use these to plot the training and validation loss for comparison, as well as the training and validation accuracy:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:10:08.527546Z",
     "iopub.status.busy": "2023-12-07T03:10:08.526899Z",
     "iopub.status.idle": "2023-12-07T03:10:08.680269Z",
     "shell.execute_reply": "2023-12-07T03:10:08.679671Z"
    },
    "id": "2SEMeQ5YXs8z"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAABWPklEQVR4nO3de3zO9f/H8ce1zU5sw7ANsyHnnHL6shzKCklJNCWNim/IIemLr5xDkSJC+n6jbwcpjSSH8KWEolCShq9jmEPYHIdrn98fn98uWzvY+bNd1/N+u12363N9rs/hdW2r6+n9eX/eb5thGAYiIiIiTsLN6gJERERE8pLCjYiIiDgVhRsRERFxKgo3IiIi4lQUbkRERMSpKNyIiIiIU1G4EREREaeicCMiIiJOReFGREREnIrCjYgFevXqRXh4eI72HTduHDabLW8LKmQOHz6MzWZj4cKFBXrejRs3YrPZ2Lhxo2NdVn9X+VVzeHg4vXr1ytNjZsXChQux2WwcPny4wM8tklsKNyIp2Gy2LD1SfvmJ5NaWLVsYN24cFy5csLoUEafgYXUBIoXJBx98kOr1f/7zH9auXZtmfa1atXJ1nnfffZekpKQc7fvyyy8zYsSIXJ1fsi43v6us2rJlC+PHj6dXr16ULFky1XuxsbG4uenfoSLZoXAjksKTTz6Z6vX333/P2rVr06z/qytXruDr65vl8xQrVixH9QF4eHjg4aH/dAtKbn5XecHLy8vS84sURfrngEg2tWnThjvvvJOffvqJVq1a4evryz//+U8AvvjiCzp27Ej58uXx8vKiatWqTJw4EbvdnuoYf+3Hkdxf4/XXX2f+/PlUrVoVLy8vmjRpwvbt21Ptm16fG5vNxvPPP8+yZcu488478fLyok6dOqxevTpN/Rs3bqRx48Z4e3tTtWpV3nnnnSz349m0aRPdunWjUqVKeHl5ERoaygsvvMDVq1fTfL4SJUpw/PhxOnfuTIkSJShbtizDhg1L87O4cOECvXr1IiAggJIlSxIdHZ2lyzM//vgjNpuN999/P817a9aswWazsWLFCgCOHDlC//79qVGjBj4+PgQGBtKtW7cs9SdJr89NVmv+5Zdf6NWrF1WqVMHb25vg4GCefvpp/vzzT8c248aN46WXXgKgcuXKjkufybWl1+fm4MGDdOvWjdKlS+Pr68vf/vY3vvrqq1TbJPcf+vTTT5k0aRIVK1bE29ubtm3bcuDAgdt+7ozMmTOHOnXq4OXlRfny5RkwYECaz75//34effRRgoOD8fb2pmLFinTv3p34+HjHNmvXruXuu++mZMmSlChRgho1ajj+OxLJLf3zTyQH/vzzTzp06ED37t158sknCQoKAsxOmCVKlGDo0KGUKFGC//73v4wZM4aEhASmTZt22+N+/PHHXLx4kb///e/YbDamTp1Kly5dOHjw4G1bEL777jtiYmLo378/fn5+vPXWWzz66KMcPXqUwMBAAHbu3En79u0JCQlh/Pjx2O12JkyYQNmyZbP0uT/77DOuXLlCv379CAwMZNu2bcyaNYs//viDzz77LNW2druddu3a0axZM15//XXWrVvH9OnTqVq1Kv369QPAMAwefvhhvvvuO5577jlq1arF0qVLiY6Ovm0tjRs3pkqVKnz66adptl+8eDGlSpWiXbt2AGzfvp0tW7bQvXt3KlasyOHDh5k7dy5t2rTht99+y1arW3ZqXrt2LQcPHqR3794EBwezZ88e5s+fz549e/j++++x2Wx06dKFffv2sWjRIt58803KlCkDkOHv5NSpU7Ro0YIrV64waNAgAgMDef/993nooYdYsmQJjzzySKrtX331Vdzc3Bg2bBjx8fFMnTqVHj168MMPP2T5MycbN24c48ePJzIykn79+hEbG8vcuXPZvn07mzdvplixYly/fp127dqRmJjIwIEDCQ4O5vjx46xYsYILFy4QEBDAnj17ePDBB6lXrx4TJkzAy8uLAwcOsHnz5mzXJJIuQ0QyNGDAAOOv/5m0bt3aAIx58+al2f7KlStp1v397383fH19jWvXrjnWRUdHG2FhYY7Xhw4dMgAjMDDQOHfunGP9F198YQDGl19+6Vg3duzYNDUBhqenp3HgwAHHup9//tkAjFmzZjnWderUyfD19TWOHz/uWLd//37Dw8MjzTHTk97nmzJlimGz2YwjR46k+nyAMWHChFTbNmzY0GjUqJHj9bJlywzAmDp1qmPdzZs3jZYtWxqAsWDBgkzrGTlypFGsWLFUP7PExESjZMmSxtNPP51p3Vu3bjUA4z//+Y9j3YYNGwzA2LBhQ6rPkvJ3lZ2a0zvvokWLDMD49ttvHeumTZtmAMahQ4fSbB8WFmZER0c7Xg8ZMsQAjE2bNjnWXbx40ahc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3vZurmLFivHll186+k94e3vzyCOP8Pzzz1O/fv0c1ePm5sby5csZMmQIH374ITabjYceeojp06fTsGHDLB8nKiqKl19+mStXrqS6SyrZzJkzcXd356OPPuLatWtERESwbt26HP1eslPzxx9/zMCBA3n77bcxDIP777+fVatWpbpbDaBJkyZMnDiRefPmsXr1apKSkjh06FC64SYoKIgtW7YwfPhwZs2axbVr16hXrx5ffvmlo/Ukv4wbN46yZcsye/ZsXnjhBUqXLk3fvn2ZPHmyYxym+vXr065dO7788kuOHz+Or68v9evXZ9WqVY47xR566CEOHz7Me++9x9mzZylTpgytW7dm/PjxjrutRHLDZhSmf1aKiCU6d+7Mnj170u0PIiJS1KjPjYiL+etUCfv372flypW0adPGmoJERPKYWm5EXExISIhjvqMjR44wd+5cEhMT2blzJ9WqVbO6PBGRXFOfGxEX0759exYtWkRcXBxeXl40b96cyZMnK9iIiNNQy42IiIg4FfW5EREREaeicCMiIiJOxeX63CQlJXHixAn8/PyyNeS5iIiIWMcwDC5evEj58uVvO1ily4WbEydOEBoaanUZIiIikgPHjh2jYsWKmW7jcuHGz88PMH84/v7+FlcjIiIiWZGQkEBoaKjjezwzLhduki9F+fv7K9yIiIgUMVnpUqIOxSIiIuJUFG5ERETEqSjciIiIiFNxuT43IiKSt+x2Ozdu3LC6DHECnp6et73NOysUbkREJEcMwyAuLo4LFy5YXYo4CTc3NypXroynp2eujqNwIyIiOZIcbMqVK4evr68GRpVcSR5k9+TJk1SqVClXf08KNyIikm12u90RbAIDA60uR5xE2bJlOXHiBDdv3qRYsWI5Po46FIuISLYl97Hx9fW1uBJxJsmXo+x2e66Oo3AjIiI5pktRkpfy6u9Jl6XyiN0OmzbByZMQEgItW4K7u9VViYiIuB613OSBmBgID4d77oEnnjCfw8PN9SIi4vzCw8OZMWNGlrffuHEjNpst3+80W7hwISVLlszXcxRGCje5FBMDXbvCH3+kXn/8uLleAUdEJHN2O2zcCIsWmc+57G6RKZvNlulj3LhxOTru9u3b6du3b5a3b9GiBSdPniQgICBH55PM6bJULtjtMHgwGEba9wwDbDYYMgQefliXqERE0hMTY/5/NOU/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XkxERATBwcGUKFGCl19+OcvnSHmu+vXrU7x4cce6iIgIkpKSiI2NdayrU6cO7u7ujtchISGcPn06S+dISEjgxIkTREREpFofERHB3r17AfPS165du6hRowaDBg3i66+/dmzXrVs3rl69SpUqVejTpw9Lly7l5s2b2fqc2aVwIyIiecLXFy5dsuaR3b68zzzzDJ9//jkXL15kwYIFVK1aldatWwMwbdo0Zs6cyfDhw9mwYQO7du2iXbt2XL9+Pc9+Vlu3bqVHjx488MADrFixgp07dzJq1Kg8PUdKxYoVS/XaZrORlJSUZ8e/6667OHToEBMnTuTq1as89thjdO3aFTBnM4+NjWXOnDn4+PjQv39/WrVqla0+P9mlPjciIpInbDZI0YBQqD322GMMHjyYjz/+mP/85z/069fP0f9m8+bNPPzwwzz55JOA2Ydm37591K5dO0vHrlWrFseOHePkyZOEhIQA8P3336faZsuWLYSFhTFq1CjHuiNHjqTaxtPTE7vdfttzLVy4kMuXLztabzZv3oybmxs1atTIUr234+/vT/ny5dm8ebMjACafJ2UfJH9/f6KiooiKiqJr1660b9+ec+fOUbp0aXx8fOjUqROdOnViwIAB1KxZk927d3PXXXflSY1/pXAjIiIup0SJEkRFRTFy5EgSEhLo1auX471q1aqxZMkStmzZQqlSpXjjjTc4depUlsNNZGQk1atXJzo6mmnTppGQkJAqxCSf4+jRo3zyySc0adKEr776iqVLl6baJjw8nEOHDrFr1y4qVqyIn59fmlvAe/TowdixY4mOjmbcuHGcOXOGgQMH0rNnT4KCgnL2w0nHSy+9xNixY6latSoNGjRgwYIF7Nq1i48++giAN954g5CQEBo2bIibmxufffYZwcHBlCxZkoULF2K322nWrBm+vr58+OGH+Pj4EBYWlmf1/ZUuS4mIiEt65plnOH/+PO3atUvVP+bll1/mrrvuol27drRp04bg4GA6d+6c5eO6ubmxdOlSrl69StOmTXn22WeZNGlSqm0eeughXnjhBZ5//nkaNGjAli1bGD16dKptHn30Udq3b88999xD2bJl070d3dfXlzVr1nDu3DmaNGlC165dadu2LbNnz87eD+M2Bg0axNChQ3nxxRepW7cuq1evZvny5VSrVg0w7/yaOnUqjRs3pkmTJhw+fJiVK1fi5uZGyZIleffdd4mIiKBevXqsW7eOL7/8ksDAwDytMSWbYRhGvh29EEpISCAgIID4+Hj8/f2tLkdEpEi6du0ahw4donLlynh7e1tdjjiJzP6usvP9rZYbERERcSoKNyIiIuJUFG5ERETEqSjciIiIiFNRuBERkRxzsXtSJJ/l1d+Two2IiGRb8oi3V6yaKVOcUvIIzSmnisgJDeInIiLZ5u7uTsmSJR3zE/n6+jpG+BXJiaSkJM6cOYOvry8eHrmLJwo3ecQw4NNPwcsLsjHWk4hIkRUcHAyQ5QkYRW7Hzc2NSpUq5TooK9zkkQUL4JlnoGJFaNcOfHysrkhEJH/ZbDZCQkIoV65cvk6CKK7D09MTN7fc95hRuMkjjz8O48bBsWMwYwaMHGl1RSIiBcPd3T3XfSRE8pI6FOcRHx+YPNlcnjIF1EorIiJiDYWbPPTEE3DXXXDxIowfb3U1IiIirknhJg+5ucHrr5vL77wDv/9ubT0iIiKuSOEmj91zD3TqBHY7DB9udTUiIiKuR+EmH7z2Gri7w/LlsHGj1dWIiIi4FoWbfFCrFvTtay4PGwZJSdbWIyIi4kosDzdvv/024eHheHt706xZM7Zt25bp9hcuXGDAgAGEhITg5eVF9erVWblyZQFVm3XjxoGfH/z0EyxaZHU1IiIirsPScLN48WKGDh3K2LFj2bFjB/Xr16ddu3YZjnZ5/fp17rvvPg4fPsySJUuIjY3l3XffpUKFCgVc+e2VKwcjRpjL//wnXL1qbT0iIiKuwmZYOKVrs2bNaNKkCbNnzwbMeSVCQ0MZOHAgI5KTQQrz5s1j2rRp/P77745J27IrISGBgIAA4uPj8ff3z1X9t3PlCtSoAX/8Aa++qg7GIiIiOZWd72/LWm6uX7/OTz/9RGRk5K1i3NyIjIxk69at6e6zfPlymjdvzoABAwgKCuLOO+9k8uTJ2O32DM+TmJhIQkJCqkdB8fWFSZPM5cmT4cyZAju1iIiIy7Is3Jw9exa73U5QUFCq9UFBQcTFxaW7z8GDB1myZAl2u52VK1cyevRopk+fziuvvJLheaZMmUJAQIDjERoamqef43aefBIaNICEBJgwoUBPLSIi4pIs71CcHUlJSZQrV4758+fTqFEjoqKiGDVqFPPmzctwn5EjRxIfH+94HDt2rAArNgf2mz7dXJ43D2JjC/T0IiIiLseycFOmTBnc3d05depUqvWnTp0iODg43X1CQkKoXr16qgnaatWqRVxcHNevX093Hy8vL/z9/VM9Ctq990LHjnDz5q1OxiIiIpI/LAs3np6eNGrUiPXr1zvWJSUlsX79epo3b57uPhERERw4cICkFAPH7Nu3j5CQEDw9PfO95tyYOtVsxVm2DL791upqREREnJell6WGDh3Ku+++y/vvv8/evXvp168fly9fpnfv3gA89dRTjBw50rF9v379OHfuHIMHD2bfvn189dVXTJ48mQEDBlj1EbKsdm3o08dc1sB+IiIi+cfDypNHRUVx5swZxowZQ1xcHA0aNGD16tWOTsZHjx7Fze1W/goNDWXNmjW88MIL1KtXjwoVKjB48GCGF5F7rMeNg48+gu3bYfFiePxxqysSERFxPpaOc2OFghznJj2vvAKjR0NYmDlruLd3gZcgIiJS5BSJcW5c1dChUL48HDkCs2ZZXY2IiIjzUbgpYCkH9ps0Cc6etbYeERERZ6NwY4GePaF+fYiPh4kTra5GRETEuSjcWMDdHV5/3VyeMwf277e2HhEREWeicGORyEjo0EED+4mIiOQ1hRsLJQ/sFxMD331ndTUiIiLOQeHGQnfeCc88Yy6/+CK41k35IiIi+UPhxmLjx0Px4rBtG3z6qdXViIiIFH0KNxYLCYF//MNcHjECEhOtrUdERKSoU7gpBF580Qw5hw/D7NlWVyMiIlK0KdwUAsWLm9MygPn855/W1iMiIlKUKdwUEtHRUK8eXLhwK+iIiIhI9incFBIpB/Z7+204cMDaekRERIoqhZtC5L77oH17uHEDRo60uhoREZGiSeGmkJk2zRzYb8kS2LLF6mpERESKHoWbQubOO+Hpp81lDewnIiKSfQo3hdCECeDrC99/b7bgiIiISNYp3BRCGthPREQk5xRuCqlhw8yQc/AgzJljdTUiIiJFh8JNIVW8OEycaC5PnAjnzllbj4iISFGhcFOI9epldjA+fx4mTbK6GhERkaJB4aYQSzmw36xZ5iUqERERyZzCTSHXrh3cf78G9hMREckqhZsiYNo0sNng009h61arqxERESncFG6KgHr1oHdvc3nYMA3sJyIikhmFmyIieWC/LVsgJsbqakRERAovhZsiokIFs9UGYPhwuH7d2npEREQKK4WbIuSllyAoCP73P5g71+pq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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "acc = history_dict['binary_accuracy']\n",
    "val_acc = history_dict['val_binary_accuracy']\n",
    "loss = history_dict['loss']\n",
    "val_loss = history_dict['val_loss']\n",
    "\n",
    "epochs = range(1, len(acc) + 1)\n",
    "\n",
    "# \"bo\" is for \"blue dot\"\n",
    "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
    "# b is for \"solid blue line\"\n",
    "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
    "plt.title('Training and validation loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:10:08.683575Z",
     "iopub.status.busy": "2023-12-07T03:10:08.683186Z",
     "iopub.status.idle": "2023-12-07T03:10:08.821271Z",
     "shell.execute_reply": "2023-12-07T03:10:08.820676Z"
    },
    "id": "Z3PJemLPXwz_"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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CNy8eRMLFiwoztrofyZMAOLjAQ8PYOFCuashIiIq/QrcA7Rjxw4MHz4cgwcP5iUwStBffwEREdL9//4X4DxzIiKioitwD9D+/fvx6NEjBAQEIDAwEAsXLkRKSkpx1mb0Hj0C+vYFhAA++ADQcgJuIiIiKoQCB6A333wTy5YtQ3JyMj766CNs2LABnp6eyMnJQUxMDB49elScdRqlTz8Frl4FfHyAuXPlroaIiKjsKNIy+Pj4eERGRmLNmjV4+PAh3n77bWzbtk2f9cnCEJbB79jxvMdn926gVStZyiAiIio1SmQZPABUq1YNs2fPxo0bN7B+/fqiHIpecP8+MGCAdH/kSIYfIiIifSvyiRDLIrl7gHr1AtatA6pVA06eBKysSrwEIiKiUqfEeoBI/zZtksKPiQmwahXDDxERUXFgADIgt28DgwZJ98eNAwID5a2HiIiorGIAMhBCAAMHAvfuAXXrPr/iOxEREekfA5CBWL0a2LYNMDOT7puby10RERFR2cUAZACuXweGD5fuT5sG1Kkjbz1ERERlHQOQzHJygP79gbQ04M03gc8+k7siIiKiso8BSGZLlgC7dkmrvVatAkwLfHU2IiIiKiwGIBlduvS8x+err4A33pC3HiIiImPBACQTlUq60Onjx8B//gMMHSp3RURERMaDAUgmc+cCBw4AdnbAihXSiQ+JiIioZPBrVwZnzwITJ0r3IyIAb29ZyyEiIjI6DEAlLCsLCAsDnj0D3n0X6NdP7oqIiIiMDwNQCZsxQ7rAablywLJlgEIhd0VERETGhwGoBP39N/Dll9L9778HPDzkrYeIiMhYMQCVoO+/l1Z/desGdO8udzVERETGi6fdK0HLlgH16gG9esldCRERkXFjACpBSuXza34RERGRfDgERkREREaHAYiIiIiMDgMQERERGR0GICIiIjI6DEBERERkdBiAiIiIyOgwABEREZHRYQAiIiIio8MAREREREaHAYiIiIiMDgMQERERGR0GICIiIjI6DEBERERkdBiAiIiIyOgwABEREZHRYQAiIiIio8MAREREREaHAYiIiIiMDgMQERERGR0GICIiIjI6DEBERERkdBiAiIiIyOgwABEREZHRYQAiIiIio8MAREREREaHAYiIiIiMDgMQERERGR0GICIiIjI6DEBERERkdBiAiIiIyOgYRABatGgRfHx8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hAAePy7517W2BhSK129namqKsLAwrFy5EhMmTIDifztt3LgRKpUKPXv2RHp6OgICAjBmzBjY29tj+/bt6NOnD/z8/NC4cePXvkZOTg5CQ0Ph5uaGI0eOIDU1VWO+UC47OzusXLkSnp6eOHPmDD788EPY2dnh888/R/fu3XH27Fns3LlTHTwcHBzyHCMjIwPBwcFo0qQJjh07hjt37uCDDz7A0KFDNULe7t274eHhgd27d+PSpUvo3r076tWrhw8//FDre0hPT0f79u0xY8YMWFhYYPXq1QgJCUF8fDwqVaoEAAgLC8OhQ4cwf/581K1bFwkJCUhJSQEAJCUloUWLFmjVqhX+/PNP2Nvb48CBA8jOzn7t51ckQkZJSUkCgDh48KBG+2effSYaN2782v0HDhwo/P39NdoOHDggAIibN29qtHft2lV069ZN63GePn0qUlNT1bfr168LACI1NVXHd/Rq2dlC7N4txLp10s/sbL0enojIYD158kScO3dOPHnyRN2Wni6EFINK9paeXvC6z58/LwCI3bt3q9uCgoJE7969893n3XffFZ9++qn6ccuWLcWIESPUj729vcV3330nhBDi999/F6ampiIpKUn9/I4dOwQAsWXLlnxf45tvvhEBAQHqx1OmTBF169bNs92Lx1m6dKlwcnIS6S98ANu3bxcmJibi1q1bQgghwsPDhbe3t8h+4Quqa9euonv37vnWok2tWrXEggULhBBCxMfHCwAiJiZG67bjxo0Tvr6+4tmzZwU6tra/pVypqakF/v6WdQjMxcUFSqUSt2/f1mi/ffs23N3dX7lvRkYGNmzYgAEDBmi05+6nyzEtLCxgb2+vcdO36GjAxwf4z3+A99+Xfvr4SO1ERGSYqlevjqZNm6rnpV66dAn79u1Tf/eoVCpMnz4d/v7+KFeuHGxtbfH7778jMTGxQMc/f/48KlasCE9PT3VbkyZN8mwXFRWFZs2awd3dHba2tpg4cWKBX+PF16pbt656yggANGvWDDk5OYiPj1e31apVC0qlUv3Yw8Mj34VJgNQDNHr0aNSoUQOOjo6wtbXF+fPn1fXFxcVBqVSiZcuWWvePi4tDUFAQzMzMdHo/RSVrADI3N0dAQABiY2PVbTk5OYiNjdX6B/CijRs3IjMzE71799Zo9/X1hbu7u8Yx09LScOTIkdces7hERwNdugAv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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(epochs, acc, 'bo', label='Training acc')\n",
    "plt.plot(epochs, val_acc, 'b', label='Validation acc')\n",
    "plt.title('Training and validation accuracy')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend(loc='lower right')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hFFyCuJoXy7r"
   },
   "source": [
    "In this plot, the dots represent the training loss and accuracy, and the solid lines are the validation loss and accuracy.\n",
    "\n",
    "Notice the training loss *decreases* with each epoch and the training accuracy *increases* with each epoch. This is expected when using a gradient descent optimization—it should minimize the desired quantity on every iteration.\n",
    "\n",
    "This isn't the case for the validation loss and accuracy—they seem to peak before the training accuracy. This is an example of overfitting: the model performs better on the training data than it does on data it has never seen before. After this point, the model over-optimizes and learns representations *specific* to the training data that do not *generalize* to test data.\n",
    "\n",
    "For this particular case, you could prevent overfitting by simply stopping the training when the validation accuracy is no longer increasing. One way to do so is to use the `tf.keras.callbacks.EarlyStopping` callback."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-to23J3Vy5d3"
   },
   "source": [
    "## Export the model\n",
    "\n",
    "In the code above, you applied the `TextVectorization` layer to the dataset before feeding text to the model. If you want to make your model capable of processing raw strings (for example, to simplify deploying it), you can include the `TextVectorization` layer inside your model. To do so, you can create a new model using the weights you just trained."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:10:08.824806Z",
     "iopub.status.busy": "2023-12-07T03:10:08.824256Z",
     "iopub.status.idle": "2023-12-07T03:10:12.064787Z",
     "shell.execute_reply": "2023-12-07T03:10:12.064013Z"
    },
    "id": "FWXsMvryuZuq"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "  1/782 [..............................] - ETA: 4:18 - loss: 0.3165 - accuracy: 0.9062"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 14/782 [..............................] - ETA: 3s - loss: 0.2834 - accuracy: 0.8973  "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 28/782 [>.............................] - ETA: 2s - loss: 0.2908 - accuracy: 0.8839"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 42/782 [>.............................] - ETA: 2s - loss: 0.3109 - accuracy: 0.8743"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 56/782 [=>............................] - ETA: 2s - loss: 0.3205 - accuracy: 0.8722"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 70/782 [=>............................] - ETA: 2s - loss: 0.3145 - accuracy: 0.8768"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 84/782 [==>...........................] - ETA: 2s - loss: 0.3163 - accuracy: 0.8743"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      " 98/782 [==>...........................] - ETA: 2s - loss: 0.3147 - accuracy: 0.8737"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "113/782 [===>..........................] - ETA: 2s - loss: 0.3179 - accuracy: 0.8681"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "127/782 [===>..........................] - ETA: 2s - loss: 0.3237 - accuracy: 0.8644"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "141/782 [====>.........................] - ETA: 2s - loss: 0.3229 - accuracy: 0.8635"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "156/782 [====>.........................] - ETA: 2s - loss: 0.3234 - accuracy: 0.8632"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "170/782 [=====>........................] - ETA: 2s - loss: 0.3255 - accuracy: 0.8623"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "184/782 [======>.......................] - ETA: 2s - loss: 0.3245 - accuracy: 0.8633"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "198/782 [======>.......................] - ETA: 2s - loss: 0.3238 - accuracy: 0.8636"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "212/782 [=======>......................] - ETA: 2s - loss: 0.3211 - accuracy: 0.8660"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "226/782 [=======>......................] - ETA: 2s - loss: 0.3207 - accuracy: 0.8664"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "240/782 [========>.....................] - ETA: 1s - loss: 0.3197 - accuracy: 0.8672"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "254/782 [========>.....................] - ETA: 1s - loss: 0.3181 - accuracy: 0.8686"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "268/782 [=========>....................] - ETA: 1s - loss: 0.3190 - accuracy: 0.8678"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "282/782 [=========>....................] - ETA: 1s - loss: 0.3161 - accuracy: 0.8691"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "296/782 [==========>...................] - ETA: 1s - loss: 0.3147 - accuracy: 0.8701"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "310/782 [==========>...................] - ETA: 1s - loss: 0.3121 - accuracy: 0.8718"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "325/782 [===========>..................] - ETA: 1s - loss: 0.3148 - accuracy: 0.8702"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "340/782 [============>.................] - ETA: 1s - loss: 0.3145 - accuracy: 0.8702"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "354/782 [============>.................] - ETA: 1s - loss: 0.3140 - accuracy: 0.8701"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "368/782 [=============>................] - ETA: 1s - loss: 0.3131 - accuracy: 0.8704"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "382/782 [=============>................] - ETA: 1s - loss: 0.3126 - accuracy: 0.8712"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "396/782 [==============>...............] - ETA: 1s - loss: 0.3123 - accuracy: 0.8711"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "410/782 [==============>...............] - ETA: 1s - loss: 0.3120 - accuracy: 0.8715"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "424/782 [===============>..............] - ETA: 1s - loss: 0.3122 - accuracy: 0.8713"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "438/782 [===============>..............] - ETA: 1s - loss: 0.3111 - accuracy: 0.8715"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "452/782 [================>.............] - ETA: 1s - loss: 0.3106 - accuracy: 0.8724"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "466/782 [================>.............] - ETA: 1s - loss: 0.3108 - accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "480/782 [=================>............] - ETA: 1s - loss: 0.3109 - accuracy: 0.8725"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "494/782 [=================>............] - ETA: 1s - loss: 0.3103 - accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "509/782 [==================>...........] - ETA: 1s - loss: 0.3108 - accuracy: 0.8729"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "523/782 [===================>..........] - ETA: 0s - loss: 0.3104 - accuracy: 0.8731"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "537/782 [===================>..........] - ETA: 0s - loss: 0.3094 - accuracy: 0.8732"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "551/782 [====================>.........] - ETA: 0s - loss: 0.3098 - accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "565/782 [====================>.........] - ETA: 0s - loss: 0.3093 - accuracy: 0.8729"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "580/782 [=====================>........] - ETA: 0s - loss: 0.3097 - accuracy: 0.8730"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "594/782 [=====================>........] - ETA: 0s - loss: 0.3089 - accuracy: 0.8735"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "608/782 [======================>.......] - ETA: 0s - loss: 0.3093 - accuracy: 0.8732"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "622/782 [======================>.......] - ETA: 0s - loss: 0.3091 - accuracy: 0.8734"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "636/782 [=======================>......] - ETA: 0s - loss: 0.3095 - accuracy: 0.8733"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "651/782 [=======================>......] - ETA: 0s - loss: 0.3105 - accuracy: 0.8728"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "666/782 [========================>.....] - ETA: 0s - loss: 0.3105 - accuracy: 0.8727"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "680/782 [=========================>....] - ETA: 0s - loss: 0.3101 - accuracy: 0.8731"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "694/782 [=========================>....] - ETA: 0s - loss: 0.3095 - accuracy: 0.8732"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "708/782 [==========================>...] - ETA: 0s - loss: 0.3098 - accuracy: 0.8732"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "722/782 [==========================>...] - ETA: 0s - loss: 0.3093 - accuracy: 0.8734"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "736/782 [===========================>..] - ETA: 0s - loss: 0.3086 - accuracy: 0.8739"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "750/782 [===========================>..] - ETA: 0s - loss: 0.3095 - accuracy: 0.8736"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "764/782 [============================>.] - ETA: 0s - loss: 0.3096 - accuracy: 0.8736"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "779/782 [============================>.] - ETA: 0s - loss: 0.3097 - accuracy: 0.8738"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "782/782 [==============================] - 3s 4ms/step - loss: 0.3098 - accuracy: 0.8737\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8737199902534485\n"
     ]
    }
   ],
   "source": [
    "export_model = tf.keras.Sequential([\n",
    "  vectorize_layer,\n",
    "  model,\n",
    "  layers.Activation('sigmoid')\n",
    "])\n",
    "\n",
    "export_model.compile(\n",
    "    loss=losses.BinaryCrossentropy(from_logits=False), optimizer=\"adam\", metrics=['accuracy']\n",
    ")\n",
    "\n",
    "# Test it with `raw_test_ds`, which yields raw strings\n",
    "loss, accuracy = export_model.evaluate(raw_test_ds)\n",
    "print(accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TwQgoN88LoEF"
   },
   "source": [
    "### Inference on new data\n",
    "\n",
    "To get predictions for new examples, you can simply call `model.predict()`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-12-07T03:10:12.068846Z",
     "iopub.status.busy": "2023-12-07T03:10:12.068154Z",
     "iopub.status.idle": "2023-12-07T03:10:12.291671Z",
     "shell.execute_reply": "2023-12-07T03:10:12.290878Z"
    },
    "id": "QW355HH5L49K"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "1/1 [==============================] - ETA: 0s"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r\n",
      "1/1 [==============================] - 0s 175ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[0.61599904],\n",
       "       [0.43837   ],\n",
       "       [0.35510692]], dtype=float32)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "examples = [\n",
    "  \"The movie was great!\",\n",
    "  \"The movie was okay.\",\n",
    "  \"The movie was terrible...\"\n",
    "]\n",
    "\n",
    "export_model.predict(examples)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MaxlpFWpzR6c"
   },
   "source": [
    "Including the text preprocessing logic inside your model enables you to export a model for production that simplifies deployment, and reduces the potential for [train/test skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew).\n",
    "\n",
    "There is a performance difference to keep in mind when choosing where to apply your TextVectorization layer. Using it outside of your model enables you to do asynchronous CPU processing and buffering of your data when training on GPU. So, if you're training your model on the GPU, you probably want to go with this option to get the best performance while developing your model, then switch to including the TextVectorization layer inside your model when you're ready to prepare for deployment.\n",
    "\n",
    "Visit this [tutorial](https://www.tensorflow.org/tutorials/keras/save_and_load) to learn more about saving models."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eSSuci_6nCEG"
   },
   "source": [
    "## Exercise: multi-class classification on Stack Overflow questions\n",
    "\n",
    "This tutorial showed how to train a binary classifier from scratch on the IMDB dataset. As an exercise, you can modify this notebook to train a multi-class classifier to predict the tag of a programming question on [Stack Overflow](http://stackoverflow.com/).\n",
    "\n",
    "A [dataset](https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz) has been prepared for you to use containing the body of several thousand programming questions (for example, \"How can I sort a dictionary by value in Python?\") posted to Stack Overflow. Each of these is labeled with exactly one tag (either Python, CSharp, JavaScript, or Java). Your task is to take a question as input, and predict the appropriate tag, in this case, Python. \n",
    "\n",
    "The dataset you will work with contains several thousand questions extracted from the much larger public Stack Overflow dataset on [BigQuery](https://console.cloud.google.com/marketplace/details/stack-exchange/stack-overflow), which contains more than 17 million posts.\n",
    "\n",
    "After downloading the dataset, you will find it has a similar directory structure to the IMDB dataset you worked with previously:\n",
    "\n",
    "```\n",
    "train/\n",
    "...python/\n",
    "......0.txt\n",
    "......1.txt\n",
    "...javascript/\n",
    "......0.txt\n",
    "......1.txt\n",
    "...csharp/\n",
    "......0.txt\n",
    "......1.txt\n",
    "...java/\n",
    "......0.txt\n",
    "......1.txt\n",
    "```\n",
    "\n",
    "Note: To increase the difficulty of the classification problem, occurrences of the words Python, CSharp, JavaScript, or Java in the programming questions have been replaced with the word *blank* (as many questions contain the language they're about).\n",
    "\n",
    "To complete this exercise, you should modify this notebook to work with the Stack Overflow dataset by making the following modifications:\n",
    "\n",
    "1. At the top of your notebook, update the code that downloads the IMDB dataset with code to download the [Stack Overflow dataset](https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz) that has already been prepared. As the Stack Overflow dataset has a similar directory structure, you will not need to make many modifications.\n",
    "\n",
    "1. Modify the last layer of your model to `Dense(4)`, as there are now four output classes.\n",
    "\n",
    "1. When compiling the model, change the loss to `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`. This is the correct loss function to use for a multi-class classification problem, when the labels for each class are integers (in this case, they can be 0, *1*, *2*, or *3*). In addition, change the metrics to `metrics=['accuracy']`, since this is a multi-class classification problem (`tf.metrics.BinaryAccuracy` is only used for binary classifiers).\n",
    "\n",
    "1. When plotting accuracy over time, change `binary_accuracy` and `val_binary_accuracy` to `accuracy` and `val_accuracy`, respectively.\n",
    "\n",
    "1. Once these changes are complete, you will be able to train a multi-class classifier. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "F0T5SIwSm7uc"
   },
   "source": [
    "## Learning more\n",
    "\n",
    "This tutorial introduced text classification from scratch. To learn more about the text classification workflow in general, check out the [Text classification guide](https://developers.google.com/machine-learning/guides/text-classification/) from Google Developers.\n"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "name": "text_classification.ipynb",
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}