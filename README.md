# Question Answering System
*Authors: Dawid Sitnik, Natalia Jakubiak, Piotr Czajkowski*

## Introduction
The problem that we are going to face during our project is question answering challenge where we are given some piece of the context in which we need to find an answer to the particular question. The answer is always quoted from the context. 

The issue seems to be quite trivial while solving by humans, but upon further observation, it demands a lot of complex tasks for a machine, which should finally understand the contextual meaning of each word from context and question as well. Then using an abstract understanding of the question it should extract the correct section of the context.

For this task, we are going to test different techniques of question answering, point their advantages and disadvantages, and decide which one performs the best. The methods which we are going to inspect are:
* **BiDAF model**
* **BERT model**
* **Classical ML approaches: Multinomial Logistic Regression, Random Forest, Gradient Boosting**

## Dataset
We are going to use the **Stanford Question Answering Dataset (SQuAD 2.0)** for training and evaluating our models. SQuAD is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 new, unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. Samples in this dataset include (question, answer, context paragraph) tuples.

#### Sample JSON file information used in SQuAD data set
```
{'data': [{'title': 'Super_Bowl_50',
   'paragraphs': [{'context': 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.',

     	'qas': [{'answers': [{'answer_start': 177, 'text': 'Denver Broncos'},
        	{'answer_start': 177, 'text': 'Denver Broncos'},
        	{'answer_start': 177, 'text': 'Denver Broncos'}],
       		'question': 'Which NFL team represented the AFC at Super Bowl 50?',
       		'id': '56be4db0acb8001400a502ec'}]}]}],
 		'version': '1.1'}
```
To utilize our models for the Question and answering, there is a need of preparing the data similar to the SQuAD data structure mentioned.
Because SQuAD is an ongoing effort, It's not exposed to the public as the open source dataset, sample data can be downloaded from [SQUAD site](https://rajpurkar.github.io/SQuAD-explorer/).

## BiDAF Model
### Introduction
To help us understand the BiDAF model lets us first explain the general structure of a neural network which enables the machine to understand the context as well as questions. 

The first layer of the net is called **Embedding Layer** and it is responsible for converting sentences into words and words into its word embedded representation, using pre-trained vector-like *GloVe*. This type of representation is much better than one hot vector representing each word. How our problem we are going to use 100 dimensional *GloVe* word embeddings.

In the second layer, we are going to use **Encoder Layer**, which used for giving each word knowledge about its predecessors and successors. To implement that layer we will use the LSTM network. The output of this part will be the concatenation of a series of hidden vectors in the forward and backward direction. The same layer is used to create hidden vectors for questions.
<p align="center">
  <img src = "https://imgur.com/eAhLaGD.png"/>
</p>

The third used layer is the so-called **Attention Layer** which is used for getting the final answer from two previous layers. This part aims to point to the fraction of the context that responds to the given question. Lets us start with the explanation of the simplest possible attention layer which is *Dot Product Attention* 
<p align="center">
  <img src = "https://imgur.com/jlY04rn.png"/>
</p>
The dot product attention is a multiplication of each context vector *Ci* by each question vector *Qj* which is *Ei*. Then we use softmax over *Ei* getting *alpha i*. This transformation ensures that the sum of all E is equal to 1. Finally, we calculate *Ai* as the dot product of the attention distribution *alpha i* and the corresponding question vector. It can be described by the above equation:
<p align="center">
  <img src = "https://imgur.com/iFEZkk1.png"/>
</p>

The performance of the model can be enhanced by using **BiDAF Attention Layer** instead of the simple one described before. The main idea behind this layer is that the attention flows both directions - from the context to the question and vice versa. 

Firstly, we compute the similiraty matrix NxM which contains similarity score *Sij* for each pair *(ci, qi)*. Sij = wT sim[ci ; qj ; ci ◦ qj ] ∈ R Here, ci ◦ qj is an elementwise product and wsim ∈ R 6h is a weight vector. Described in equation below: 
<p align="center">
  <img src = "https://imgur.com/nHnVUW4.png"/>
</p>
The next action that is performed is Context to Question Attention (similar to the dot product described above). In this case we take the row-wise softmax of S to obtain attention distributions α i , which we use to take weighted sums of the question hidden states q j , yielding C2Q attention outputs a i .
<p align="center">
  <img src = "https://imgur.com/H5pPylu.png"/>
</p>
Next, we perform Question-to-Context Attention. For each context location i ∈ {1, . . . , N}, we take the max of the corresponding row of the similarity matrix, m i = max j Sij ∈ R. Then we take the softmax over the resulting vector m ∈ R N — this gives us an attention distribution β ∈ R N over context locations. We then use β to take a weighted sum of the context hidden states c i — this is the Q2C attention output c prime:
<p align="center">
  <img src = "https://imgur.com/b0SjDeX.png"/>
</p>
At the end the context position c i is combined with output from C2Q and Q2C attentions as described below:
<p align="center">
  <img src = "https://imgur.com/n9ygwhP.png"/>
</p>

The last layer used in our neural network is **Output Layer** which is a softmax layer that helps to decide what is the start and the end index for the answer span. In that part, the context hidden states are combined with the attention vector from the previous layer to create blended reps. These reps are the input to a fully connected layer which is using softmax and a p_end vector with probability for end index. Because we know that in most cases start and end indexes are spaced from each other for maximally 15 words, we look for start and end indexes that maximize *p_start * p_end*.

In that case, the loss function is the sum of the cross-entropy loss for the start and end locations. It is minimized using Adam Optimizer.

### Running the Scripts
1. Create directories.
- *\dwr*
- *\squad*
- *\data\squadDownload* 

2.Download the dataset and glove vectors.
- Glove Word Vectors: http://nlp.stanford.edu/data/glove.6B.zip (place it to *\dwr* folder)
- SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/ (place it to *\squad* folder)

3.Run the scripts in following order
- data_preprocessing.py
- model.py

### Data Preprocessing
The used SQuAD dataset consists of 2 files:
- train-v2.0.json
- dev-v2.0.json

The data was in the form of triplets - context, question, and its answer span, which is the answer with its start and end indices. Those files were used to generate four new files containing a tokenized version of the question, context, and answer with its span. The important thing about those files is, that their lines are aligned in triplets. Each line in answer span consists of starting and ending indices of the corresponding context in which the answer can be found. 

To obtain vector representation of the text the GloVe Stanford embedding was used. GloVe performs training on aggregated global word-word co-occurrence statistics from a corpus and the resulting representation showcase interesting linear substructures of the word vector space. A word embeddings with dimensionality d = [50, 100, 200, 300], 6B tokens, and vocabulary of size 400k, pre-trained on Wikipedia, and Gigaword were used. Words that couldn't be found in the GloVe dictionary has been treated as 0 vectors. For the tokenization of the words, the basic tokenizer was used. In the end, the context with the question was converted to token ids indexed against the entire vocabulary. 

### Model Configuration
The model was built and trained using TensorFlow, because of its simplicity and abstraction which enabled creating the network by making only small changes to the existing LSTM layer. It also provides sequence to sequence models. In this case, the BahdamuAttention was used. For the intermediate calculations at each time step, a basic attention wrapper was used. Because of its ability to use the moving average of the parameters, the Adam algorithm was used for controlling the learning rate. For controlling the learning process, the gradient was computed and the loss function minimized. 

### Evaluate Metrics 
For model evaluation, we used described in the initial SQuAD paper ExactMatch metric. It measures the percentage of predictions that match one of the ground truth answers exactly.

### Result 
the final model was trained with 30 epochs of batch-size 32. Training each epoch took about 10 hours which gives almost two weeks of training. The exact match of the model equaled 0.60 which is still far behind the best solutions, but it can be still treated as a satisfying result.

## BERT Model
### Introduction
BERT( Bidirectional Encoder Representations from Transformers) method of pre-training language representations. With the use of pre-trained BERT models, we can utilize pre-trained memory information of sentence structure, language, and text grammar-related memory of large corpus of millions, or billions, of annotated training examples, that it has trained.

### How BERT works
BERT makes use of **Transformer**, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. 
As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered **bidirectional**, though it would be more accurate to say that it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word). While the concept of bidirectional was around for a long time, BERT was first on its kind to successfully pre-train bidirectional in a deep neural network.

#### BERT Input Format
The input representation used by BERT is able to represent a single text sentence as well as a pair of sentences (eg., [Question, Answer]) in a single sequence of tokens.
* The first token of every input sequence is the special classification token – **[CLS]**. This token is used in classification tasks as an aggregate of the entire sequence representation. It is ignored in non-classification tasks.
* For sentence pair tasks, the WordPiece tokens of the two sentences are separated by another **[SEP]** token. This input sequence also ends with the **[SEP]** token.
 Sentence Pair Input
* A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar to token/word embeddings with a vocabulary of 2.
* A positional embedding is also added to each token to indicate its position in the sequence.

#### Tokenization with BERT
It has three main steps:
1. Text normalization: Convert all whitespace characters to spaces, and (for the Uncased model) lowercase the input and strip out accent markers. E.g., *John Johanson's, → john johanson's,*.
1. Punctuation splitting: Split all punctuation characters on both sides (i.e., add whitespace around all punctuation characters). Punctuation characters are defined as (a) Anything with a P* Unicode class, (b) any non-letter/number/space ASCII character (e.g., characters like $ which are technically not punctuation). E.g., *john johanson's, → john johanson ' s ,*
1. WordPiece tokenization: Apply whitespace tokenization to the output of the above procedure, and apply WordPiece tokenization to each token separately. (Our implementation is directly based on the one from tensor2tensor, which is linked). E.g., *john johanson ' s , → john johan ##son ' s ,*.

### Fine-Tuning
Because pre-training is fairly expensive (hundreds of GPU hours needed to train the original BERT model from scratch), in our project we are going to use the **pre-trained BERT model**, add an untrained layer of neurons on the end, and train the new model for our question answering task. The authors recommend only 2-4 epochs of training for fine-tuning BERT on a specific NLP task. We are going to train model with 2 epochs of batch-size 24.
BERT has release BERT-Base and BERT-Large models. We are going to use *BERT-Large-Uncased Model*: huge model, with 24 Transformer blocks, 1024 hidden units in each layer, and 340M parameters. This model is pre-trained on 40 epochs over a 3.3 billion word corpus, including BooksCorpus (800 million words) and English Wikipedia (2.5 billion words). (Uncased means that the text has been lowercased before WordPiece tokenization, e.g., John Smith becomes john smith.)

### Testing and Results
The whole fine-tuning and predicting process of BERT model with Cloud TPU is contained in the script: *finetuning_and_predicition.ipynb*.
It took 30 minutes on a single Cloud TPU to fine-tune BERT for Question Answering task.
Prediction results:
```html
{
  "exact": 76.21494146382548,
  "f1": 79.26313095827322,
  "total": 11873,
  "HasAns_exact": 76.75438596491227,
  "HasAns_f1": 82.85950638791795,
  "HasAns_total": 5928,
  "NoAns_exact": 75.67703952901599,
  "NoAns_f1": 75.67703952901599,
  "NoAns_total": 5945,
  "best_exact": 77.5878042617704,
  "best_exact_thresh": -7.583559513092041,
  "best_f1": 80.32618854039865,
  "best_f1_thresh": -5.022722780704498
}
```

## Classical ML approaches

### Introduction
The above-mentioned methods are designed specially for NLP and Machine Comprehension problems solving. However, since traditional Machine Learning methods are well known and quite easy to understand, it is worth to give them a try. Unfortunately, they are too simple for a problem as complicated as question answering and to find the exact answer, so for this part of the project we will only look for a sentence in a context containing the answer, not the exact answer itself.

### Sentence embeddings
In this scenario, the data is preprocessed using Facebook Sentence Embedding - **InferSent**. It is a more advanced embedding than traditional approaches as it not only derives semantic relationships between words from the co-occurrence matrices, but it analyses the sentence as a whole. For instance, the order of the words in a sentence, which can sometimes change its meaning, is important for InferSent, while for GloVe it is not. The tool has been trained on natural language inference data. For each sentence, a 4096-dimensional vector of numbers is returned.

Infersent requires a set of word embeddings. In this project GloVe vector has been used for that. In order to provide a set of sentences to be analyzed, the context needs to be split into valid sentences (tokenized). The **TextBlob** library has been used to achieve that.

In the program, all the sentences from all the contexts and all the questions (over 150000 sentences in total) have been transformed into 4096-dimensional vectors. Due to a large number of calculations required, his step took a lot of time - about 12 hours. In the script this step was also divided into eight sub-steps due to memory problems that have appeared.

### Features creation
To simplify the process, contexts with more than 12 sentences and questions referring to them are deleted from the data set. This doesn't make a big difference, as there are only a few very long contexts in SQuAD.

Using vectors created with InferSent, for each question-sentence pair two distance measures are counted:
* **Euclidean distance** - the 'ordinary' straight-line distance between two points in Euclidean space. Maximum value for this data set: 10.
* **Cosine distance** - it equals the cosine of the angle between two points in the space. Maximum possible value: 1.

For shorter contexts, values of missing questions are replaced with maximum values available, so that the sentence would not be considered as a correct one due to a big distance between it and a question.

Each sentence in a context has now a set of values describing their 'distance' from the question, so for each context-question pair there are 24 features provided.

### Train and test sets
Instead of using the SQuAD *dev-v2.0.json* set for testing the models, the *train-v2.0.json* is randomly split into test and train data. This approach allows us to run the scripts with different sizes of the sets and have a different split every time. The library used for that is *scikit-learn*.

### Algorithms
Three popular ML algorithms are used to train and test the data. All of them are based on supervised learning.
* **Multinomial logistic regression** - a simple classification algorithm used to predict the probabilities of each class based on the input and returning the result class.
* **Random forest** - creating a number of different decision trees during training phase and returning the class (sentence) based on mode of the results of all trees. Each tree has multiple nodes, where the records are divided in two or more branches based on one of the features. At the bottom of each the leaves determining a record class are created.
* **Gradient boosting** - a classification technique producing a strong model based on a number of weak models (such as decision trees). It optimizes their results based on counting a loss function. For the implementation, the XGboost library, providing some extra feautres, is used.

First two algorithms are implemented in the above-mentioned *scikit-learn* library. All the classifiers have a variety of parameters that can be adjusted depending on our needs, but for this project most of them are left default, as the results they return are satisfying.

### Results
The accuracy of classifiers for testing data is presented below:
* Multinominal logistic regression: 64,8 %
* Random forest: 69%
* Gradient boosting: 71%

The results are surprisingly high, regarding that none of the algorithms is designed for question answering problems solving. Probably the biggest credit for it goes to Facebook for creating a great tool for sentence embedding. The biggest problem is that the program returns only the sentence containing the answer, not the answer itself, so probably it could not be used in business systems, but the accuracy is high enough to solve simple problems.

## Conclusion
To conclude, we have created three tools solving a Question Answering problem. Their performances are at 60-76%, which can be considered as a good result. BiDAF model returned the worst results - 60% accuracy with a very long computing time and required a large number of epochs for training. Classical ML approaches models returned slightly higher results, but they weren't able to extract the exact answer from the question. On the other hand, they did not require a lot of time to train - most of the computing was done at the phase of data preprocessing. BERT model turned out to be the best out of all ones that we proposed. Its accuracy was at 76% and what is more, it only required two epochs to train, so the computing time was relatively low. For the future work, the advantages of each of them could be combined to make even a better performance (e.g. using Facebook Sentence Embedding instead of GloVe in BiDAF model).
