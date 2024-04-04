# Hermes 👟🪽

Hermes is a project used to experiment with vector-based search in MongoDB databases on local machines. The project is named after the Greek god of trade, heralds, and travelers, in 
reference to the search and retrieval of information and as nod to MongoDB Atlas, the cloud-based vector search service. 

The project is inspired by the goals of Mozilla's [llamafile](https://github.com/Mozilla-Ocho/llamafile), making modern "AI" much more accessible to both developers and users. 
In addition to that, it supports the idea of moving away from cloud-based-everything towards more local and self-hosted solutions, using the power of the edge. 

This project is a work in progress and is not intended for production use. 

You can use the `--fill` option to populate the database with dummy data. The dummy data is a list of movie titles and descriptions coming 
from a modified version of the [Embedded Movies dataset](https://huggingface.co/datasets/christophsonntag/gte_embedded_movies) using 
the open source [General Text Embeddings model](https://huggingface.co/thenlper/gte-large) instead of OpenAI's text-embedding-ada-002 
embedding model used in MongoDB Atlas. 

Similar to Atlas, you can choose between cosine, euclidean, and dotProduct distance measures. However, currently only cosine distance is 
implemented. 

## Installation
Create a virtual environment and install the requirements by running ```pip install -r requirements.txt```. 
If you run Hermes for the first time, you can use the `--fill` option to populate the database with dummy data. 
It may take a while to fill the database with dummy data and to download the embedding model. 

## Usage
```
$python3 hermes.py "What is the best romantic movie to watch and why?"
Searching for 'What is the best romantic movie to watch and why?' using cosine distance...
{'_id': ObjectId('660f0d6d167d526cd3302396'), 'title': 'Shut Up and Kiss Me!', 'cos_similarity': 0.8115843141879644}
{'_id': ObjectId('660f0d6d167d526cd330229b'), 'title': 'Pearl Harbor', 'cos_similarity': 0.8050779350683754}
{'_id': ObjectId('660f0d6d167d526cd3302190'), 'title': 'Titanic', 'cos_similarity': 0.8015175492140932}
{'_id': ObjectId('660f0d6c167d526cd330205f'), 'title': 'China Girl', 'cos_similarity': 0.797677718678184}
```

## Retrieval Augmented Generation
The project shares 

## Disclaimer
WIP - Do not use. 
