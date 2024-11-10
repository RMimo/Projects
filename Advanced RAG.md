# Advanced RAG Techniques


## Naive Retrieval-Augmented Generation (RAG) Overview

Naive RAG involves three primary phases:

1. **Indexing**: 
   - Prepares the document collection for retrieval by cleaning and extracting relevant information from each document. This phase involves parsind and preprocessing documents, chunking the parsed documents, using an embedding model to generate vectors out of the chunks, and storing them into a vector database.
   
2. **Retrieval**: 
   - Converts the user query into a vector representation using the embedding model.
   - Compares the vectorized query with the vectors stored in the vector database to retrieve the most relevant chunks. 

3. **Generation**: 
   - Augments the user query with the retrieved chunks into a single prompt.
   - Generates an answer to the user query using a language model based on this combined information.

However, Naive RAG has some pitfalls:

1. Limited contextual understanding: 




