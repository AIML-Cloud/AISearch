# AISearch
## AI Based Contextual Search Engine to find relevant document shared in a common vector space
## ![Search Engine](https://github.com/Neerajcerebrum/AISearch/blob/develop/images/Search.png)

we can search content for its meaning in addition to keywords, and increase the chances the user will find the information. The involvement of semantic search are profound — for example, such a procedure would allow developers to search for any type of information code, syntax and instumental data in repositories even if they are not familiar with the syntax or fail to anticipate the right keywords or sequence. More importantly, you can generalize this approach to objects such as pictures, audio and other things.

## ![Dimensionality Engine](https://github.com/Neerajcerebrum/AISearch/blob/develop/images/Dimensionality.png)

The goal is to map trained dataset into the vector space of natural language queries, such that (text, query) pairs that describe the same concept are close neighbors, whereas unrelated (text, query) pairs are further apart, measured by cosine similarity.
<!-- 
##NLP text similarity, how it works and maths behind it
![Search Engine](https://github.com/Neerajcerebrum/AISearch/blob/develop/images/Brain.png) -->

## Accessing the last hidden state of the model to extract the embeddings.
![Search Engine](https://github.com/Neerajcerebrum/AISearch/blob/develop/images/model.png)

3D shared vector space will be loaded with our extracted embeddings and every new query
will pe plotted in the shared space. In search engine the knn model is used to return top 2 nearest neighbors to the user.

## Shared 3D space for the training dataset embeddings and User query
 ![Search Engine](https://github.com/Neerajcerebrum/AISearch/blob/develop/images/embed.png)

## Search Engine return top 2 responses to the user. 
![Search Engine](https://github.com/Neerajcerebrum/AISearch/blob/develop/images/se.png)