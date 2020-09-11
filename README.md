# imf2vec
Python implementation of [Exploiting Music Play Sequence for Music Recommendation](https://www.ijcai.org/Proceedings/2017/0511.pdf)

Matrix factorisation for Implicit Feedback regularised by *Word2Vec* stle item embeddings learnt from sequences of user-item interactions. 

Loss function
$$
    \mathcal{L}(X,Y) = \frac{1}{2}\sum_{u,i} c_{ui}(p_{ui}-x_u^Ty_i)^2 +  \frac{\alpha}{2}\sum_{i,j \in n(k,i)} (s_{ij}-y_i^Ty_j)^2 + \frac{\lambda}{2}\big( \sum_u ||x_u||^2 +  \sum_i ||y_i||^2\big) \label{eq:imf2vec_loss}

$$
