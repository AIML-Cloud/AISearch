def list2arr(l):
    "Convert list into pytorch Variable."
    # return torch.tensor(np.expand_dims(np.array(1), axis=-1))
    return torch.LongTensor(np.expand_dims(np.array(l), -1)).cpu()
    #return torch.tensor(l).to(torch.int64)
    # return torch.FloatTensor(l)


def make_prediction_from_list(model, l):
    """
    Encode a list of integers that represent a sequence of tokens.  The
    purpose is to encode a sentence or phrase.
    Parameters
    -----------
    model : fastai language model
    l : list
        list of integers, representing a sequence of tokens that you want to encode
    """
    arr = list2arr(l)# turn list into pytorch Variable with bs=1
    model.reset()  # language model is stateful, so you must reset upon each prediction
    hidden_states = model(arr)[-1][-1] # RNN Hidden Layer output is last output, and only need the last layer
    #return avg-pooling, max-pooling, and last hidden state
    return hidden_states.mean(0), hidden_states.max(0)[0], hidden_states[-1]


def get_embeddings(lm_model, list_list_int):
    """
    Vectorize a list of sequences List[List[int]] using a fast.ai language model.
    Paramters
    ---------
    lm_model : fastai language model
    list_list_int : List[List[int]]
        A list of sequences to encode
    Returns
    -------
    tuple: (avg, mean, last)
        A tuple that returns the average-pooling, max-pooling over time steps as well as the last time step.
    """
    n_rows = len(list_list_int)
    n_dim = 400
    avgarr = np.empty((n_rows, n_dim))
    maxarr = np.empty((n_rows, n_dim))
    lastarr = np.empty((n_rows, n_dim))
    for i in tqdm(range(len(list_list_int))):
        avg_, max_, last_ = make_prediction_from_list(lm_model, list_list_int[i])        
        avgarr[i,:] = avg_.data.numpy()
        maxarr[i,:] = max_.data.numpy()
        lastarr[i,:] = last_.data.numpy()
        
    return avgarr, maxarr, lastarr