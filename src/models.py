import sklearn.semi_supervised as sklss


def get_model(str_model, gamma=0.5):
    if str_model == 'label_propagation':
        model = sklss.LabelPropagation(gamma=gamma, max_iter=10000)
    elif str_model == 'label_spreading':
        model = sklss.LabelSpreading(gamma=gamma, max_iter=100)
    else:
        raise ValueError

    return model
