import torch
import traceback

def get_network_prediction(network,
                            x,
                            adj_e_index,
                            adj_e_features,
                            col_e_idx,
                            col_e_features = None):
    try:
        probs, *_ = network(x=x,
                            adj_e_index=adj_e_index,
                            adj_e_features=adj_e_features,
                            col_e_idx=col_e_idx,
                            col_e_features=col_e_features)

    except: # be care of exceptions, such as GPU out of memory
        print(traceback.format_exc())
        raise

    return probs