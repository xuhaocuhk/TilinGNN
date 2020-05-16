import torch
import traceback

def get_network_prediction(network,
                            x,
                            adj_e_index,
                            adj_e_features,
                            col_e_idx,
                            col_e_features = None):
    # torch.cuda.empty_cache()

    ## return None result if the network could not handle data
    try:
        probs, *_ = network(x=x,
                            adj_e_index=adj_e_index,
                            adj_e_features=adj_e_features,
                            col_e_idx=col_e_idx,
                            col_e_features=col_e_features)
    except:
        print(traceback.format_exc())
        return None

    return probs