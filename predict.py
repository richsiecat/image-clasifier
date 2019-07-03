#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from time import time
from get_input_parse import get_predict_args
from model_utils import load_checkpoint

def main():
    start_time = time()
    in_arg = get_predict_args()

    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    # Load model trained
    model = load_checkpoint(in_arg.checkpoint)
    # Process Image
    image_tensor = process_image(in_arg.image)

    device = get_device(in_arg.gpu);
    top_k = check_value(in_arg.top_k, 5)

    model.eval();
    torch_image = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)
    model = model.cpu()

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale & Find the top results
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(top_k)
 
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
 
    # Print out probabilities
    print_probability(top_flowers, top_probs)
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
