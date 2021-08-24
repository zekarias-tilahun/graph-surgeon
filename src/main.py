import execute
import utils

import torch

import numpy as np
import time

torch.manual_seed(0)


def main():
    args = utils.parse_args(use_best=True)
    assert args.task in {'bc', 'mcc', 'mlc'}

    utils.log(args, verbose=args.verbose)
    
    # Self-supervised training
    self_exec = execute.SelfExec(args=args)
    self_exec.execute()
    
    self_exec.pause_training_mode()
    torch.cuda.empty_cache()
    num_classes = int(self_exec.dataset.num_classes)
    
    # Inference and Evaluation
    
    embeddings = self_exec.infer_embedding(**self_exec.get_inference_args())
    
    lev_exec = execute.LinearEvalExec(
        in_dim=embeddings.shape[1], out_dim=num_classes, 
        device=args.device, task=args.task)

    lev_exec.execute(
        x=embeddings, y=self_exec.dataset.data.y,
        train_mask=self_exec.dataset.data.train_mask,
        val_mask=self_exec.dataset.data.val_mask, 
        test_mask=self_exec.dataset.data.test_mask
    )
    

if __name__ == "__main__":
    main()
