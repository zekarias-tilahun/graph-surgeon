import execute
import utils

import logging
import optuna

import torch

import sys

import yaml

torch.manual_seed(0)


def tune_surgeon(dataset_args, name, tune=True):
    dataset_args.device = torch.device("cuda:0")
    # Floating point parameter
    print(dataset_args)

    def objective(trial=None):
        torch.cuda.empty_cache()
        if trial is not None:
            dataset_args.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            dataset_args.gamma = trial.suggest_float("gamma", 1e-4, 1.0)
            dataset_args.dropout = trial.suggest_float("dropout", 0.2, 0.85)
            dataset_args.aug_dim = trial.suggest_int("aug_dim", 64, 512, log=True)
            dataset_args.layers = trial.suggest_categorical("layers", [2, 3, 4])

        self_exec = execute.SelfExec(args=args)
        
        self_exec.execute()
        self_exec.pause_training_mode()
        
        num_classes = self_exec.dataset.num_classes
        data = self_exec.dataset.data
        embeddings = self_exec.infer_embedding(**self_exec.get_inference_args())

        emb_dim = embeddings.shape[1]

        # Evaluating under the linear setting
        lev_exec = execute.LinearEvalExec(
            in_dim=emb_dim, out_dim=num_classes, device=args.device,
            task=args.task, verbose=args.verbose)
        val_acc = lev_exec.execute(
            x=embeddings, y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask
        )

        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned()
        return val_acc
            
    if tune:
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=100)

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        with open(f"../params/{name}.yml", 'w') as f:
            yaml.safe_dump(trial.params, f)
    else:
        print("Eval")
        with open(f"../params/{name}.yml") as f:
            best_args = yaml.safe_load(f)
            print(best_args)
            dataset_args.lr = best_args['lr']
            dataset_args.gamma = best_args['gamma']
            dataset_args.dropout = best_args['dropout']
            dataset_args.aug_dim = best_args['aug_dim']
            dataset_args.layers = best_args['layers']
            print(dataset_args)
            objective()
        
        
if __name__ == "__main__":
    args = utils.parse_args(use_best=False)
    tune_surgeon(args, args.name)
