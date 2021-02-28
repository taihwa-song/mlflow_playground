from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient

_random_state=37

@click.command(help="Perform grid search over train (main entry point).")
@click.option("--max-runs", type=click.INT, default=32, help="Maximum number of runs to evaluate.")
@click.option("--max-p", type=click.INT, default=1, help="Maximum number of parallel runs.")
def run(max_runs, max_p):
    tracking_client = mlflow.tracking.MlflowClient()
    np.random.seed(_random_state)

    def new_eval(experiment_id):
        def eval(parms):
            md, msl = parms
            with mlflow.start_run(nested=True) as child_run:
                p = mlflow.projects.run(
                    run_id=child_run.info.run_id,
                    uri=".",
                    entry_point="train",
                    parameters={
                        "max_depth": md,
                        "min_samples_leaf": msl,
                    },
                    experiment_id=experiment_id,
                    synchronous=False,
                )
                succeeded = p.wait()
                if succeeded:
                    training_run = tracking_client.get_run(p.run_id)
                    metrics = training_run.data.metrics
                    # cap the loss at the loss of the null model
                    train_loss = metrics["train_acc"]
                    test_loss = metrics["test_acc"]
                else:
                    # run failed => return null loss
                    tracking_client.set_terminated(p.run_id, "FAILED")
                    train_loss = -np.finfo(np.float64).max
                    test_loss = -np.finfo(np.float64).max
                mlflow.log_params({
                    "param_max_depth": md,
                    "param_min_samples_leaf": msl,
                })
                mlflow.log_metrics( {
                        "train_acc": train_loss,
                        "test_acc": test_loss,
                })
            return p.run_id

        return eval
    
    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        runs = [(np.random.randint(1, 10), np.random.randint(1, 10)) for _ in range(max_runs)]
        
        with ThreadPoolExecutor(max_workers=max_p) as executor:
            _ = executor.map(
                new_eval(experiment_id),
                runs,
            )

        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
        )

        best_val_train = -np.finfo(np.float64).max
        best_val_test = -np.finfo(np.float64).max
        best_run = None
        for r in runs:
            if r.data.metrics["test_acc"] > best_val_test:
                best_run = r
                best_val_train = r.data.metrics["train_acc"]
                best_val_test = r.data.metrics["test_acc"]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                "train_acc": best_val_train,
                "test_acc": best_val_test,
            }
        )



if __name__ == "__main__":
    run()