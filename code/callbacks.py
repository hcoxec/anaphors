import sys
import pathlib
import os
import torch
import re
import wandb
import json

from datetime import datetime
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union
from os import path, walk, makedirs, remove


from analyser import Analyser
from utils import registry, Interaction, Callback

from rich.columns import Columns
from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from collections import OrderedDict

class ConsoleLogger(Callback):
    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)

class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    optimizer_scheduler_state_dict: Optional[Dict[str, Any]]


class CheckpointSaver(Callback):
    def __init__(
        self,
        checkpoint_path: Union[str, pathlib.Path],
        checkpoint_freq: int = 1,
        prefix: str = "",
        max_checkpoints: int = sys.maxsize,
    ):
        """Saves a checkpoint file for training.
        :param checkpoint_path:  path to checkpoint directory, will be created if not present
        :param checkpoint_freq:  Number of epochs for checkpoint saving
        :param prefix: Name of checkpoint file, will be {prefix}{current_epoch}.tar
        :param max_checkpoints: Max number of concurrent checkpoint files in the directory.
        """
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_freq = checkpoint_freq
        self.prefix = prefix
        self.max_checkpoints = max_checkpoints
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs, epoch: int):
        self.epoch_counter = epoch
        if self.checkpoint_freq > 0 and (epoch % self.checkpoint_freq == 0):
            filename = f"{self.prefix}_{epoch}" if self.prefix else str(epoch)
            self.save_checkpoint(filename=filename)

    def on_train_end(self):
        self.save_checkpoint(
            filename=f"{self.prefix}_final" if self.prefix else "final"
        )

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        if len(self.get_checkpoint_files()) > self.max_checkpoints:
            self.remove_oldest_checkpoint()
        path = self.checkpoint_path / f"{filename}.tar"
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        optimizer_schedule_state_dict = None
        if self.trainer.optimizer_scheduler:
            optimizer_schedule_state_dict = (
                self.trainer.optimizer_scheduler.state_dict()
            )

        game = self.trainer.game
        return Checkpoint(
            epoch=self.epoch_counter,
            model_state_dict=game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
            optimizer_scheduler_state_dict=optimizer_schedule_state_dict,
        )

    def get_checkpoint_files(self):
        """
        Return a list of the files in the checkpoint dir
        """
        return [name for name in os.listdir(self.checkpoint_path) if ".tar" in name]

    @staticmethod
    def natural_sort(to_sort):
        """
        Sort a list of files naturally
        E.g. [file1,file4,file32,file2] -> [file1,file2,file4,file32]
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(to_sort, key=alphanum_key)

    def remove_oldest_checkpoint(self):
        """
        Remove the oldest checkpoint from the dir
        """
        checkpoints = self.natural_sort(self.get_checkpoint_files())
        os.remove(os.path.join(self.checkpoint_path, checkpoints[0]))

@registry.register("callback", "wandb")
class WandbLogger(Callback):
    def __init__(
        self,
        config,
        **kwargs,
    ):

        self.config = config
        
        #to allow different logs for the same seed a random id is appended
        rand_s = datetime.now().time().microsecond 
        if config.mode == 'train':
            wandb.init(
                project=config.wandb_name, 
                group=config.run_id, 
                id=f'seed_{config.seed}_{rand_s}',  
                job_type=config.mode,
                reinit=True,
                **kwargs)
            wandb.config.update(config.__dict__)

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def on_train_begin(self, trainer_instance):  # noqa: F821
        self.trainer = trainer_instance
        #wandb.watch(self.trainer.game, log="all")

    def on_batch_end(
        self, logs, loss: float, batch_id: int, is_training: bool = True
    ):
        pass


    def on_epoch_end(self, loss: float, logs, epoch: int):
        if self.trainer.distributed_context.is_leader:
            current_acc = logs.aux['acc'].mean()
            partial_acc = logs.aux['acc_or'].mean()
            self.log_to_wandb(
                {
                    "train_loss": loss, 
                    "epoch": epoch, 
                    'acc': current_acc, 
                    'acc_or':partial_acc
                }, 
                commit=True
            )

    def on_validation_end(self, loss: float, logs, epoch: int):
        if self.trainer.distributed_context.is_leader:
            current_acc = logs.aux['acc'].mean()
            partial_acc = logs.aux['acc_or'].mean()
            self.log_to_wandb(
                {
                    "validation_loss": loss, 
                    "epoch": epoch, 
                    'val_acc': current_acc, 
                    'val_acc_or':partial_acc
                }, 
                commit=False
            )

@registry.register("callback", "stream_analysis")
class StreamAnalysis(Callback):
    def __init__(self, streamer, train_data, test_data, config, indices, chars, all_data=None) -> None:
        super().__init__(streamer=streamer)
        self.train_data = train_data
        self.test_data = test_data
        self.args = config
        self.all_data = all_data
        self.indices = indices
        self.chars = chars

    def on_train_begin(self, trainer_instance):  # noqa: F821
        self.trainer = trainer_instance
        analyser = Analyser(
                self.args,
                self.trainer.game,
                self.train_data,
                self.indices,
                self.chars
            )
        analyser.get_interactions(self.train_data.to(self.args.device))
        self.on_epoch_end(loss=42, logs=analyser.interaction, epoch=0)

        

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        results = {}
        
        self.analyser = Analyser(
                self.args,
                self.trainer.game,
                self.train_data,
                self.indices,
                self.chars
            )
        
        #TODO: add further measures to logged results during training (atm only evaluated after training is finished)
        train_results = {}  

        self.analyser.get_interactions(self.train_data.to(self.args.device))        
        #signals = self.analyser.signals
        train_results[f"train_acc"] = self.analyser.interaction.aux['acc'].mean().item()
        train_results[f"train_acc_or"] = self.analyser.interaction.aux['acc_or'].mean().item() 
        
    # train_results = self.analyser.full_prob_analysis(
        #         self.train_data, 
        #         signals.tolist(), 
        #         n_gram_size=1,
        #         split_name=self.train_data.split
        #     )
        # train_results[f"{self.train_data.split}_acc"] = logs.aux['acc'].mean().item()
        # train_results[f"{self.train_data.split}_acc_or"] = logs.aux['acc_or'].mean().item()
        
        # if epoch % self.args.topsim_freq == 0:
        #     train_results[f"topsim"] = self.analyser.top_sim(
        #             signals.tolist(), 
        #             self.train_data.tensors[0]
        #         )

        # if epoch % self.args.posdis_freq == 0:
        #     train_results[f"posdis"] = self.analyser.pos_dis(
        #             self.train_data.tensors[0],
        #             signals
        #         )
                
        #for data in self.test_data:
        self.analyser.get_interactions(self.test_data.to(self.args.device))
        train_results[f"val_acc"] = self.analyser.interaction.aux['acc'].mean().item()
        train_results[f"val_acc_or"] = self.analyser.interaction.aux['acc_or'].mean().item()
            
        results.update(train_results)
        self.save_in_stream(results, epoch)
                
        if self.args.use_wandb:
            wandb.log(results, commit=False)

class CustomProgress(Progress):
    class CompletedColumn(ProgressColumn):
        def render(self, task):
            """Calculate common unit for completed and total."""
            download_status = f"{int(task.completed)}/{int(task.total)} btc"
            return Text(download_status, style="progress.download")

    class TransferSpeedColumn(ProgressColumn):
        """Renders human readable transfer speed."""

        def render(self, task):
            """Show data transfer speed."""
            speed = task.speed
            if speed is None:
                return Text("?", style="progress.data.speed")
            speed = f"{speed:,.{2}f}"
            return Text(f"{speed} btc/s", style="progress.data.speed")

    def __init__(self, *args, use_info_table: bool = True, **kwargs):
        super(CustomProgress, self).__init__(*args, **kwargs)

        self.info_table = Table(show_footer=False)
        self.info_table.add_column("Phase")

        self.test_style = "black on white"
        self.train_style = "white on black"
        self.use_info_table = use_info_table

    def add_info_table_cols(self, new_cols):
        """
        Add cols from ordered dict if not present in info_table
        """

        cols = set([x.header for x in self.info_table.columns])
        missing = set(new_cols) - cols
        if len(missing) == 0:
            return

        # iterate on new_cols since they are in order
        for c in new_cols:
            if c in missing:
                self.info_table.add_column(c)

    def update_info_table(self, aux: Dict[str, float], phase: str):
        """
        Update the info_table with the latest results
        :param aux:
        :param phase: either 'train' or 'test'
        """

        self.add_info_table_cols(aux.keys())
        epoch = aux.pop("epoch")
        aux = OrderedDict((k, f"{v:4.3f}") for k, v in aux.items())
        if phase == "train":
            st = self.train_style
        else:
            st = self.test_style
        self.info_table.add_row(phase, str(epoch), *list(aux.values()), style=st)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Display progress together with info table"""

        # this method is called once before the init, so check if the attribute is present
        if hasattr(self, "use_info_table"):
            use_table = self.use_info_table
            info_table = self.info_table
        else:
            use_table = False
            info_table = Table()

        if use_table:
            task_table = self.make_tasks_table(self.tasks)
            rendable = Columns((info_table, task_table), align="left", expand=True)
        else:
            rendable = self.make_tasks_table(self.tasks)

        yield rendable

class ProgressBarLogger(Callback):
    """
    Displays a progress bar with information about the current epoch and the epoch progression.
    """

    def __init__(
        self,
        n_epochs: int,
        train_data_len: int = 0,
        test_data_len: int = 0,
        use_info_table: bool = True,
    ):
        """
        :param n_epochs: total number of epochs
        :param train_data_len: length of the dataset generation for training
        :param test_data_len: length of the dataset generation for testing
        :param use_info_table: true to add an information table on top of the progress bar
        """

        self.n_epochs = n_epochs
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.use_info_table = use_info_table

        self.progress = CustomProgress(
            TextColumn(
                "[bold]Epoch {task.fields[cur_epoch]}/{task.fields[n_epochs]} | [blue]{task.fields[mode]}",
                justify="right",
            ),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            CustomProgress.CompletedColumn(),
            "•",
            CustomProgress.TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            use_info_table=use_info_table,
        )

        self.progress.start()
        self.train_p = self.progress.add_task(
            description="",
            mode="Train",
            cur_epoch=0,
            n_epochs=self.n_epochs,
            start=False,
            visible=False,
            total=self.train_data_len,
        )
        self.test_p = self.progress.add_task(
            description="",
            mode="Test",
            cur_epoch=0,
            n_epochs=self.n_epochs,
            start=False,
            visible=False,
            total=self.test_data_len,
        )

    @staticmethod
    def build_od(logs, loss, epoch):
        od = OrderedDict()
        od["epoch"] = epoch
        od["loss"] = loss
        aux = {k: float(torch.mean(v)) for k, v in logs.aux.items()}
        od.update(aux)

        return od

    def on_epoch_begin(self, epoch: int):
        self.progress.reset(
            task_id=self.train_p,
            total=self.train_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Train",
        )
        self.progress.start_task(self.train_p)
        self.progress.update(self.train_p, visible=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.progress.stop_task(self.train_p)
        self.progress.update(self.train_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.train_data_len == 0:
            self.train_data_len = self.progress.tasks[self.train_p].completed

        self.progress.reset(
            task_id=self.train_p,
            total=self.train_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Train",
        )

        if self.use_info_table:
            od = self.build_od(logs, loss, epoch)
            self.progress.update_info_table(od, "train")

    def on_validation_begin(self, epoch: int):
        self.progress.reset(
            task_id=self.test_p,
            total=self.test_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Test",
        )

        self.progress.start_task(self.test_p)
        self.progress.update(self.test_p, visible=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.progress.stop_task(self.test_p)
        self.progress.update(self.test_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.test_data_len == 0:
            self.test_data_len = self.progress.tasks[self.test_p].completed

        self.progress.reset(
            task_id=self.test_p,
            total=self.test_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Test",
        )

        if self.use_info_table:
            od = self.build_od(logs, loss, epoch)
            self.progress.update_info_table(od, "test")

    def on_train_end(self):
        self.progress.stop()

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training:
            self.progress.update(self.train_p, refresh=True, advance=1)
        else:
            self.progress.update(self.test_p, refresh=True, advance=1)