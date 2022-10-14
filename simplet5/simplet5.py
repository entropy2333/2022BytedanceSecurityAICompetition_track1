import torch
import numpy as np
import pandas as pd
from transformers import (T5ForConditionalGeneration, MT5ForConditionalGeneration, ByT5Tokenizer,
                          PreTrainedTokenizer, T5TokenizerFast as T5Tokenizer, MT5TokenizerFast as
                          MT5Tokenizer, BertTokenizerFast as BartTokenizer,
                          BartForConditionalGeneration)
from .modeling_cpt import (
    CPTForConditionalGeneration,)
from .fgm import FGM
from typing import List
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

torch.cuda.empty_cache()
pl.seed_everything(42)


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[labels == 0] = -100  # to make sure we have correct labels for T5 text generation

        return dict(
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )


class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        num_workers: int = 2,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer,
        model,
        output_dir: str = "outputs",
        save_only_last_epoch: bool = False,
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.1,
        num_training_steps: int = 1000,
        use_fgm: bool = False,
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            output_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.num_training_steps = num_training_steps
        self.fgm = FGM(self.model, epsilon=0.5) if use_fgm else None
        self.automatic_optimization = False if use_fgm else True

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        if self.fgm is not None:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            opt.zero_grad()

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)

        if self.fgm is not None:
            self.manual_backward(loss)
            self.fgm.attack()
            loss_adv, outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=labels_attention_mask,
                labels=labels,
            )
            self.fgm.restore()
            self.manual_backward(loss_adv)
            opt.step()
            scheduler.step()

        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        sceduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_ratio * self.num_training_steps,
            num_training_steps=self.num_training_steps,
        )
        sceduler = {
            "scheduler": sceduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [sceduler]

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        path = f"{self.output_dir}/simplet5-epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )


class SimpleT5:
    """ Custom SimpleT5 class """

    def __init__(self) -> None:
        """ initiates SimpleT5 class """
        pass

    def from_pretrained(self,
                        model_type="t5",
                        model_name="t5-base",
                        special_tokens=[],
                        **kwargs) -> None:
        """
        loads T5/MT5 Model model for training/finetuning
        Args:
            model_type (str, optional): "t5" or "mt5" . Defaults to "t5".
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
        """
        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}",
                                                         add_special_tokens=special_tokens,
                                                         **kwargs)
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_name}",
                                                                    return_dict=True)
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}",
                                                          add_special_tokens=special_tokens,
                                                          **kwargs)
            self.model = MT5ForConditionalGeneration.from_pretrained(f"{model_name}",
                                                                     return_dict=True)
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}",
                                                           add_special_tokens=special_tokens,
                                                           **kwargs)
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_name}",
                                                                    return_dict=True)
        elif model_type == "bart":
            self.tokenizer = BartTokenizer.from_pretrained(f"{model_name}",
                                                           add_special_tokens=special_tokens,
                                                           **kwargs)
            self.model = BartForConditionalGeneration.from_pretrained(f"{model_name}",
                                                                      return_dict=True)
        elif model_type == "cpt":
            self.tokenizer = BartTokenizer.from_pretrained(f"{model_name}",
                                                           add_special_tokens=special_tokens,
                                                           **kwargs)
            self.model = CPTForConditionalGeneration.from_pretrained(f"{model_name}",
                                                                     return_dict=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: bool = True,
        output_dir: str = "outputs",
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32,
        logger="default",
        dataloader_num_workers: int = 2,
        save_only_last_epoch: bool = False,
        accumulate_grad_batches: int = 1,
        learning_rate: float = 1e-4,
        gradient_clip_algorithm: str = "norm",
        gradient_clip_val: float = 1.0,
        warmup_ratio: float = 0.1,
        use_fgm: bool = False,
    ):
        """
        trains T5/MT5 model on custom dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            output_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
            logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.
            dataloader_num_workers (int, optional): number of workers in train/test/val dataloader
            save_only_last_epoch (bool, optional): If True, saves only the last epoch else models are saved at every epoch
        """
        self.data_module = LightningDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.T5Model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            output_dir=output_dir,
            save_only_last_epoch=save_only_last_epoch,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            num_training_steps=max_epochs * len(train_df) // batch_size,
            use_fgm=use_fgm,
        )

        # add callbacks
        callbacks = [
            TQDMProgressBar(refresh_rate=50),
            LearningRateMonitor(logging_interval="step"),
        ]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        # add gpu support
        gpus = torch.cuda.device_count() if use_gpu else 0

        # add logger
        loggers = True if logger == "default" else logger

        # prepare trainer
        trainer = pl.Trainer(
            logger=loggers,
            callbacks=callbacks,
            max_epochs=max_epochs,
            gpus=gpus,
            precision=precision,
            # log_every_n_steps=1,
            strategy=DDPStrategy(find_unused_parameters=False) if gpus > 1 else None,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            # track_grad_norm=2,
        )

        # fit trainer
        trainer.fit(self.T5Model, self.data_module)

    def load_model(self,
                   model_type: str = "t5",
                   model_dir: str = "outputs",
                   use_gpu: bool = False,
                   special_tokens: list = [],
                   **kwargs):
        """
        loads a checkpoint for inferencing/prediction
        Args:
            model_type (str, optional): "t5" or "mt5". Defaults to "t5".
            model_dir (str, optional): path to model directory. Defaults to "outputs".
            use_gpu (bool, optional): if True, model uses gpu for inferencing/prediction. Defaults to True.
        """
        if model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}",
                                                         add_special_tokens=special_tokens,
                                                         **kwargs)
        elif model_type == "mt5":
            self.model = MT5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_dir}",
                                                          add_special_tokens=special_tokens,
                                                          **kwargs)
        elif model_type == "byt5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_dir}",
                                                           add_special_tokens=special_tokens,
                                                           **kwargs)
        elif model_type == "bart":
            self.model = BartForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = BartTokenizer.from_pretrained(f"{model_dir}",
                                                           add_special_tokens=special_tokens,
                                                           **kwargs)
        elif model_type == "cpt":
            self.model = CPTForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = BartTokenizer.from_pretrained(f"{model_dir}",
                                                           add_special_tokens=special_tokens,
                                                           **kwargs)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(source_text, return_tensors="pt", add_special_tokens=True)
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            ) for g in generated_ids
        ]
        return preds

    def batch_predict(
        self,
        source_text: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        inputs = self.tokenizer.batch_encode_plus(source_text,
                                                  return_tensors="pt",
                                                  padding="longest",
                                                  truncation=True,
                                                  max_length=max_length,
                                                  add_special_tokens=True,
                                                  return_token_type_ids=False)
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
        preds = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return preds