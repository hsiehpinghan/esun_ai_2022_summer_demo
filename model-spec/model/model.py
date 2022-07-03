import torch
import torch.nn as nn
import torchmetrics

from transformers import BertForMaskedLM
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EsunModel(LightningModule):
    def __init__(self, tokenizer, max_length, model_name, lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = BertForMaskedLM.from_pretrained(model_name)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lr = lr
        self.correction_acc_metric_train = torchmetrics.Accuracy()
        self.char_error_rate_metric_train = torchmetrics.CharErrorRate()
        self.correction_acc_metric_val = torchmetrics.Accuracy()
        self.char_error_rate_metric_val = torchmetrics.CharErrorRate()
        self.correction_acc_metric_test = torchmetrics.Accuracy()
        self.char_error_rate_metric_test = torchmetrics.CharErrorRate()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(),
                                     lr=self.lr)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': ReduceLROnPlateau(optimizer=optimizer,
                                                                mode='min',
                                                                factor=0.1,
                                                                patience=3),
                                 'monitor': 'train_epoch_loss',
                                 'frequency': 1}}

    def forward(self, original_texts, correct_texts=None):
        if correct_texts:
            correct_input_ids = self.tokenizer(correct_texts,
                                               padding=True,
                                               truncation=True,
                                               max_length=self.max_length,
                                               return_tensors='pt')['input_ids']
            correct_input_ids[correct_input_ids==0] = -100
            correct_input_ids = correct_input_ids.to(self.device)
        else:
            correct_input_ids = None
        encoded_text = self.tokenizer(original_texts,
                                      padding=True,
                                      truncation=True,
                                      max_length=self.max_length,
                                      return_tensors='pt')
        encoded_text.to(self.device)
        bert_outputs = self.bert(**encoded_text,
                                 labels=correct_input_ids,
                                 return_dict=True,
                                 output_hidden_states=True)
        active_indexes = encoded_text['attention_mask'].view(-1, bert_outputs.hidden_states[-1].shape[1]) == 1
        if correct_input_ids is None:
            outputs = {'bert_outputs_logits': bert_outputs.logits,
                       'active_indexes': active_indexes}
        else:
            outputs = {'bert_outputs_logits': bert_outputs.logits,
                       'active_indexes': active_indexes,
                       'correct_input_ids': correct_input_ids,
                       'bert_outputs_loss': bert_outputs.loss}
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch=batch,
                          correction_acc_metric=self.correction_acc_metric_train,
                          char_error_rate_metric=self.char_error_rate_metric_train)

    def validation_step(self, batch, batch_idx):
        return self._step(batch=batch,
                          correction_acc_metric=self.correction_acc_metric_val,
                          char_error_rate_metric=self.char_error_rate_metric_val)

    def test_step(self, batch, batch_idx):
        return self._step(batch=batch,
                          correction_acc_metric=self.correction_acc_metric_test,
                          char_error_rate_metric=self.char_error_rate_metric_test)

    def predict_step(self, batch, batch_idx):
        (original_texts, correct_texts) = batch
        outputs = self(original_texts=original_texts,
                       correct_texts=correct_texts)
        bert_outputs_logits = outputs['bert_outputs_logits']
        active_indexes = outputs['active_indexes']
        predict_input_ids = torch.argmax(bert_outputs_logits,
                                         dim=-1)
        predict_texts = []
        for (sub_predict_input_ids, sub_active_indexes) in zip(predict_input_ids, active_indexes):
            predict_text = self.tokenizer.decode(sub_predict_input_ids,
                                                 skip_special_tokens=True).replace(' ', '')
            end_index = torch.sum(sub_active_indexes==True)
            predict_text = predict_text[:(end_index-2)]
            predict_texts.append(predict_text)
        return {'predict_texts': predict_texts}

    def training_step_end(self, outputs):
        return self._step_end(prefix='train',
                              outputs=outputs,
                              correction_acc_metric=self.correction_acc_metric_train,
                              char_error_rate_metric=self.char_error_rate_metric_train)

    def training_epoch_end(self, outputs_list):
        self._epoch_end(prefix='train',
                        outputs_list=outputs_list,
                        correction_acc_metric=self.correction_acc_metric_train,
                        char_error_rate_metric=self.char_error_rate_metric_train)

    def validation_epoch_end(self, outputs_list):
        self._epoch_end(prefix='val',
                        outputs_list=outputs_list,
                        correction_acc_metric=self.correction_acc_metric_val,
                        char_error_rate_metric=self.char_error_rate_metric_val)

    def test_epoch_end(self, outputs_list) -> None:
        self._epoch_end(prefix='test',
                        outputs_list=outputs_list,
                        correction_acc_metric=self.correction_acc_metric_test,
                        char_error_rate_metric=self.char_error_rate_metric_test)

    def _step(self, batch, correction_acc_metric, char_error_rate_metric):
        (original_texts, correct_texts) = batch
        outputs = self(original_texts=original_texts,
                       correct_texts=correct_texts)
        bert_outputs_logits = outputs['bert_outputs_logits']
        correct_input_ids = outputs['correct_input_ids']
        active_indexes = outputs['active_indexes']
        bert_outputs_loss = outputs['bert_outputs_loss']
        predict_input_ids = torch.argmax(bert_outputs_logits,
                                         dim=-1)
        active_predict_input_ids = predict_input_ids[active_indexes]
        active_correct_input_ids = correct_input_ids[active_indexes]
        correction_acc_metric(active_predict_input_ids, active_correct_input_ids)
        predict_texts = []
        assert len(predict_input_ids) == len(active_indexes)
        for (sub_predict_input_ids, sub_active_indexes) in zip(predict_input_ids, active_indexes):
            predict_text = self.tokenizer.decode(sub_predict_input_ids,
                                                 skip_special_tokens=True).replace(' ', '')
            end_index = torch.sum(sub_active_indexes==True)
            predict_text = predict_text[:(end_index-2)]
            predict_texts.append(predict_text)
        char_error_rate_metric(predict_texts, correct_texts)
        loss = bert_outputs_loss
        return {'bert_outputs_loss': bert_outputs_loss,
                'loss': loss}
    
    def _step_end(self, prefix, outputs, correction_acc_metric, char_error_rate_metric):
        self.log_dict(dictionary={f'{prefix}_step_bert_outputs_loss': outputs['bert_outputs_loss'],
                                  f'{prefix}_step_loss': outputs['loss'],
                                  f'{prefix}_step_correction_acc': correction_acc_metric,
                                  f'{prefix}_step_char_error_rate': char_error_rate_metric},
                      on_step=True,
                      on_epoch=False,
                      prog_bar=True,
                      logger=True)

    def _epoch_end(self, outputs_list, prefix, correction_acc_metric, char_error_rate_metric):
        self.log_dict(dictionary={f'{prefix}_epoch_bert_outputs_loss': torch.stack([outputs['bert_outputs_loss'] for outputs in outputs_list]),
                                  f'{prefix}_epoch_loss': torch.stack([outputs['loss'] for outputs in outputs_list]),
                                  f'{prefix}_epoch_correction_accuracy': correction_acc_metric,
                                  f'{prefix}_epoch_char_error_rate': char_error_rate_metric},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True)