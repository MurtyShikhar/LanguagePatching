from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from munch import Munch

EPS = 1e-9

from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

import warnings
import copy

from torch.nn import CrossEntropyLoss

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5ForConditionalGenerationMultipleHeads(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        aux_decoder=None,
        aux_lm_head=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        if aux_decoder:
            decoder = aux_decoder
        else:
            decoder = self.decoder

        if self.model_parallel:
            torch.cuda.set_device(decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(decoder.first_device)
            hidden_states = hidden_states.to(decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(decoder.first_device)

        # Decode
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if aux_lm_head:
            lm_head = aux_lm_head
        else:
            lm_head = self.lm_head

        lm_logits = lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def get_from_pretrained_t5():
    print("splicing parts from pretrained model")
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    aux_decoder = model.decoder
    aux_lm_head = model.lm_head
    return aux_decoder, aux_lm_head


class T5Interpeter(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        label_list=["positive", "negative"],
        primary_mode="task_predictor",
        train_multihead=False,
    ):
        super().__init__()
        self.model = model
        self.primary_mode = primary_mode
        self.train_multihead = train_multihead
        print("primary mode: {}".format(primary_mode))
        if self.train_multihead:
            decoder_config = copy.deepcopy(self.model.config)
            decoder_config.is_decoder = True
            decoder_config.is_encoder_decoder = False
            decoder_config.num_layers = self.model.config.num_decoder_layers

            aux_decoder, aux_lm_head = get_from_pretrained_t5()
            self.aux_decoder = aux_decoder
            self.aux_lm_head = aux_lm_head
            """
            self.aux_decoder = T5Stack(decoder_config,
                                       nn.Embedding(self.model.config.vocab_size, self.model.config.d_model))
            self.aux_lm_head = nn.Linear(self.model.config.d_model, self.model.config.vocab_size, bias=False)
            """
        else:
            self.aux_decoder = None
            self.aux_lm_head = None

        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss()
        pos_idx = tokenizer(label_list[0])["input_ids"]
        neg_idx = tokenizer(label_list[1])["input_ids"]
        self.pos_idx = pos_idx[0]
        self.neg_idx = neg_idx[0]
        self.label_list = [self.pos_idx, self.neg_idx]
        self.label_list_words = label_list

    def forward_helper(self, batch, mode):
        for key in batch:
            batch[key] = batch[key].to(self.model.device)
        # labels are -100 unless the input_id refers to either positive or negative
        if mode == "patch_applies_predictor":
            assert self.aux_decoder is not None
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                aux_decoder=self.aux_decoder,
                aux_lm_head=self.aux_lm_head,
            )
        else:
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        return out

    def get_task_tensors(self, logits, batch):
        cls_logits = logits[:, 0]
        if "labels" in batch:
            return cls_logits, batch["labels"][:, 0]
        else:
            return cls_logits, None

    def compute_confusion_matrix(self, preds, labels_curr):
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for pred, label in zip(preds, labels_curr):
            # label might be padded
            if type(label) == list and label[-1] == -100:
                idx = label.index(-100)
                label = label[:idx]
            if label == self.pos_idx:
                tp += int(pred == self.pos_idx)
                fn += int(pred == self.neg_idx)
            else:
                tn += int(pred == self.neg_idx)
                fp += int(pred == self.pos_idx)
        return tp, tn, fp, fn

    def get_acc(self, batch, mode):
        with torch.no_grad():
            out = self.forward_helper(batch, mode=mode)
        logits, labels = self.get_task_tensors(out.logits, batch)
        labels = labels.cpu().tolist()
        task_logits = logits[
            :, self.label_list
        ]  # first logit is for positive, second logit is for negative.
        preds = task_logits.argmax(dim=-1)

        # just compare positive and negative
        preds_task = [self.label_list[pred] for pred in preds]
        return task_logits, labels, preds_task

    def get_loss(self, batch):
        if type(batch) == dict:
            out_list = []
            for key in batch:
                if key == "patch_grounding_data":
                    out_list.append(
                        self.forward_helper(batch[key], mode="patch_applies_predictor")
                    )
                else:
                    out_list.append(
                        self.forward_helper(batch[key], mode="task_predictor")
                    )
            loss_curr = sum(out.loss for out in out_list)
        else:
            out = self.forward_helper(batch, mode=self.primary_mode)
            loss_curr = out.loss
        try:
            wandb.log({"loss": loss_curr.item()})
        except:
            pass
        return loss_curr

    def evaluator(self, examples, mode=None, verbose=True):
        task_logits_all = []
        labels = []

        correct = 0.0
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        if not mode:
            mode = self.primary_mode
        if verbose:
            iterate_over = tqdm(examples)
        else:
            iterate_over = examples

        for batch in iterate_over:
            task_logits, labels_curr, preds = self.get_acc(batch, mode)
            # sum(p == l for p, l in zip(preds, labels_curr))
            tp_curr, tn_curr, fp_curr, fn_curr = self.compute_confusion_matrix(
                preds, labels_curr
            )
            tp += tp_curr
            fp += fp_curr
            tn += tn_curr
            fn += fn_curr

            correct += tp_curr + tn_curr
            task_logits_all.append(task_logits)
            labels += labels_curr

        task_logits = torch.cat(task_logits_all)
        probs = F.softmax(task_logits, dim=1).cpu().numpy()
        precision = tp / (tp + fp + EPS)  # prevent div by 0
        recall = tp / (tp + fn + EPS)  # prevent div by 0
        f1 = 2 * precision * recall / (precision + recall + EPS)

        return {
            "labels": labels,
            "probs": probs,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "logits": task_logits.cpu(),
            "acc": (correct) / (1.0 * len(labels)),
        }
