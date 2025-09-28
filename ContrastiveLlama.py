from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

import torch
import wandb
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput
import torch.nn.functional as F

from ContrastiveDataCollator import separate_batch_prompts

import os
os.environ["WANDB_MODE"] = "disabled"

@dataclass
class ContrastiveLlamaOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_original_instruction: torch.FloatTensor = None
    logits_per_paraphrased_instruction: torch.FloatTensor = None
    original_instruction_hidden_states: torch.FloatTensor = None
    paraphrased_instruction_hidden_states: torch.FloatTensor = None
    original_instruction_outputs: BaseModelOutputWithPast = None
    paraphrased_instruction_outputs: BaseModelOutputWithPast = None

@dataclass
class ContrastiveEvalOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.LongTensor = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = 1 if temperature is None else temperature
        self.cosine = nn.CosineSimilarity(dim=-1)


    def kl_divergence(self, original_output_logits, paraphrased_output_logits, temperature=1.0, pad_value=-1e4):
        orig0 = original_output_logits[0]
        orig1 = original_output_logits[1]
        para0 = paraphrased_output_logits[0]
        para1 = paraphrased_output_logits[1]
        
        def calculate_kl(logits_q, logits_p, temp, pad_val):
            len_q, voc_size = logits_q.shape
            len_p, _ = logits_p.shape
            
            target_len = max(len_q, len_p)
            if len_q < target_len:
                pad_length = target_len - len_q
                pad_tensor = torch.full((pad_length, voc_size), pad_val, 
                                    device=logits_q.device, dtype=logits_q.dtype)
                logits_q = torch.cat([logits_q, pad_tensor], dim=0)
            
            if len_p < target_len:
                pad_length = target_len - len_p
                pad_tensor = torch.full((pad_length, voc_size), pad_val,
                                    device=logits_p.device, dtype=logits_p.dtype)
                logits_p = torch.cat([logits_p, pad_tensor], dim=0)
            
            q = F.softmax(logits_q / temp, dim=-1)
            p = F.softmax(logits_p / temp, dim=-1)
            
            kl = F.kl_div(
                torch.log(q + 1e-10),
                p,
                reduction='batchmean',
                log_target=False
            )
            return kl
        
        k1 = calculate_kl(para0, orig0, temperature, pad_value)
        k2 = calculate_kl(para0, orig1, temperature, pad_value)
        k3 = calculate_kl(para1, orig1, temperature, pad_value)
        k4 = calculate_kl(para1, orig0, temperature, pad_value)
        
        original_loss = k1 - k2 + k3 - k4
        loss = torch.exp(original_loss - torch.max(torch.tensor(0.0, device=original_loss.device), original_loss))

        return loss
        

    def get_output_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:        
        output_mask = (labels != -100).float()  # (batch_size, seq_len)
        
        output_logits_list = []
        for i in range(logits.size(0)):
            sample_mask = output_mask[i] == 1.0
            output_logits = logits[i, sample_mask]
            output_logits_list.append(output_logits)
        
        return output_logits_list

    def compute_log_probs(self, logits, labels, attention_mask=None):
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        batch_size, seq_len = labels.shape
        labels_reshaped = labels.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]

        mask = (labels != -100).float() 
        labels_reshaped = labels_reshaped * mask.unsqueeze(2).long() 

        token_log_probs = log_probs.gather(dim=2, index=labels_reshaped).squeeze(2)  # [batch_size, seq_len]
        token_log_probs = token_log_probs * mask
        
        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask
        
        sequence_log_probs = token_log_probs.sum(dim=1)  # [batch_size]
        
        return sequence_log_probs

    def compute_dpo_loss(self, original_logits, paraphrased_logits, labels, attention_mask=None):
        log_p_original = self.compute_log_probs(original_logits, labels, attention_mask)
        log_p_paraphrased = self.compute_log_probs(paraphrased_logits, labels, attention_mask)
        
        loss = -torch.log(torch.sigmoid(log_p_paraphrased - log_p_original)).mean()
        
        return loss

    def forward(self, original_logits, paraphrased_logits, original_labels):
        # loss = self.compute_dpo_loss(original_logits, paraphrased_logits, original_labels)
        # return loss

        original_output_logits = self.get_output_logits(original_logits, original_labels)
        paraphrased_output_logits = self.get_output_logits(paraphrased_logits, original_labels)

        losses = []
        for i in range(0, len(original_output_logits), 2):
            original_pair = original_output_logits[i:i+2]
            paraphrased_pair = paraphrased_output_logits[i:i+2]
            pair_loss = self.kl_divergence(original_pair, paraphrased_pair)
            losses.append(pair_loss)

        loss = torch.mean(torch.stack(losses))
        
        return loss



class ContrastiveLlama(LlamaPreTrainedModel):
    def __init__(self, config, do_predict=False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # From original LlamaForCausalLM
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Parameters for Contrastive loss
        self.pooling_method = config.pooling_method
        self.temperature = config.temperature
        self.contrastive_loss = ContrastiveLoss(self.temperature)
        self.contrastive_loss_ratio = config.contrastive_loss_ratio

        self.do_predict = do_predict
        self.do_contrastive = config.do_contrastive

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_decoder_outputs(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def get_entropy_loss_for_token_prediction(self, logits, labels):
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        return loss

    def get_pooled_hidden_states(self, hidden_states):
        """
        Get hidden states of the last token of each sequence (reference: LlamaForSequenceClassification)
        hidden_states: (batch_size, seq_length, vocab_num)
        return: (batch_size, vocab_num)
        """
        if self.pooling_method == 'last':
            return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), -1]
        elif 'average' in self.pooling_method:
            if self.pooling_method == 'average_first_last':
                hidden_states = torch.cat((hidden_states[:, 0], hidden_states[:, -1])).unsqueeze(0)
            if self.pooling_method == 'average_first_last' or self.pooling_method == 'average_all':
                return torch.mean(hidden_states, dim=1)
            else:
                raise ValueError(f"Pooling method {self.self.pooling_method} not supported")
        elif self.pooling_method == 'max':
            return torch.max(hidden_states, dim=1).values
        else:
            raise ValueError(f"Pooling method {self.pooling_metlora_rhod} not supported")

    def get_tensor_except_i(self, input_tensor, i):
        seq_length = input_tensor.size(0)
        if i == 0:
            new_tensor = input_tensor[1:, :, :]
        elif i == seq_length - 1:
            new_tensor = input_tensor[:-1, :, :]
        else:
            left = input_tensor[:i, :, :]
            right = input_tensor[i + 1:, :, :]
            new_tensor = torch.cat((left, right), dim=1)
        return new_tensor

    def scale_contrastive_loss(self, generation_loss, contrastive_loss, max_scale_ratio):
        if contrastive_loss != 0 and contrastive_loss > generation_loss:
            new_contrastive_loss = contrastive_loss * (
                min(max_scale_ratio, generation_loss.detach() / contrastive_loss.detach()))
        else:
            new_contrastive_loss = contrastive_loss
        return new_contrastive_loss

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Union[ContrastiveLlamaOutput, ContrastiveEvalOutput]:
        if not self.do_predict:
            original_tokenized_full_prompt, paraphrased_tokenized_full_prompt = separate_batch_prompts(input_ids,
                                                                                                       attention_mask,
                                                                                                       labels,
                                                                                                       int(input_ids.size(
                                                                                                           0) / 2))
            original_outputs = self.get_decoder_outputs(**original_tokenized_full_prompt)
            paraphrased_outputs = self.get_decoder_outputs(**paraphrased_tokenized_full_prompt)

            original_entire_sentence_hidden_states = original_outputs[0]
            paraphrased_entire_sentence_hidden_states = paraphrased_outputs[0]

            original_instruction_logits = self.lm_head(original_entire_sentence_hidden_states)
            paraphrased_instruction_logits = self.lm_head(paraphrased_entire_sentence_hidden_states)


            # Contrastive Loss
            contrastive_loss = 0
            if self.do_contrastive:
                contrastive_loss = self.contrastive_loss(
                    original_instruction_logits,
                    paraphrased_instruction_logits,
                    original_tokenized_full_prompt["labels"],
                )

            # Generation loss
            original_instruction_loss = self.get_entropy_loss_for_token_prediction(
                original_instruction_logits,
                original_tokenized_full_prompt["labels"]
            )
            paraphrased_instruction_loss = self.get_entropy_loss_for_token_prediction(
                paraphrased_instruction_logits,
                paraphrased_tokenized_full_prompt["labels"]
            )

            generation_loss = original_instruction_loss + paraphrased_instruction_loss


            contrastive_loss = contrastive_loss * self.contrastive_loss_ratio
            contrastive_loss = self.scale_contrastive_loss(generation_loss, contrastive_loss,
                                                           self.contrastive_loss_ratio)

            loss = contrastive_loss + generation_loss

            wandb.log({
                'total_loss': loss,
                'contrastive_loss': contrastive_loss,
                'generation_loss': generation_loss,
                'original_generation_loss': original_instruction_loss,
                'paraphrased_generation_loss': paraphrased_instruction_loss
            })

            return ContrastiveLlamaOutput(
                loss=loss,
                original_instruction_hidden_states=original_entire_sentence_hidden_states,
                paraphrased_instruction_hidden_states=paraphrased_entire_sentence_hidden_states,
                original_instruction_outputs=original_outputs,
                paraphrased_instruction_outputs=paraphrased_outputs,
            )
        else:
            # For evaluation (process single inputs instead of pair-wise for contrastive learning)
            outputs = self.get_decoder_outputs(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                labels
            )
            logits = self.lm_head(outputs[0])
            loss = self.get_entropy_loss_for_token_prediction(logits, labels)
            return ContrastiveEvalOutput(
                loss=loss,
                logits=logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states
            )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
