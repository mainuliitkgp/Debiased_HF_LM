# generate full sentences

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
import scipy.stats
import time
import random
import os
import sys
import argparse
from typing import Iterable, Optional, Tuple


import transformers
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    AutoModelWithLMHead, 
    AutoTokenizer,
)


#############################
# Copied from huggingface code
##############################
def _use_cache(outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        return True


def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            enforce_repetition_penalty_(
                scores, batch_size, num_beams, input_ids, repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

            for i, banned_tokens in enumerate(banned_tokens):
                scores[i, banned_tokens] = -float("inf")

        return scores
    
    
def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens
######################################



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="INLP",
                        help="choose the algorithm: INLP, A-INLP, subspace")
    parser.add_argument("--prompt", type=str, default="",
                        help="Please provide a prompt to generate a text")
    args = parser.parse_args()
    return args


def top_k_top_p_filtering(
    logits,    # (1, 50257)
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def mean_ds(x, dim=None):
    return (
        x.float().mean().type_as(x)
        if dim is None
        else x.float().mean(dim).type_as(x)
    )


def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)


# hyperparameters
p = 0.7  # used for top k filtering
nums_iter = 1
do_sample = True
max_len = 30
no_repeat_ngram_size = 3
bad_words_ids = None
min_len = 0
repetition_penalty = 1.5
batch_size = 5
eos_token_id = 50256 # model.config.eos_token_id
pad_token_id = eos_token_id
temperature = 1.0
top_k = 0
top_p = 0.9
bias_thre = (0.15, -0.1)


def generate_sentences(prompt_text):
            MODEL_CLASSES = {
                "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
                "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
                "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
                "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
                "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
                "xlm": (XLMWithLMHeadModel, XLMTokenizer),
            }
            model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
            tokenizer = tokenizer_class.from_pretrained("gpt2")
            model = model_class.from_pretrained("gpt2")
            device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            P = np.load("./P.npy")

            embedding = model.lm_head.weight.cpu().detach().numpy()
            embedding_norm = np.array([x / np.linalg.norm(x) for x in embedding])

            method = "A-INLP"

            gender_direction = np.load("./gpt2_gender_direction.npy")
          
            A = [1.0]

            for a in range(len(A)):
                ppl = 0.
                generated_sentence = []
                for i in range(nums_iter):
                    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
                    input_list = input_ids.cpu().detach().numpy().tolist()[0]
                    input_lists = [input_list for ii in range(batch_size)]
                    input_ids = torch.LongTensor(input_lists)       # [nums, len_template]
                    input_ids = input_ids.to(device)

                    past, attention_mask, use_cache = None, input_ids.new_ones(input_ids.shape), True
                    unfinished_sents = input_ids.new(batch_size).fill_(1)
                    sent_lengths = input_ids.new(batch_size).fill_(max_len)
                    cur_len = input_ids.shape[-1]
                    out = None

                    while cur_len < max_len:
                        model_inputs = model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask,
                                                                           use_cache=use_cache)

                        outputs = model(**model_inputs)     # [0]: (batch_size, seq_len, vocab_size)

                        # out is used to calculate ppl
                        if out is None:     # (batch, pos, dim)
                            out = outputs[0][:, -1:, :].clone()      # embedding of last token
                        else:
                            out = torch.cat((out, outputs[0][:, -1:, :].clone()), 1)

                        ratio = [A[a] for ii in range(batch_size)]    # alpha across the batch

                        if method == "A-INLP":
                            next_token_logits = outputs[0][:, -1, :]  # batch * vocab
                            scores = postprocess_next_token_scores(  # batch * vocab
                                scores=next_token_logits,
                                input_ids=input_ids,
                                no_repeat_ngram_size=no_repeat_ngram_size,
                                bad_words_ids=bad_words_ids,
                                cur_len=cur_len,
                                min_length=min_len,
                                max_length=max_len,
                                eos_token_id=eos_token_id,
                                repetition_penalty=repetition_penalty,
                                batch_size=batch_size,
                                num_beams=1,
                            )
                            if do_sample:
                                # Temperature (higher temperature => more likely to sample low probability tokens)
                                if temperature != 1.0:
                                    scores = scores / temperature
                            logits_filter = top_k_top_p_filtering(scores, top_p=p)  # batch * vocab
                            top_p_mask = logits_filter.eq(-float("Inf"))  # batch * vocab

                            # bias sensitive tokens
                            top_k_tokens = []
                            for ii in range(batch_size):
                                tmp = (top_p_mask == False)[ii].nonzero().cpu().detach().numpy().tolist()  # batch tuple
                                top_k_tokens.append([x[0] for x in tmp])
                            probs_bias = F.softmax(logits_filter, dim=-1).cpu().detach().numpy()  # batch * vocab
                            for ii in range(batch_size):
                                bias = 0.
                                for t in top_k_tokens[ii]:
                                    bias += embedding[int(t)].dot(gender_direction) / np.linalg.norm(embedding[int(t)]) \
                                            * probs_bias[ii][int(t)]
                                if bias <= bias_thre[0] and bias >= bias_thre[1]:
                                    ratio[ii] = 1
                                else:
                                    ratio[ii] = max(1 - abs(bias), 0.6)

                        outputs_P = model.transformer(input_ids=input_ids)[0][:, -1].cpu().detach().numpy()  # transformer output: (2, batch, len, dim), output_P: (batch, dim)
                        outputs_P = np.multiply(np.array([1-ratio[ii] for ii in range(batch_size)]).reshape(-1, 1), outputs_P.dot(P)) + \
                                    np.multiply(np.array([ratio[ii] for ii in range(batch_size)]).reshape(-1, 1), outputs_P)
                        new_logits = outputs_P.dot(np.transpose(embedding))     # batch * vocab
                        new_logits = torch.from_numpy(new_logits).float()
                        new_logits = new_logits.to(device)
                        next_token_logits = new_logits

                        scores = postprocess_next_token_scores(
                            scores=next_token_logits,
                            input_ids=input_ids,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            bad_words_ids=bad_words_ids,
                            cur_len=cur_len,
                            min_length=min_len,
                            max_length=max_len,
                            eos_token_id=eos_token_id,
                            repetition_penalty=repetition_penalty,
                            batch_size=batch_size,
                            num_beams=1,
                        )

                        if _use_cache(outputs, use_cache):
                            past = outputs[1]

                        if do_sample:
                            # Temperature (higher temperature => more likely to sample low probability tokens)
                            if temperature != 1.0:
                                scores = scores / temperature
                            # Top-p/top-k filtering
                            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)  # batch * vocab
                            # Sample
                            probs = F.softmax(next_token_logscores, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                        else:
                            # Greedy decoding
                            next_token = torch.argmax(next_token_logits, dim=-1)

                        # update generations and finished sentences
                        if eos_token_id is not None:
                            # pad finished sentences if eos_token_id exist
                            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
                        else:
                            tokens_to_add = next_token

                        # add token and increase length by one
                        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                        cur_len = cur_len + 1

                        if eos_token_id is not None:
                            eos_in_sents = tokens_to_add == eos_token_id
                            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                            # unfinished_sents is set to zero if eos in sentence
                            unfinished_sents.mul_((~eos_in_sents).long())

                        # stop when there is a </s> in each sentence, or if we exceed the maximul length
                        if unfinished_sents.max() == 0:
                            break

                        # extend attention_mask for new generated input if only decoder
                        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                                                   dim=-1)
                    logits = F.log_softmax(out, dim=-1)     # batch * seq_len * vocab
                    # print(logits.size(), input_ids[:, -logits.shape[1]:].size())
                    losses = F.nll_loss(logits.reshape(-1, logits.shape[-1]), input_ids[:, -logits.shape[1]:].reshape(-1).to(logits.device), reduction='none')
                    nll_loss = mean_ds(losses)
                    perplexity = 2 ** nll_loss.cpu().detach().numpy()
                    ppl += perplexity
                    #print("avg perplextity: ", perplexity)
                    # print(input_ids.tolist()[0])
                    for ii in range(batch_size):
                        gen_sent = tokenizer.decode(input_ids.tolist()[ii], clean_up_tokenization_spaces=True)
                        print(gen_sent)
                        if '\n' in gen_sent:
                            gen_idx = gen_sent.index('\n')
                        else:
                            gen_idx = len(gen_sent)
                        generated_sentence.append(gen_sent[:gen_idx])


if __name__ == '__main__':
    print('Not Implemented.')

