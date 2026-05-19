// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 Binbin Zhang (binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decoder/asr_decoder.h"

#include <ctype.h>

#include <algorithm>
#include <limits>
#include <random>
#include <utility>
#include "utils/string.h"
#include "utils/timer.h"

namespace wenet {

AsrDecoder::AsrDecoder(std::shared_ptr<FeaturePipeline> feature_pipeline,
                       std::shared_ptr<DecodeResource> resource,
                       const DecodeOptions& opts)
    : feature_pipeline_(std::move(feature_pipeline)),
      model_(resource->model->Copy()),
      post_processor_(resource->post_processor),
      context_hanzi_graph_(resource->context_hanzi_graph),
      corrector_(resource->corrector),
      hotword_cache_(resource->hotword_cache),
      symbol_table_(resource->symbol_table),
      fst_(resource->fst),
      unit_table_(resource->unit_table),
      oov_mapping_(resource->oov_mapping),
      opts_(opts),
      max_append_path(resource->max_append_path),
      ctc_endpointer_(new CtcEndpoint(opts.ctc_endpoint_config)) {
  if (opts_.reverse_weight > 0) {
    CHECK(model_->is_bidirectional_decoder());
  }

  if (nullptr == fst_) {
    searcher_.reset(new CtcPrefixBeamSearch(opts.ctc_prefix_search_opts,
                                            resource->context_hanzi_graph));
  } else {
    searcher_.reset(new CtcWfstBeamSearch(*fst_, opts.ctc_wfst_search_opts,
                                          resource->context_hanzi_graph));
  }

  VLOG(1) << "Current searcher type: "
          << (dynamic_cast<CtcWfstBeamSearch*>(searcher_.get()) ? "WFST"
                                                                : "PrefixBeam");

  ctc_endpointer_->frame_shift_in_ms(frame_shift_in_ms());
}

void AsrDecoder::Reset() {
  start_ = false;
  result_.clear();
  num_frames_ = 0;
  global_frame_offset_ = 0;
  model_->Reset();
  searcher_->Reset();
  feature_pipeline_->Reset();
  ctc_endpointer_->Reset();

  global_ctc_log_probs_.clear();
}

void AsrDecoder::ResetContinuousDecoding() {
  global_frame_offset_ = num_frames_;
  start_ = false;
  result_.clear();
  model_->Reset();
  searcher_->Reset();
  ctc_endpointer_->Reset();

  global_ctc_log_probs_.clear();
}

DecodeState AsrDecoder::Decode(bool block) {
  return this->AdvanceDecoding(block);
}

bool AsrDecoder::TextToIds(const std::string& text,
                           const std::vector<WordPiece> word_pieces,
                           std::vector<int>* ids) {
  ids->clear();
  std::vector<std::string> chars;
  SplitUTF8StringToChars(text, &chars);

  int max_valid_id = model_->eos();
  int unk_id = 1;
  const std::string sp_prefix = "\xe2\x96\x81";

  for (size_t i = 0; i < chars.size(); ++i) {
    std::string ch = chars[i];
    if (ch.size() == 1 && ch[0] == ' ') continue;

    if (ch.size() == 1 && ch[0] >= 'a' && ch[0] <= 'z') {
      ch[0] = std::toupper(ch[0]);
    }

    int id = -1;
    bool found = false;

    if (i < word_pieces.size()) {
      const std::string& tpl_token = word_pieces[i].word;

      if (tpl_token.size() >= 4 && tpl_token[0] == '\xe2' &&
          tpl_token[1] == '\x96' && tpl_token[2] == '\x81') {
        std::string prefixed_ch = sp_prefix + ch;
        id = unit_table_->Find(prefixed_ch);
      }
    }

    if (id == -1) {
      id = unit_table_->Find(ch);
    }

    if (id == -1 && oov_mapping_ != nullptr) {
      auto it = oov_mapping_->find(ch);
      if (it != oov_mapping_->end()) {
        std::string proxy_ch = it->second;
        int proxy_id = unit_table_->Find(proxy_ch);

        if (proxy_id != -1) {
          id = proxy_id;
          VLOG(2) << "Proxy Mapping: " << ch << " -> " << proxy_ch
                  << " (ID: " << id << ")";
        } else {
          LOG(WARNING) << "Proxy char " << proxy_ch << " for " << ch
                       << " is also OOV!";
        }
      }
    }

    if (id == -1) {
      LOG(WARNING) << "TextToIds OOV: " << ch << " -> <unk>";
      id = unk_id;
    }

    if (id > max_valid_id) {
      id = unk_id;
    }

    ids->push_back(id);
  }
  return true;
}

float AsrDecoder::CalculateMatchBonus(
    const HotwordCorrection::MatchResult& match) {
  std::vector<std::string> chars;
  wenet::SplitUTF8StringToChars(match.hotword, &chars);
  int char_len = chars.size();
  float length_factor = std::log2(std::max(2, char_len));

  float base_bonus = opts_.bonus_weight * match.score * length_factor;
  float safe_divisor = std::max(opts_.confidence_floor, match.avg_confidence);
  float bonus = base_bonus / safe_divisor;

  return bonus;
}

std::string AsrDecoder::ApplyMatchesToSentence(
    const std::string& original,
    std::vector<HotwordCorrection::MatchResult> matches) {
  if (matches.empty()) return original;

  std::sort(matches.begin(), matches.end(),
            [](const HotwordCorrection::MatchResult& a,
               const HotwordCorrection::MatchResult& b) {
              return a.start < b.start;
            });

  std::string new_sentence;
  size_t current_pos = 0;

  for (const auto& m : matches) {
    if (m.start < current_pos) continue;
    if (m.start > original.length() || m.end > original.length()) continue;

    new_sentence.append(original.substr(current_pos, m.start - current_pos));
    new_sentence.append(m.hotword);
    current_pos = m.end;
  }
  if (current_pos < original.length()) {
    new_sentence.append(original.substr(current_pos));
  }
  return new_sentence;
}

void AsrDecoder::AppendPath() {
  if (corrector_ != nullptr && !result_.empty()) {
    std::unordered_set<std::string> generated_sentences;
    std::vector<DecodeResult> new_candidates;

    for (const auto& path : result_) {
      auto correction = opts_.use_confidence_reward
                            ? corrector_->correct_with_confidence(
                                  path.sentence, path.token_log_probs)
                            : corrector_->correct(path.sentence);

      const auto& all_matches = correction.matchs;
      if (all_matches.empty()) continue;
      for (const auto& m : all_matches) {
        VLOG(2) << "current path: " << path.sentence
                << "\n matches: " << m.hotword << " start: " << m.start
                << " end: " << m.end << " score: " << m.score
                << " avg_conf: " << m.avg_confidence;
      }

      // =========================================================
      // 策略 A: 全量替换
      // =========================================================
      {
        DecodeResult path_full = path;
        path_full.sentence = correction.text;

        float total_bonus = 0.0f;
        path_full.corrected_hotwords.clear();
        for (const auto& m : all_matches) {
          total_bonus += CalculateMatchBonus(m);
          path_full.corrected_hotwords.push_back(m.hotword);
        }
        path_full.score += total_bonus;

        if (generated_sentences.find(path_full.sentence) ==
            generated_sentences.end()) {
          new_candidates.push_back(path_full);
          generated_sentences.insert(path_full.sentence);
        }
      }

      if (all_matches.size() <= 1) continue;

      // =========================================================
      // 策略 B: 高置信度替换
      // =========================================================
      {
        float high_conf_threshold = 0.93f;
        std::vector<HotwordCorrection::MatchResult> robust_matches;
        float bonus = 0.0f;

        for (const auto& m : all_matches) {
          if (m.score >= high_conf_threshold) {
            robust_matches.push_back(m);
            bonus += CalculateMatchBonus(m);
          }
        }

        if (!robust_matches.empty() &&
            robust_matches.size() < all_matches.size()) {
          DecodeResult path_robust = path;
          path_robust.sentence =
              ApplyMatchesToSentence(path.sentence, robust_matches);
          path_robust.score += bonus;
          path_robust.corrected_hotwords.clear();

          for (const auto& m : robust_matches) {
            path_robust.corrected_hotwords.push_back(m.hotword);
          }

          if (generated_sentences.find(path_robust.sentence) ==
              generated_sentences.end()) {
            new_candidates.push_back(path_robust);
            generated_sentences.insert(path_robust.sentence);
          }
        }
      }

      // =========================================================
      // 策略 C: Top-1 替换
      // =========================================================
      {
        auto best_match_iter = std::max_element(
            all_matches.begin(), all_matches.end(),
            [](const auto& a, const auto& b) { return a.score < b.score; });

        if (best_match_iter != all_matches.end()) {
          DecodeResult path_top1 = path;
          path_top1.sentence =
              ApplyMatchesToSentence(path.sentence, {*best_match_iter});
          path_top1.score += CalculateMatchBonus(*best_match_iter);
          path_top1.corrected_hotwords.push_back(best_match_iter->hotword);
          if (generated_sentences.find(path_top1.sentence) ==
              generated_sentences.end()) {
            new_candidates.push_back(path_top1);
            generated_sentences.insert(path_top1.sentence);
          }
        }
      }
    }

    if (!new_candidates.empty()) {
      if (new_candidates.size() > max_append_path) {
        new_candidates.resize(max_append_path);
      }
      result_.insert(result_.end(), new_candidates.begin(),
                     new_candidates.end());
    }
  }
}

DecodeState AsrDecoder::AdvanceDecoding(bool block) {
  DecodeState state = DecodeState::kEndBatch;
  model_->set_chunk_size(opts_.chunk_size);
  model_->set_num_left_chunks(opts_.num_left_chunks);
  int num_required_frames = model_->num_frames_for_chunk(start_);
  std::vector<std::vector<float>> chunk_feats;

  if (!block && !feature_pipeline_->input_finished() &&
      feature_pipeline_->NumQueuedFrames() < num_required_frames) {
    return DecodeState::kWaitFeats;
  }

  if (!feature_pipeline_->Read(num_required_frames, &chunk_feats)) {
    state = DecodeState::kEndFeats;
    VLOG(2) << "switch to state = " << state;
  }

  num_frames_ += chunk_feats.size();
  VLOG(2) << "Required " << num_required_frames << " get "
          << chunk_feats.size();
  Timer timer;

  int start_frame = global_ctc_log_probs_.size();
  chunk_start_indices.push_back(start_frame);

  std::vector<std::vector<float>> ctc_log_probs;
  model_->ForwardEncoder(chunk_feats, &ctc_log_probs);

  global_ctc_log_probs_.insert(global_ctc_log_probs_.end(),
                               ctc_log_probs.begin(), ctc_log_probs.end());

  int forward_time = timer.Elapsed();
  if (opts_.ctc_wfst_search_opts.blank_scale != 1.0) {
    for (int i = 0; i < ctc_log_probs.size(); i++) {
      ctc_log_probs[i][0] = ctc_log_probs[i][0] +
                            std::log(opts_.ctc_wfst_search_opts.blank_scale);
    }
  }
  timer.Reset();
  searcher_->Search(ctc_log_probs);
  int search_time = timer.Elapsed();
  VLOG(3) << "forward takes " << forward_time << " ms, search takes "
          << search_time << " ms";
  UpdateResult();

  if (state != DecodeState::kEndFeats) {
    if (ctc_endpointer_->IsEndpoint(ctc_log_probs, DecodedSomething())) {
      VLOG(1) << "Endpoint is detected at " << num_frames_;
      state = DecodeState::kEndpoint;
    }
  }

  start_ = true;

  VLOG(1) << "current state is " << state;
  return state;
}

void AsrDecoder::UpdateResult(bool finish) {
  const auto& hypotheses = searcher_->Outputs();
  const auto& inputs = searcher_->Inputs();
  const auto& likelihood = searcher_->Likelihood();
  const auto& times = searcher_->Times();
  result_.clear();
  int first_token_padding_flag = false;

  CHECK_EQ(hypotheses.size(), likelihood.size());
  for (size_t i = 0; i < hypotheses.size(); i++) {
    std::string debug_tokens;
    for (int id : hypotheses[i]) {
      debug_tokens += symbol_table_->Find(id) + "(" + std::to_string(id) + ") ";
    }

    const std::vector<int>& hypothesis = hypotheses[i];

    DecodeResult path;
    path.score = likelihood[i];
    int offset = global_frame_offset_ * feature_frame_shift_in_ms();
    for (size_t j = 0; j < hypothesis.size(); j++) {
      std::string word = symbol_table_->Find(hypothesis[j]);

      if (searcher_->Type() == kWfstBeamSearch) {
        path.sentence += (' ' + word);
      } else {
        path.sentence += (word);
      }
    }

    // TimeStamp is only supported in final result
    // TimeStamp of the output of CtcWfstBeamSearch may be inaccurate due to
    // various FST operations when building the decoding graph. So here we use
    // time stamp of the input(e2e model unit), which is more accurate, and it
    // requires the symbol table of the e2e model used in training.
    if (unit_table_ != nullptr && finish) {
      const std::vector<int>& input = inputs[i];
      const std::vector<int>& time_stamp = times[i];
      CHECK_EQ(input.size(), time_stamp.size());

      path.token_log_probs.clear();

      for (size_t j = 0; j < input.size(); j++) {
        std::string word = unit_table_->Find(input[j]);

        int frame_idx = time_stamp[j];
        int token_id = input[j];
        float token_log_prob = 0.0f;

        if (frame_idx >= 0 && frame_idx < global_ctc_log_probs_.size()) {
          int ctc_idx = token_id;
          if (ctc_idx >= 0 &&
              ctc_idx < global_ctc_log_probs_[frame_idx].size()) {
            token_log_prob = global_ctc_log_probs_[frame_idx][ctc_idx];
          }
        }
        path.token_log_probs.push_back(token_log_prob);

        int start = time_stamp[j] * frame_shift_in_ms() - time_stamp_gap_ > 0
                        ? time_stamp[j] * frame_shift_in_ms() - time_stamp_gap_
                        : 0;
        if (j > 0) {
          start = (time_stamp[j] - time_stamp[j - 1]) * frame_shift_in_ms() <
                          time_stamp_gap_
                      ? (time_stamp[j - 1] + time_stamp[j]) / 2 *
                            frame_shift_in_ms()
                      : start;
        }
        int end = time_stamp[j] * frame_shift_in_ms();
        if (j < input.size() - 1) {
          end = (time_stamp[j + 1] - time_stamp[j]) * frame_shift_in_ms() <
                        time_stamp_gap_
                    ? (time_stamp[j + 1] + time_stamp[j]) / 2 *
                          frame_shift_in_ms()
                    : end;
        }
        WordPiece word_piece(word, offset + start, offset + end);
        path.word_pieces.emplace_back(word_piece);
      }
    }

    if (post_processor_ != nullptr) {
      path.sentence = post_processor_->Process(path.sentence, finish);
    }
    result_.emplace_back(path);
  }

  if (hypotheses.size() == 0 && !start_) {
    first_token_padding_flag = true;
  }

  if (!finish) {
    VLOG(1) << "Partial CTC result " << result_[0].sentence;
    if (context_hanzi_graph_ != nullptr) {
      int cur_state = 0;
      float score = 0;
      for (int ilabel : inputs[0]) {
        cur_state = context_hanzi_graph_->GetNextState(
            cur_state, ilabel, &score, &(result_[0].contexts));
      }
      std::string contexts;
      for (const auto& context : result_[0].contexts) {
        contexts += context + ", ";
      }
      VLOG(1) << "Contexts: " << contexts;
    }
  }
}

void AsrDecoder::Rescoring() {
  int num_original_paths;
  int num_paths;
  std::vector<std::vector<int>> hypotheses;
  std::vector<int> valid_indices;
  std::vector<std::vector<int>> original_inputs;

  searcher_->FinalizeSearch();
  UpdateResult(true);

  if (corrector_ != nullptr && hotword_cache_ != nullptr) {
    auto boosts = hotword_cache_->GetActiveHotwordsWithBoost();
    corrector_->update_dynamic_hotwords(boosts);
  }

  num_original_paths = result_.size();
  AppendPath();
  num_paths = result_.size();

  if (0.0 == opts_.rescoring_weight) {
    std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);
    return;
  }

  original_inputs = searcher_->Inputs();
  for (int i = 0; i < (int)original_inputs.size(); i++) {
    hypotheses.push_back(original_inputs[i]);
    valid_indices.push_back(i);
  }

  for (size_t i = num_original_paths; i < num_paths; ++i) {
    std::vector<int> ids;
    bool success = false;
    success = TextToIds(result_[i].sentence, result_[i].word_pieces, &ids);
    if (success) {
      hypotheses.push_back(ids);
      valid_indices.push_back(i);
    } else {
      LOG(WARNING) << "Failed to convert text to ids (should not happen): "
                   << result_[i].sentence;
    }
  }

  if (hypotheses.empty()) return;

  std::vector<float> rescoring_score;
  model_->AttentionRescoring(hypotheses, opts_.reverse_weight,
                             &rescoring_score);

  for (size_t k = 0; k < valid_indices.size(); ++k) {
    int idx = valid_indices[k];
    result_[idx].score = opts_.rescoring_weight * rescoring_score[k] +
                         opts_.ctc_weight * result_[idx].score;
  }

  std::sort(result_.begin(), result_.end(), DecodeResult::CompareFunc);

  if (!result_.empty() && hotword_cache_ != nullptr) {
    const auto& top1_path = result_[0];

    for (const auto& word : top1_path.corrected_hotwords) {
      hotword_cache_->Touch(word);
      VLOG(2) << "HotwordCache Updated (Top1 Hit): " << word;
    }
  }

  for (int i = 0; i < result_.size(); i++) {
    VLOG(2) << "Final Rank " << i << ": " << result_[i].sentence
            << " Score:" << result_[i].score;
  }
}

}  // namespace wenet
