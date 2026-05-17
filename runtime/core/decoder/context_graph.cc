// Copyright (c) 2021 Mobvoi Inc (Zhendong Peng)
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

#include "decoder/context_graph.h"

#include <fstream>
#include <queue>
#include <utility>

#include "fst/determinize.h"

#include "utils/string.h"
#include "utils/utils.h"

namespace wenet {

bool SplitContextToUnits(const std::string& context,
                         const std::shared_ptr<fst::SymbolTable>& unit_table,
                         const std::shared_ptr<std::unordered_map<std::string, std::string>>& oov_mapping, 
                         std::vector<int>* units) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(context, &chars);

  bool no_oov = true;
  bool beginning = true;
  for (size_t start = 0; start < chars.size();) {
    for (size_t end = chars.size(); end > start; --end) {
      std::string unit;
      for (size_t i = start; i < end; i++) {
        unit += chars[i];
      }
      
      if (IsAlpha(unit) && beginning) {
        unit = kSpaceSymbol + unit;
      }

      int unit_id = unit_table->Find(unit);
      if (unit_id != -1) {
        units->emplace_back(unit_id);
        start = end;
        beginning = false;
        continue; 
      }

      if (end == start + 1) {
        if (unit[0] == kSpaceSymbol[0]) {
          units->emplace_back(unit_table->Find(kSpaceSymbol));
          beginning = false;
          break; 
        }

        ++start; 
        
        if (unit == " ") {
          beginning = true;
          continue;
        }

        // ==============================================
        // 新增 oov_mapping 逻辑，用于将 oov 词映射到同音字
        // ==============================================
        bool saved = false;
        if (oov_mapping) {
            auto it = oov_mapping->find(unit);
            if (it != oov_mapping->end()) {
                 std::string proxy = it->second;
                 int proxy_id = unit_table->Find(proxy);
                 if (proxy_id != -1) {
                     units->emplace_back(proxy_id);
                     beginning = false;
                     saved = true;
                 }
            }
        }
        
        if (saved) {
            continue; 
        }

        no_oov = false;
        LOG(WARNING) << unit << " is oov.";
      }
    }
  }
  return no_oov;
}

// Split the UTF-8 string into unit ids according to unit_table
bool SplitContextToUnits(const std::string& context,
                         const std::shared_ptr<fst::SymbolTable>& unit_table,
                         std::vector<int>* units) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(context, &chars);

  bool no_oov = true;
  bool beginning = true;
  for (size_t start = 0; start < chars.size();) {
    for (size_t end = chars.size(); end > start; --end) {
      std::string unit;
      for (size_t i = start; i < end; i++) {
        unit += chars[i];
      }
      // Add '▁' at the beginning of English word.
      // TODO(zhendong.peng): Support bpe model
      if (IsAlpha(unit) && beginning) {
        unit = kSpaceSymbol + unit;
      }

      int unit_id = unit_table->Find(unit);
      if (unit_id != -1) {
        units->emplace_back(unit_id);
        start = end;
        beginning = false;
        continue;
      }

      if (end == start + 1) {
        // Matching using '▁' separately for English
        if (unit[0] == kSpaceSymbol[0]) {
          units->emplace_back(unit_table->Find(kSpaceSymbol));
          beginning = false;
          break;
        }
        ++start;
        if (unit == " ") {
          beginning = true;
          continue;
        }
        no_oov = false;
        LOG(WARNING) << unit << " is oov.";
      }
    }
  }
  return no_oov;
}

ContextGraph::ContextGraph(ContextConfig config) : config_(config) {}


//void PinyinMapper::AddMapping(int hanzi_unit,
//                              const std::vector<std::string>& pys) {
//  std::vector<int> units;
//  for (const auto &py : pys) {
//    int py_unit = unit_table_->Find(py);
//    if (py_unit != fst::kNoSymbol) {
//      units.push_back(py_unit);
//    }
//  }
//  if (!units.empty()) {
//    char2pinyin_[hanzi_unit] = units;
//  }
//}

void PinyinMapper::LoadCharToPinyin(
    const fst::SymbolTable& hanzi_table,
    const fst::SymbolTable& pinyin_table,
    const std::string& dict_path) {

  std::ifstream fin(dict_path);
  CHECK(fin.is_open()) << "Failed to open " << dict_path;

  std::string line;
  while (std::getline(fin, line)) {
    std::istringstream iss(line);
    std::string hanzi;
    iss >> hanzi;

    int hanzi_id = hanzi_table.Find(hanzi);
    if (hanzi_id < 0) continue;

    std::vector<int> pinyin_ids;
    std::string pinyin;
    while (iss >> pinyin) {
      int pid = pinyin_table.Find(pinyin);
      if (pid >= 0) {
        pinyin_ids.push_back(pid);
      }
    }

    if (!pinyin_ids.empty()) {
      char2pinyin_[hanzi_id] = pinyin_ids;
      //LOG(INFO) << char2pinyin_[hanzi_id];
    }
  }
}


bool PinyinMapper::CharToPinyinUnits(int hanzi_unit,
                                     std::vector<int>* pinyin_units) const {
  auto it = char2pinyin_.find(hanzi_unit);
  if (it == char2pinyin_.end()) return false; // 没有映射
  *pinyin_units = it->second;
  return true;
}



int ContextGraph::TraceContext(int cur_state, int unit_id, int* final_state) {
  CHECK_GE(cur_state, 0);
  int next_state = 0;
  Matcher matcher(*graph_, fst::MATCH_INPUT);
  matcher.SetState(cur_state);
  if (matcher.Find(unit_id)) {
    next_state = matcher.Value().nextstate;
    if (graph_->Final(next_state) != Weight::Zero()) {
      *final_state = next_state;
    }
    return next_state;
  }
  LOG(FATAL) << "Trace context failed.";
}
//每一个拼音 arc： 加一条 原拼音 arc 再加 N 条 近音 arc（低权重）支持近音扩展
void ContextGraph::BuildPinyinContextGraph(
    const std::vector<PinyinHotword>& hotwords,
    const std::shared_ptr<fst::SymbolTable>& unit_table, 
    const std::shared_ptr<PinyinMapper>& pinyin_mapper) {

  std::unique_ptr<fst::StdVectorFst> raw_fst(
      new fst::StdVectorFst());

  int start_state = raw_fst->AddState();
  raw_fst->SetStart(start_state);
  float kFuzzyPenalty = -0.5;
  for (const auto& hw : hotwords) {
    int cur_state = start_state;

    for (size_t i = 0; i < hw.pinyins.size(); ++i) {
      const std::string& py = hw.pinyins[i];
      int py_unit = unit_table->Find(py);
      if (py_unit == fst::kNoSymbol) {
        cur_state = -1;
        break;
      }

      int next_state = raw_fst->AddState();

      float base_score =
          hw.score + i * config_.incremental_context_score;

      // LOG(INFO) << "Naive pinyin " << py << " score:" << base_score;
      
      //原始拼音 arc（高权重）
      raw_fst->AddArc(
          cur_state,
          fst::StdArc(py_unit, py_unit, base_score, next_state));

      //近音扩展 arc（低权重）
      auto it = kFuzzyPinyinMap.find(py);
      if (it != kFuzzyPinyinMap.end()) {
        for (const auto& fuzzy_py : it->second) {
          int fuzzy_unit = unit_table->Find(fuzzy_py);
          if (fuzzy_unit == fst::kNoSymbol) continue;

          // LOG(INFO) << "Fuzzy pinyin " << py << " score:" << base_score + kFuzzyPenalty;

          raw_fst->AddArc(
              cur_state,
              fst::StdArc(fuzzy_unit, fuzzy_unit,
                          base_score + kFuzzyPenalty,
                          next_state));
        }
      }

      cur_state = next_state;
    }

    if (cur_state >= 0) {
      raw_fst->SetFinal(cur_state, fst::StdArc::Weight::One());
      context_table_[cur_state] = hw.text;
    }
  }
  pinyin_mapper_ = pinyin_mapper;
  graph_.reset(new fst::StdVectorFst());
  fst::Determinize(*raw_fst, graph_.get());
  ConvertToAC();
}

//原拼音 arc
//void ContextGraph::BuildPinyinContextGraph(
//    const std::vector<PinyinHotword>& hotwords,
//    const fst::SymbolTable& unit_table) {
//
//  //构建原始 FST
//  std::unique_ptr<fst::StdVectorFst> raw_fst(
//      new fst::StdVectorFst());
//
//  int start_state = raw_fst->AddState();
//  raw_fst->SetStart(start_state);
//
//  for (const auto& hw : hotwords) {
//    int cur_state = start_state;
//
//    for (size_t i = 0; i < hw.pinyins.size(); ++i) {
//      const std::string& py = hw.pinyins[i];
//      int py_unit = unit_table.Find(py);
//      if (py_unit == fst::kNoSymbol) {
//        LOG(WARNING) << "Skip hotword, unknown pinyin: " << py;
//        cur_state = -1;
//        break;
//      }
//
//      int next_state = raw_fst->AddState();
//
//      //加分策略：越靠后，权重越大（防止前缀误触发）
//      float arc_score =
//          hw.score + i * config_.incremental_context_score;
//
//      raw_fst->AddArc(
//          cur_state,
//          fst::StdArc(py_unit, py_unit, arc_score, next_state));
//
//      cur_state = next_state;
//    }
//
//    if (cur_state >= 0) {
//      raw_fst->SetFinal(cur_state, fst::StdArc::Weight::One());
//      context_table_[cur_state] = hw.text;  // 记录命中的热词
//    }
//  }
//
//  //确定化（合并公共前缀）
//  graph_.reset(new fst::StdVectorFst());
//  fst::Determinize(*raw_fst, graph_.get());
//
//  //转成 Aho–Corasick 自动机（fallback）
//  ConvertToAC();
//}


void ContextGraph::BuildContextGraph(
    const std::vector<std::string>& contexts,
    const std::shared_ptr<fst::SymbolTable>& unit_table) {
  BuildContextGraph(contexts, /*oov_mapping=*/nullptr, unit_table);
}

void ContextGraph::BuildContextGraph(
    const std::vector<std::string>& contexts,
    const std::shared_ptr<std::unordered_map<std::string, std::string>>& oov_mapping,
    const std::shared_ptr<fst::SymbolTable>& unit_table) {
  // Split context phrase into unit ids according to the `unit_table`
  std::unordered_map<std::string, std::vector<int>> context_units;
  for (const auto& context : contexts) {
    std::vector<int> units;
    bool no_oov = SplitContextToUnits(context, unit_table, oov_mapping, &units);
    if (!no_oov) {
      LOG(WARNING) << "Ignore unknown unit found during compilation.";
      continue;
    }
    context_units[context] = units;
  }

  // Build the context graph
  std::unique_ptr<fst::StdVectorFst> ofst(new fst::StdVectorFst());
  int start_state = ofst->AddState();
  ofst->SetStart(start_state);
  for (const auto& context : contexts) {
    if (context_units.count(context) == 0) continue;
    std::vector<int> units = context_units[context];
    int state = start_state;
    int next_state = state;
    for (size_t i = 0; i < units.size(); ++i) {
      next_state = ofst->AddState();
      if (i == units.size() - 1) {
        ofst->SetFinal(next_state, Weight::One());
      }
      float score =
          i * config_.incremental_context_score + config_.context_score;
      ofst->AddArc(state, fst::StdArc(units[i], units[i], score, next_state));
      state = next_state;
    }
  }
  graph_ = std::unique_ptr<fst::StdVectorFst>(new fst::StdVectorFst());
  // input/output label are sorted after Determinize
  fst::Determinize(*ofst, graph_.get());

  // Determinize will change the final state id
  for (const auto& context : contexts) {
    if (context_units.count(context) == 0) continue;
    std::vector<int> units = context_units[context];
    int final_state = -1;
    int cur_state = 0;
    for (int unit : units) {
      cur_state = TraceContext(cur_state, unit, &final_state);
    }
    CHECK_GT(final_state, 0);
    context_table_[final_state] = context;
  }

  // Convert context graph to AC automaton
  ConvertToAC();
}

void ContextGraph::ConvertToAC() {
  CHECK(graph_ != nullptr) << "Context graph should not be nullptr!";
  int num_states = graph_->NumStates();
  std::vector<int> fail_states(num_states, 0);
  std::vector<float> total_weights(num_states, 0);
  Matcher matcher(*graph_, fst::MATCH_INPUT);
  // start state
  fail_states[0] = -1;
  total_weights[0] = 0;

  // Please see:
  // https://web.stanford.edu/group/cslipublications/cslipublications/koskenniemi-festschrift/9-mohri.pdf
  std::queue<int> states_queue;
  states_queue.push(0);
  while (!states_queue.empty()) {
    int state = states_queue.front();
    states_queue.pop();

    for (ArcIterator aiter(*graph_, state); !aiter.Done(); aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      int next_state = arc.nextstate;
      total_weights[next_state] = total_weights[state] + arc.weight.Value();
      // Backtracking the failure state for next_state
      for (int fail_state = fail_states[state]; fail_state != -1;
           fail_state = fail_states[fail_state]) {
        matcher.SetState(fail_state);
        if (matcher.Find(arc.ilabel)) {
          fail_states[next_state] = matcher.Value().nextstate;
          break;
        }
      }
      states_queue.push(next_state);
    }
  }

  // Compute fail weight, add fail arc
  for (int state = 0; state < num_states; state++) {
    int fail_state = fail_states[state];
    if (fail_state < 0) continue;
    if (graph_->Final(fail_state) != Weight::Zero()) {
      fallback_finals_[state] = fail_state;
      if (graph_->NumArcs(fail_state) == 0) continue;
    }
    if (graph_->Final(state) != Weight::Zero() && fail_state == 0) continue;

    float fail_weight = total_weights[fail_state] - total_weights[state];
    if (graph_->Final(state) != Weight::Zero()) {
      fail_weight = 0;
    }
    graph_->AddArc(state, fst::StdArc(0, 0, fail_weight, fail_state));
  }
  // Sort arcs by ilabel, means move the fallback arc from last to first for the
  // matcher
  fst::ArcSort(graph_.get(), fst::ILabelCompare<fst::StdArc>());
}

// int ContextGraph::GetNextState(
//     int cur_state,
//     int unit_id,           // 汉字 token id
//     float* score,
//     std::unordered_set<std::string>* contexts) {

//   CHECK_GE(cur_state, 0);
//   CHECK_NE(unit_id, 0);

//   Matcher matcher(*graph_, fst::MATCH_INPUT);
//   matcher.SetState(cur_state);
//   // pinyin_mapper_ = std::make_unique<PinyinMapper>(); 
//   // ---------- 1) 直接拼音精确匹配 -------------
//   std::vector<int> pinyin_units;
//   bool has_pinyin = pinyin_mapper_->CharToPinyinUnits(unit_id, &pinyin_units);

//   if (has_pinyin) {
//     // 对每个可能的拼音版本去匹配 FST
//     for (int py_unit : pinyin_units) {
//       matcher.SetState(cur_state);
//       if (matcher.Find(py_unit)) {
//         const fst::StdArc& arc = matcher.Value();
//         int next_state = arc.nextstate;

//         // prefix reward（小）
//         *score += arc.weight.Value();

//         // final？
//         if (contexts && graph_->Final(next_state) != Weight::Zero()) {
//           contexts->insert(context_table_[next_state]);
//         }

//         // 到了叶子 reset
//         if (graph_->NumArcs(next_state) == 0) {
//           return 0;
//         }
//         return next_state;
//       }
//     }
//   }

//   // -------- 2) fallback（不加分） -------------
//   ArcIterator aiter(*graph_, cur_state);
//   const fst::StdArc& fallback_arc = aiter.Value();
//   if (fallback_arc.ilabel == 0) {
//     int next_state = fallback_arc.nextstate;
//     return GetNextState(next_state, unit_id, score, contexts);
//   }

//   return 0;
// }

int ContextGraph::GetNextState(
    int cur_state,
    int unit_id,           // 汉字 token id
    float* score,
    int* current_match_length,
    std::unordered_set<std::string>* contexts) {

  CHECK_GE(cur_state, 0);
  CHECK_NE(unit_id, 0);

  Matcher matcher(*graph_, fst::MATCH_INPUT);
  matcher.SetState(cur_state);
  // pinyin_mapper_ = std::make_unique<PinyinMapper>(); 
  // ---------- 1) 直接拼音精确匹配 -------------
  std::vector<int> pinyin_units;
  bool has_pinyin = pinyin_mapper_->CharToPinyinUnits(unit_id, &pinyin_units);

  if (has_pinyin) {
    // 对每个可能的拼音版本去匹配 FST
    for (int py_unit : pinyin_units) {
      matcher.SetState(cur_state);
      if (matcher.Find(py_unit)) {
        const fst::StdArc& arc = matcher.Value();
        int next_state = arc.nextstate;
        // prefix reward（小）
        *score += (arc.weight.Value() * (*current_match_length));
        *current_match_length += 1;
        
        // final？
        if (contexts && graph_->Final(next_state) != Weight::Zero()) {
          contexts->insert(context_table_[next_state]);
          *current_match_length = 0;
        }

        // 到了叶子 reset
        if (graph_->NumArcs(next_state) == 0) {
          *current_match_length = 0;
          return 0;
        }
        return next_state;
      }
    }
  }

  // -------- 2) fallback（不加分） -------------
  ArcIterator aiter(*graph_, cur_state);
  const fst::StdArc& fallback_arc = aiter.Value();
  if (fallback_arc.ilabel == 0) {
    int next_state = fallback_arc.nextstate;
    *current_match_length = 0;
    return GetNextState(next_state, unit_id, score, current_match_length,contexts);
  }

  return 0;
}



int ContextGraph::GetNextState(int cur_state, int unit_id, float* score,
                              std::unordered_set<std::string>* contexts) {
 CHECK_GE(cur_state, 0);
 // Find(0) matches any epsilons on the underlying FST explicitly
 CHECK_NE(unit_id, 0);
 int next_state = 0;

 Matcher matcher(*graph_, fst::MATCH_INPUT);
 matcher.SetState(cur_state);
 if (matcher.Find(unit_id)) {
   const fst::StdArc& arc = matcher.Value();
   next_state = arc.nextstate;
   *score += arc.weight.Value();
   // Collect all contexts in the decode result
   if (contexts != nullptr) {
     if (graph_->Final(next_state) != Weight::Zero()) {
       contexts->insert(context_table_[next_state]);
     }
     int fallback_final = next_state;
     while (fallback_finals_.count(fallback_final) > 0) {
       fallback_final = fallback_finals_[fallback_final];
       contexts->insert(context_table_[fallback_final]);
     }
   }

   // Leaves go back to the start state
   if (graph_->NumArcs(next_state) == 0) {
     return 0;
   }
   return next_state;
 }

 // Check whether the first arc is fallback arc
 ArcIterator aiter(*graph_, cur_state);
 const fst::StdArc& arc = aiter.Value();
 // The start state has no fallback arc
 if (arc.ilabel == 0) {
   next_state = arc.nextstate;
   *score += arc.weight.Value();
   // fallback
   return GetNextState(next_state, unit_id, score);
 }

 return 0;
}

//int ContextGraph::GetNextState(int cur_state, int unit_id, float* score,float* r_score,
//                               std::unordered_set<std::string>* contexts) {
//  CHECK_GE(cur_state, 0);
//  // Find(0) matches any epsilons on the underlying FST explicitly
//  CHECK_NE(unit_id, 0);
//  int next_state = 0;
//
//  Matcher matcher(*graph_, fst::MATCH_INPUT);
//  matcher.SetState(cur_state);
//  if (matcher.Find(unit_id)) {
//    const fst::StdArc& arc = matcher.Value();
//    next_state = arc.nextstate;
//    *score += arc.weight.Value();
//    *r_score += arc.weight.Value();
//    VLOG(1) << "context_graph: "  << *score << "arc.weight.Value: " <<arc.weight.Value();
//    // Collect all contexts in the decode result
//    if (contexts != nullptr) {
//      if (graph_->Final(next_state) != Weight::Zero()) {
//        contexts->insert(context_table_[next_state]);
//      }
//      int fallback_final = next_state;
//      while (fallback_finals_.count(fallback_final) > 0) {
//        fallback_final = fallback_finals_[fallback_final];
//        contexts->insert(context_table_[fallback_final]);
//      }
//    }
//
//    // Leaves go back to the start state
//    if (graph_->NumArcs(next_state) == 0) {
//      return 0;
//    }
//    return next_state;
//  }
//
//  // Check whether the first arc is fallback arc
//  ArcIterator aiter(*graph_, cur_state);
//  const fst::StdArc& arc = aiter.Value();
//  // The start state has no fallback arc
//  if (arc.ilabel == 0) {
//    next_state = arc.nextstate;
//    *score += arc.weight.Value();
//    *r_score += arc.weight.Value();;
//    // fallback
//    return GetNextState(next_state, unit_id, score);
//  }
//
//  return 0;
//}
//new op ok
//int ContextGraph::GetNextState(
//    int cur_state,
//    int unit_id,
//    float* score,
//    float* r_score,
//    std::unordered_set<std::string>* contexts) {
//
//  CHECK_GE(cur_state, 0);
//  CHECK_NE(unit_id, 0);  // 0 is epsilon / blank
//
//  *score = 0.0f;
//  *r_score = 0.0f;
//
//  int state = cur_state;
//  int next_state = 0;
//
//  Matcher matcher(*graph_, fst::MATCH_INPUT);
//
//  // -------- 1. 尝试在当前状态匹配 --------
//  matcher.SetState(state);
//  if (matcher.Find(unit_id)) {
//    const fst::StdArc& arc = matcher.Value();
//    next_state = arc.nextstate;
//
//    // prefix reward（安全：arc.weight 只代表 prefix）
//    float delta = arc.weight.Value();
//    if (delta > 0.0f) {
//      *score += delta;
//      *r_score += delta;
//    }
//
//    // final reward：只在 final state 加
//    if (contexts != nullptr &&
//        graph_->Final(next_state) != Weight::Zero()) {
//      contexts->insert(context_table_[next_state]);
//    }
//
//    // 叶子节点：命中即结束，回 root，不允许刷分
//    if (graph_->NumArcs(next_state) == 0) {
//      return 0;
//    }
//
//    return next_state;
//  }
//
//  // -------- 2. fallback 查找（不加分） --------
//  // 沿 fallback 链回退，直到 root
//  while (state != 0) {
//    ArcIterator aiter(*graph_, state);
//    const fst::StdArc& arc = aiter.Value();
//
//    // 非 fallback arc，停止
//    if (arc.ilabel != 0) {
//      break;
//    }
//
//    // fallback，不加分
//    state = arc.nextstate;
//
//    matcher.SetState(state);
//    if (matcher.Find(unit_id)) {
//      const fst::StdArc& match_arc = matcher.Value();
//      next_state = match_arc.nextstate;
//
//      float delta = match_arc.weight.Value();
//      if (delta > 0.0f) {
//        *score += delta;
//        *r_score += delta;
//      }
//
//      if (contexts != nullptr &&
//          graph_->Final(next_state) != Weight::Zero()) {
//        contexts->insert(context_table_[next_state]);
//      }
//
//      if (graph_->NumArcs(next_state) == 0) {
//        return 0;
//      }
//      return next_state;
//    }
//  }
//
//  // -------- 3. 完全失败，回 root --------
//  return 0;
//}

}  // namespace wenet
