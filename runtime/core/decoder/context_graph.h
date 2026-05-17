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

#ifndef DECODER_CONTEXT_GRAPH_H_
#define DECODER_CONTEXT_GRAPH_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fst/compose.h"
#include "fst/fst.h"
#include "fst/matcher.h"
#include "fst/vector-fst.h"

namespace wenet {

using ArcIterator = fst::ArcIterator<fst::StdFst>;
using Matcher = fst::SortedMatcher<fst::StdFst>;
using Weight = fst::StdArc::Weight;

bool SplitContextToUnits(const std::string& context,
                         const std::shared_ptr<fst::SymbolTable>& unit_table,
                         std::vector<int>* units);

struct ContextConfig {
  int max_contexts = 5000;
  int max_context_length = 100;
  float context_score = 3.0;
  float incremental_context_score = 0.0;
};

//近音拼音表
static const std::unordered_map<std::string,
                                std::vector<std::string>> kFuzzyPinyinMap = {
  {"zhi", {"zi", "shi"}},
  {"zi",  {"zhi"}},
  {"shi", {"zhi"}},

  {"zhang", {"zang"}},
  {"zang", {"zhang"}},

  {"cheng", {"ceng"}},
  {"ceng", {"cheng"}},

  {"shang", {"sang"}},
  {"sang", {"shang"}},

  {"ren", {"len"}},
  {"len", {"ren"}},

  {"fan", {"han"}},
  {"han", {"fan"}},
  {"ye", {"mian"}},
  {"tu", {"fu"}},

  // 可控扩展，不要太多
};

struct PinyinHotword {
  std::string text;        // 汉字（可选，仅用于记录）
  std::vector<std::string> pinyins;  // {"zhi","fu","bao"}
  float score;             // 热词权重
};

class PinyinMapper {
 public:
  // 构造函数：加载拼音映射表
  //explicit PinyinMapper(const std::shared_ptr<fst::SymbolTable>& unit_table) 
  //    : unit_table_(unit_table) {}
  PinyinMapper() = default;
   // 从 hanzi_pinyin.txt 构建映射
  void LoadCharToPinyin(
      const fst::SymbolTable& hanzi_table,
      const fst::SymbolTable& pinyin_table,
      const std::string& dict_path);
  //汉字 unit_id → 拼音 unit_id(s)
  bool CharToPinyinUnits(int hanzi_unit,
                         std::vector<int>* pinyin_units) const;
  // 加载映射表
  void AddMapping(int hanzi_unit, const std::vector<std::string>& pys);

 private:
  //std::shared_ptr<fst::SymbolTable> unit_table_;
  // 核心映射表
  // key: hanzi unit_id
  // val: pinyin unit_id list (多音字)
  std::unordered_map<int, std::vector<int>> char2pinyin_; // hanzi_unit -> 拼音 unit_id
};

class ContextGraph {
 public:
  explicit ContextGraph(ContextConfig config);
  int TraceContext(int cur_state, int unit_id, int* final_state);
  void BuildContextGraph(const std::vector<std::string>& context,
                         const std::shared_ptr<std::unordered_map<std::string, std::string>>& oov_mapping,
                         const std::shared_ptr<fst::SymbolTable>& unit_table);

  void BuildContextGraph(const std::vector<std::string>& context,
                         const std::shared_ptr<fst::SymbolTable>& unit_table);
  //void BuildPinyinContextGraph(
  //  const std::vector<PinyinHotword>& hotwords,
  //  const fst::SymbolTable& unit_table);
  
  void BuildPinyinContextGraph(
    const std::vector<PinyinHotword>& hotwords,
    const std::shared_ptr<fst::SymbolTable>& unit_table,
    const std::shared_ptr<PinyinMapper>& pinyin_mapper);
  void ConvertToAC();
  int GetNextState(int cur_state, int unit_id, float* score,
                   std::unordered_set<std::string>* contexts = nullptr);
  int GetNextState(int cur_state, int unit_id, float* score, int* current_match_length,
                   std::unordered_set<std::string>* contexts = nullptr);
  int GetNextState(int cur_state, int unit_id, float* score,float* r_score,
                   std::unordered_set<std::string>* contexts = nullptr);
  // check context state is the final state
  bool IsFinalState(int state) {
    return graph_->Final(state) != Weight::Zero();
  }

 private:
  ContextConfig config_;
  std::shared_ptr<PinyinMapper> pinyin_mapper_ = nullptr;
  std::unique_ptr<fst::StdVectorFst> graph_;
  std::unordered_map<int, int> fallback_finals_;  // States fallback to final
  std::unordered_map<int, std::string> context_table_;  // Finals to context
};

}  // namespace wenet

#endif  // DECODER_CONTEXT_GRAPH_H_
