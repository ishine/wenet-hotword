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

// Splits a UTF-8 context string into unit_table ids. When oov_mapping is
// non-null, characters missing from unit_table are routed through their
// homophone proxies before being declared OOV. Returns true iff every
// character mapped successfully.
bool SplitContextToUnits(
    const std::string& context,
    const std::shared_ptr<fst::SymbolTable>& unit_table,
    std::vector<int>* units,
    const std::shared_ptr<std::unordered_map<std::string, std::string>>&
        oov_mapping = nullptr);

struct ContextConfig {
  int max_contexts = 5000;
  int max_context_length = 100;
  float context_score = 3.0;
  float incremental_context_score = 0.0;
};

// Aho-Corasick context-biasing FST over the acoustic-model unit table.
// Built once from a list of hotword strings; queried per CTC search step
// via GetNextState to accumulate arc weights along matched prefixes.
class ContextGraph {
 public:
  explicit ContextGraph(ContextConfig config);

  // Upstream-compatible builder: hotword strings, unit table only.
  void BuildContextGraph(const std::vector<std::string>& contexts,
                         const std::shared_ptr<fst::SymbolTable>& unit_table);
  // Same as above plus an OOV → homophone-proxy mapping for hotwords whose
  // characters fall outside unit_table.
  void BuildContextGraph(
      const std::vector<std::string>& contexts,
      const std::shared_ptr<std::unordered_map<std::string, std::string>>&
          oov_mapping,
      const std::shared_ptr<fst::SymbolTable>& unit_table);

  int TraceContext(int cur_state, int unit_id, int* final_state);
  void ConvertToAC();

  int GetNextState(int cur_state, int unit_id, float* score,
                   std::unordered_set<std::string>* contexts = nullptr);

  bool IsFinalState(int state) {
    return graph_->Final(state) != Weight::Zero();
  }

 private:
  ContextConfig config_;
  std::unique_ptr<fst::StdVectorFst> graph_;
  std::unordered_map<int, int> fallback_finals_;        // States fallback to final
  std::unordered_map<int, std::string> context_table_;  // Finals to context
};

}  // namespace wenet

#endif  // DECODER_CONTEXT_GRAPH_H_
