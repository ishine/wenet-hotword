// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
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

#include <iomanip>
#include <thread>
#include <utility>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/json.h"
#include "utils/string.h"
#include "utils/thread_pool.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_bool(output_nbest, false, "output n-best of decode result");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(result, "", "result output file");
DEFINE_bool(continuous_decoding, false, "continuous decoding mode");
DEFINE_int32(thread_num, 1, "num of decode thread");
DEFINE_int32(warmup, 0, "num of warmup decode, 0 means no warmup");
DEFINE_bool(daemon, false,
            "Run in daemon mode: load model/resources once, then read JSON "
            "trial configs from stdin and write JSON results to stdout. "
            "Send 'EXIT' to stop.");

std::shared_ptr<wenet::DecodeOptions> g_decode_config;
std::shared_ptr<wenet::FeaturePipelineConfig> g_feature_config;
std::shared_ptr<wenet::DecodeResource> g_decode_resource;

std::ofstream g_result;
std::mutex g_mutex;
int g_total_waves_dur = 0;
int g_total_decode_time = 0;

void Decode(std::pair<std::string, std::string> wav, bool warmup = false) {
  wenet::WavReader wav_reader(wav.second);
  int num_samples = wav_reader.num_samples();
  CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);

  auto feature_pipeline =
      std::make_shared<wenet::FeaturePipeline>(*g_feature_config);
  feature_pipeline->AcceptWaveform(wav_reader.data(), num_samples);
  feature_pipeline->set_input_finished();
  LOG(INFO) << "num frames " << feature_pipeline->num_frames();

  wenet::AsrDecoder decoder(feature_pipeline, g_decode_resource,
                            *g_decode_config);

  int wave_dur = static_cast<int>(static_cast<float>(num_samples) /
                                  wav_reader.sample_rate() * 1000);
  int decode_time = 0;
  std::string final_result;
  while (true) {
    wenet::Timer timer;
    wenet::DecodeState state = decoder.Decode();
    if (state == wenet::DecodeState::kEndFeats) {
      decoder.Rescoring();
    }
    int chunk_decode_time = timer.Elapsed();
    decode_time += chunk_decode_time;
    if (decoder.DecodedSomething()) {
      LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
    }

    if (FLAGS_continuous_decoding && state == wenet::DecodeState::kEndpoint) {
      if (decoder.DecodedSomething()) {
        decoder.Rescoring();
        LOG(INFO) << "Final result (continuous decoding): "
                  << decoder.result()[0].sentence;
        final_result.append(decoder.result()[0].sentence);
      }
      decoder.ResetContinuousDecoding();
    }

    if (state == wenet::DecodeState::kEndFeats) {
      break;
    } else if (FLAGS_chunk_size > 0 && FLAGS_simulate_streaming) {
      float frame_shift_in_ms =
          static_cast<float>(g_feature_config->frame_shift) /
          wav_reader.sample_rate() * 1000;
      auto wait_time =
          decoder.num_frames_in_current_chunk() * frame_shift_in_ms -
          chunk_decode_time;
      if (wait_time > 0) {
        LOG(INFO) << "Simulate streaming, waiting for " << wait_time << "ms";
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(wait_time)));
      }
    }
  }
  if (decoder.DecodedSomething()) {
    final_result.append(decoder.result()[0].sentence);
  }
  LOG(INFO) << wav.first << " Final result: " << final_result << std::endl;
  LOG(INFO) << "Decoded " << wave_dur << "ms audio taken " << decode_time
            << "ms.";

  if (!warmup) {
    g_mutex.lock();
    std::ostream& buffer = FLAGS_result.empty() ? std::cout : g_result;
    if (!FLAGS_output_nbest) {
      buffer << wav.first << " " << final_result << std::endl;
    } else {
      buffer << "wav " << wav.first << std::endl;
      auto& results = decoder.result();
      for (auto& r : results) {
        if (r.sentence.empty()) continue;
        buffer << "candidate " << r.score << " " << r.sentence << std::endl;
      }
    }
    g_total_waves_dur += wave_dur;
    g_total_decode_time += decode_time;
    g_mutex.unlock();
  }
}

// ===================== Daemon helpers =====================

struct DaemonFlagDefaults {
  double ctc_weight, rescoring_weight, reverse_weight, length_penalty;
  double bonus_weight, confidence_floor, neighbor_threshold;
  double fuzzy_threshold, fuzzy_threshold_en;
  int nbest, max_append_path, chunk_size, num_left_chunks;
  bool use_confidence_reward, enable_hotword_cache;
  std::string hotword_path, confusion_matrix_path, result, wav_scp;
};

static DaemonFlagDefaults g_daemon_defaults;

static void SaveDaemonDefaults() {
  g_daemon_defaults.ctc_weight = FLAGS_ctc_weight;
  g_daemon_defaults.rescoring_weight = FLAGS_rescoring_weight;
  g_daemon_defaults.reverse_weight = FLAGS_reverse_weight;
  g_daemon_defaults.length_penalty = FLAGS_length_penalty;
  g_daemon_defaults.bonus_weight = FLAGS_bonus_weight;
  g_daemon_defaults.confidence_floor = FLAGS_confidence_floor;
  g_daemon_defaults.neighbor_threshold = FLAGS_neighbor_threshold;
  g_daemon_defaults.fuzzy_threshold = FLAGS_fuzzy_threshold;
  g_daemon_defaults.fuzzy_threshold_en = FLAGS_fuzzy_threshold_en;
  g_daemon_defaults.nbest = FLAGS_nbest;
  g_daemon_defaults.max_append_path = FLAGS_max_append_path;
  g_daemon_defaults.chunk_size = FLAGS_chunk_size;
  g_daemon_defaults.num_left_chunks = FLAGS_num_left_chunks;
  g_daemon_defaults.use_confidence_reward = FLAGS_use_confidence_reward;
  g_daemon_defaults.enable_hotword_cache = FLAGS_enable_hotword_cache;
  g_daemon_defaults.hotword_path = FLAGS_hotword_path;
  g_daemon_defaults.confusion_matrix_path = FLAGS_confusion_matrix_path;
  g_daemon_defaults.result = FLAGS_result;
  g_daemon_defaults.wav_scp = FLAGS_wav_scp;
}

static void ResetDaemonFlags() {
  FLAGS_ctc_weight = g_daemon_defaults.ctc_weight;
  FLAGS_rescoring_weight = g_daemon_defaults.rescoring_weight;
  FLAGS_reverse_weight = g_daemon_defaults.reverse_weight;
  FLAGS_length_penalty = g_daemon_defaults.length_penalty;
  FLAGS_bonus_weight = g_daemon_defaults.bonus_weight;
  FLAGS_confidence_floor = g_daemon_defaults.confidence_floor;
  FLAGS_neighbor_threshold = g_daemon_defaults.neighbor_threshold;
  FLAGS_fuzzy_threshold = g_daemon_defaults.fuzzy_threshold;
  FLAGS_fuzzy_threshold_en = g_daemon_defaults.fuzzy_threshold_en;
  FLAGS_nbest = g_daemon_defaults.nbest;
  FLAGS_max_append_path = g_daemon_defaults.max_append_path;
  FLAGS_chunk_size = g_daemon_defaults.chunk_size;
  FLAGS_num_left_chunks = g_daemon_defaults.num_left_chunks;
  FLAGS_use_confidence_reward = g_daemon_defaults.use_confidence_reward;
  FLAGS_enable_hotword_cache = g_daemon_defaults.enable_hotword_cache;
  FLAGS_hotword_path = g_daemon_defaults.hotword_path;
  FLAGS_confusion_matrix_path = g_daemon_defaults.confusion_matrix_path;
  FLAGS_result = g_daemon_defaults.result;
  FLAGS_wav_scp = g_daemon_defaults.wav_scp;
}

static void ApplyDaemonParams(const json::JSON& params) {
  if (params.hasKey("ctc_weight"))
    FLAGS_ctc_weight = params.at("ctc_weight").ToFloat();
  if (params.hasKey("rescoring_weight"))
    FLAGS_rescoring_weight = params.at("rescoring_weight").ToFloat();
  if (params.hasKey("reverse_weight"))
    FLAGS_reverse_weight = params.at("reverse_weight").ToFloat();
  if (params.hasKey("length_penalty"))
    FLAGS_length_penalty = params.at("length_penalty").ToFloat();
  if (params.hasKey("nbest"))
    FLAGS_nbest = static_cast<int>(params.at("nbest").ToInt());
  if (params.hasKey("chunk_size"))
    FLAGS_chunk_size = static_cast<int>(params.at("chunk_size").ToInt());
  if (params.hasKey("num_left_chunks"))
    FLAGS_num_left_chunks =
        static_cast<int>(params.at("num_left_chunks").ToInt());
  if (params.hasKey("fuzzy_threshold"))
    FLAGS_fuzzy_threshold = params.at("fuzzy_threshold").ToFloat();
  if (params.hasKey("fuzzy_threshold_en"))
    FLAGS_fuzzy_threshold_en = params.at("fuzzy_threshold_en").ToFloat();
  if (params.hasKey("max_append_path"))
    FLAGS_max_append_path =
        static_cast<int>(params.at("max_append_path").ToInt());
  if (params.hasKey("use_confidence_reward"))
    FLAGS_use_confidence_reward =
        params.at("use_confidence_reward").ToBool();
  if (params.hasKey("bonus_weight"))
    FLAGS_bonus_weight = params.at("bonus_weight").ToFloat();
  if (params.hasKey("confidence_floor"))
    FLAGS_confidence_floor = params.at("confidence_floor").ToFloat();
  if (params.hasKey("neighbor_threshold"))
    FLAGS_neighbor_threshold = params.at("neighbor_threshold").ToFloat();
  if (params.hasKey("hotword_path"))
    FLAGS_hotword_path = params.at("hotword_path").ToString();
  if (params.hasKey("confusion_matrix_path"))
    FLAGS_confusion_matrix_path =
        params.at("confusion_matrix_path").ToString();
  if (params.hasKey("enable_hotword_cache"))
    FLAGS_enable_hotword_cache =
        params.at("enable_hotword_cache").ToBool();
}

static std::shared_ptr<wenet::DecodeResource> BuildTrialResource(
    std::shared_ptr<wenet::DecodeResource> base) {
  auto resource = std::make_shared<wenet::DecodeResource>();
  // Heavy resources: reuse from base
  resource->model = base->model;
  resource->unit_table = base->unit_table;
  resource->symbol_table = base->symbol_table;
  resource->fst = base->fst;
  resource->oov_mapping = base->oov_mapping;
  resource->context_hanzi_graph = base->context_hanzi_graph;

  // Variable resources: rebuild from current flags
  resource->max_append_path = FLAGS_max_append_path;
  if (FLAGS_enable_hotword_cache) {
    resource->hotword_cache = std::make_shared<wenet::HotwordCache>(20, 2);
  }

  if (!FLAGS_hotword_path.empty() && !FLAGS_pinyin_dict_path.empty()) {
    HotwordCorrection::PinyinProvider::initialize(FLAGS_pinyin_dict_path);
    HotwordCorrection::SetNeighborThreshold(FLAGS_neighbor_threshold);
    HotwordCorrection::LoadConfusionMatrix(FLAGS_confusion_matrix_path);
    if (!FLAGS_cmu_dict_path.empty()) {
      HotwordCorrection::EnglishProvider::initialize(FLAGS_cmu_dict_path);
    }
    auto corrector = std::make_shared<HotwordCorrection::PhonemeCorrector>(
        FLAGS_fuzzy_threshold, FLAGS_fuzzy_threshold_en);
    std::ifstream hw_file(FLAGS_hotword_path);
    if (hw_file.is_open()) {
      std::stringstream buffer;
      buffer << hw_file.rdbuf();
      corrector->update_hotwords(buffer.str());
    }
    resource->corrector = corrector;
  }

  resource->post_processor = base->post_processor;
  return resource;
}

static std::string CompactJson(const std::string& json_str) {
  std::string out;
  out.reserve(json_str.size());
  bool in_string = false;
  for (char c : json_str) {
    if (c == '\"') {
      in_string = !in_string;
      out += c;
    } else if (in_string || !std::isspace(static_cast<unsigned char>(c))) {
      out += c;
    }
  }
  return out;
}

static std::vector<std::pair<std::string, std::string>> LoadWavScp(
    const std::string& path) {
  std::vector<std::pair<std::string, std::string>> waves;
  std::ifstream wav_scp(path);
  std::string line;
  while (getline(wav_scp, line)) {
    std::vector<std::string> strs;
    wenet::SplitString(line, &strs);
    if (strs.size() >= 2) {
      waves.emplace_back(strs[0], strs[1]);
    }
  }
  return waves;
}

static void RunDaemon() {
  g_feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto base_resource = wenet::InitDecodeResourceFromFlags();
  SaveDaemonDefaults();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "EXIT") break;

    auto j = json::JSON::Load(line);
    if (!j.hasKey("wav_scp")) {
      json::JSON err;
      err["status"] = "error";
      err["message"] = "missing wav_scp";
      std::cout << CompactJson(err.dump()) << std::endl;
      continue;
    }

    // Reset flags to defaults, then apply trial params
    ResetDaemonFlags();
    if (j.hasKey("params")) ApplyDaemonParams(j["params"]);
    if (j.hasKey("result")) FLAGS_result = j["result"].ToString();
    FLAGS_wav_scp = j["wav_scp"].ToString();

    // Rebuild configs/resources for this trial
    g_decode_config = wenet::InitDecodeOptionsFromFlags();
    g_decode_resource = BuildTrialResource(base_resource);

    // Load waves
    auto waves = LoadWavScp(FLAGS_wav_scp);
    if (waves.empty()) {
      json::JSON err;
      err["status"] = "error";
      err["message"] = "empty wav_scp: " + FLAGS_wav_scp;
      std::cout << CompactJson(err.dump()) << std::endl;
      continue;
    }

    if (!FLAGS_result.empty()) {
      g_result.open(FLAGS_result, std::ios::out);
    }

    g_total_waves_dur = 0;
    g_total_decode_time = 0;

    {
      ThreadPool pool(FLAGS_thread_num);
      for (auto& wav : waves) {
        pool.enqueue(Decode, wav, false);
      }
    }

    if (g_result.is_open()) g_result.close();

    json::JSON out;
    out["status"] = "ok";
    out["total_waves_dur"] = g_total_waves_dur;
    out["total_decode_time"] = g_total_decode_time;
    if (g_total_waves_dur > 0) {
      out["rtf"] = static_cast<float>(g_total_decode_time) /
                    static_cast<float>(g_total_waves_dur);
    }
    std::cout << CompactJson(out.dump()) << std::endl;
  }

  LOG(INFO) << "Daemon exiting.";
}

// ===================== End daemon helpers =====================

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_daemon) {
    RunDaemon();
    return 0;
  }

  g_decode_config = wenet::InitDecodeOptionsFromFlags();
  g_feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  g_decode_resource = wenet::InitDecodeResourceFromFlags();

  if (FLAGS_wav_path.empty() && FLAGS_wav_scp.empty()) {
    LOG(FATAL) << "Please provide the wave path or the wav scp.";
  }
  std::vector<std::pair<std::string, std::string>> waves;
  if (!FLAGS_wav_path.empty()) {
    waves.emplace_back(make_pair("test", FLAGS_wav_path));
  } else {
    std::ifstream wav_scp(FLAGS_wav_scp);
    std::string line;
    while (getline(wav_scp, line)) {
      std::vector<std::string> strs;
      wenet::SplitString(line, &strs);
      CHECK_GE(strs.size(), 2);
      waves.emplace_back(make_pair(strs[0], strs[1]));
    }

    if (waves.empty()) {
      LOG(FATAL) << "Please provide non-empty wav scp.";
    }
  }

  if (!FLAGS_result.empty()) {
    g_result.open(FLAGS_result, std::ios::out);
  }

  // Warmup
  if (FLAGS_warmup > 0) {
    LOG(INFO) << "Warming up...";
    {
      ThreadPool pool(FLAGS_thread_num);
      auto wav = waves[0];
      for (int i = 0; i < FLAGS_warmup; i++) {
        pool.enqueue(Decode, wav, true);
      }
    }
    LOG(INFO) << "Warmup done.";
  }

  {
    ThreadPool pool(FLAGS_thread_num);
    for (auto& wav : waves) {
      pool.enqueue(Decode, wav, false);
    }
  }

  LOG(INFO) << "Total: decoded " << g_total_waves_dur << "ms audio taken "
            << g_total_decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(g_total_decode_time) / g_total_waves_dur;
  return 0;
}
