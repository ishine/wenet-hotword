#ifndef DECODER_CORRECTOR_H_
#define DECODER_CORRECTOR_H_

#include <string>
#include <vector>
#include <set>
#include <memory>
#include <mutex>
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cpp-pinyin/Pinyin.h>
#include <cpp-pinyin/G2pglobal.h>

namespace HotwordCorrection {

// Built-in sparse fallback. Used when no CSV is loaded via LoadConfusionMatrix.
extern const std::map<std::pair<std::string, std::string>, float> CONFUSION_MATRIX;

// Load a dense confusion matrix from CSV. Empty/missing path leaves the
// sparse fallback active. Format: `from,to,cost` per line; '#' starts a
// comment; both directions must be listed explicitly if asymmetric.
void LoadConfusionMatrix(const std::string& path);

// Override the neighbor-expansion cutoff used by FastRAG when reading the
// dense matrix (`g_neighbor_threshold` in corrector.cc). Must be called
// BEFORE LoadConfusionMatrix — that's where the neighbor lists are built.
// Caller responsibility: invoke once at init time, no thread safety.
void SetNeighborThreshold(float thr);

enum class Lang { ZH, EN, NUM, OTHER };

struct Phoneme {
    std::string value;
    Lang lang;
    bool is_word_start = false;
    bool is_word_end = false;
    size_t char_start = 0;
    size_t char_end = 0;
    bool is_tone() const; // 实现移至 .cc
};

struct MatchResult { 
    size_t start; 
    size_t end; 
    float score; 
    float avg_confidence;
    std::string hotword; 
};
struct SimilarityResult { std::string original; std::string hotword; float score; };
struct CorrectionResult { 
    std::string text; 
    // std::vector<SimilarityResult> matchs; 
    std::vector<MatchResult> matchs; 
    // std::vector<SimilarityResult> similars; 
};
struct SplitPhoneme { std::string hanzi; std::string initial; std::string final; std::string tone; bool is_valid_pinyin = true; };

namespace Utils {
    int utf8_char_len(unsigned char c);
    bool is_chinese(const std::string& char_bytes);
    int lcs_length(const std::string& s1, const std::string& s2);
}

class PinyinSplitter {
public:
    static const std::vector<std::string> INITIALS; // 声明静态成员
    static SplitPhoneme split(const std::string& hanzi, const std::string& tone3);
};

class PinyinProvider {
    static std::unique_ptr<::Pinyin::Pinyin> g_pinyin; 
    static std::once_flag init_flag;                   
    static std::mutex _mutex;
public:
    static void initialize(const std::string& dictPath);
    static size_t process_zh(const std::string& text, size_t pos, std::vector<Phoneme>& seq);
};

class EnglishProvider {
public:
    static void initialize(const std::string& dict_path);
    static bool get_phonemes(const std::string& word, std::vector<std::string>& out);

private:
    static std::unordered_map<std::string, std::vector<std::string>> _dict;
    static std::once_flag init_flag;
    static std::mutex _mutex;
};

size_t process_en_num(const std::string& text, size_t pos, std::vector<Phoneme>& seq, bool split_char);
std::vector<Phoneme> get_phoneme_info(const std::string& text);
float _get_tuple_cost(const Phoneme& t1, const Phoneme& t2);
std::vector<std::tuple<float, int, int>> fuzzy_substring_search_constrained(
    const std::vector<Phoneme>& hw_info, const std::vector<Phoneme>& input_info, float threshold = 0.6f);
std::vector<std::tuple<float, int, int>> fuzzy_substring_search_constrained_with_confidence(
    const std::vector<Phoneme>& hw_info, const std::vector<Phoneme>& input_info, float threshold = 0.6f, const std::vector<float>& token_confidences = {},  
    float confidence_threshold = 0.7f);

class PhonemeEncoder {
    std::unordered_map<std::string, int> phoneme_to_code;
    int next_code = 1;
public:
    int encode(const std::string& p);
    std::vector<int> encode_sequence(const std::vector<Phoneme>& phonemes);
};

class PhonemeIndex {
    PhonemeEncoder encoder;
    std::unordered_map<int, std::vector<std::pair<std::string, std::vector<int>>>> index;
public:
    void add(const std::string& hotword, const std::vector<Phoneme>& phonemes);
    std::vector<std::pair<std::string, std::vector<int>>> get_candidates(const std::vector<Phoneme>& input_phonemes);
    std::vector<int> encode_input(const std::vector<Phoneme>& ps);
};

class FastRAG {
    float threshold;
    PhonemeIndex index;
    float _distance(const std::vector<int>& main, const std::vector<int>& sub);
public:
    FastRAG(float thr = 0.6f);
    void add_hotwords(const std::unordered_map<std::string, std::vector<Phoneme>>& hotwords);
    std::vector<std::pair<std::string, float>> search(const std::vector<Phoneme>& input_phonemes, int top_k = 10);
};

class PhonemeCorrector {
    float threshold; 
    float threshold_en;
    float similar_threshold;
    std::unordered_map<std::string, std::vector<Phoneme>> hotwords;
    std::unique_ptr<FastRAG> fast_rag;
    std::mutex _lock;
public:
    PhonemeCorrector(float threshold = 0.7f, float threshold_en = 0.6f);
    void update_hotwords(const std::string& hotword_text);
    void update_dynamic_hotwords(const std::unordered_map<std::string, float>& word_boost_map);
    CorrectionResult correct(const std::string& text);

    CorrectionResult correct_with_confidence(
        const std::string& text,
        const std::vector<float>& token_confidences
    );
private:
    std::unordered_map<std::string, float> dynamic_hotword_boosts_;
};

} // namespace HotwordCorrection

#endif // DECODER_CORRECTOR_H_