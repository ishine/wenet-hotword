#include "decoder/corrector.h"
#include "utils/log.h"
#include "utils/utils.h"
#include <iomanip>
#include <sstream>
#include <fstream>     
#include <algorithm>   
#include <cmath>
namespace HotwordCorrection {

const std::map<std::pair<std::string, std::string>, float> CONFUSION_MATRIX = {
    {{"z", "zh"}, 0.1f}, {{"zh", "z"}, 0.1f},
    {{"c", "ch"}, 0.1f}, {{"ch", "c"}, 0.1f},
    {{"s", "sh"}, 0.1f}, {{"sh", "s"}, 0.1f},

    {{"l", "n"}, 0.2f},  {{"n", "l"}, 0.2f},
    {{"f", "h"}, 0.3f},  {{"h", "f"}, 0.3f},
    {{"an", "ang"}, 0.2f},   {{"ang", "an"}, 0.2f},
    {{"en", "eng"}, 0.2f},   {{"eng", "en"}, 0.2f},
    {{"in", "ing"}, 0.2f},   {{"ing", "in"}, 0.2f},
    
    {{"ian", "iang"}, 0.2f}, {{"iang", "ian"}, 0.2f},
    {{"uan", "uang"}, 0.2f}, {{"uang", "uan"}, 0.2f},

    // // EN
    // {{"B", "b"}, 0.05f}, {{"b", "B"}, 0.05f},
    // {{"P", "p"}, 0.05f}, {{"p", "P"}, 0.05f}, 
    // {{"M", "m"}, 0.05f}, {{"m", "M"}, 0.05f},
    // {{"F", "f"}, 0.05f}, {{"f", "F"}, 0.05f},
    // {{"D", "d"}, 0.05f}, {{"d", "D"}, 0.05f},
    // {{"T", "t"}, 0.05f}, {{"t", "T"}, 0.05f},
    // {{"N", "n"}, 0.05f}, {{"n", "N"}, 0.05f},
    // {{"L", "l"}, 0.05f}, {{"l", "L"}, 0.05f},
    // {{"G", "g"}, 0.05f}, {{"g", "G"}, 0.05f},
    // {{"K", "k"}, 0.05f}, {{"k", "K"}, 0.05f},
    // // {{"H", "h"}, 0.05f}, {{"h", "H"}, 0.05f},
    // {{"S", "s"}, 0.05f}, {{"s", "S"}, 0.05f},
    // {{"W", "w"}, 0.05f}, {{"w", "W"}, 0.05f},
    // {{"Z", "z"}, 0.05f}, {{"z", "Z"}, 0.05f},

    // {{"JH", "j"}, 0.05f}, {{"j", "JH"}, 0.05f},   
    // {{"CH", "ch"}, 0.05f}, {{"ch", "CH"}, 0.05f}, 
    // {{"SH", "sh"}, 0.05f}, {{"sh", "SH"}, 0.05f}, 

    // {{"IY", "i"}, 0.05f}, {{"i", "IY"}, 0.05f}, 
    // {{"EY", "ei"}, 0.05f}, {{"ei", "EY"}, 0.05f},
    // {{"AY", "ai"}, 0.05f}, {{"ai", "AY"}, 0.05f},
    // {{"OW", "ou"}, 0.05f}, {{"ou", "OW"}, 0.05f},
    // {{"UW", "u"}, 0.05f}, {{"u", "UW"}, 0.05f},
    // {{"AA", "a"}, 0.05f}, {{"a", "AA"}, 0.05f},
    // {{"AH", "a"}, 0.05f}, {{"a", "AH"}, 0.05f},
    // {{"ER", "er"}, 0.05f}, {{"er", "ER"}, 0.05f}
};

float CalculateAvgConfidenceInRange(
    const std::string& text, 
    const std::vector<float>& token_log_probs, 
    size_t byte_start, 
    size_t byte_end) 
{
    if (token_log_probs.empty()) return 1.0f;
    
    float sum_conf = 0.0f;
    int count = 0;
    size_t curr_byte_pos = 0;
    int token_idx = 0;
    size_t text_len = text.length();

    while (curr_byte_pos < text_len) {
        int char_len = Utils::utf8_char_len(text[curr_byte_pos]);
        size_t next_byte_pos = curr_byte_pos + char_len;

        if (curr_byte_pos >= byte_start && curr_byte_pos < byte_end) {
            if (token_idx < (int)token_log_probs.size()) {
                sum_conf += std::exp(token_log_probs[token_idx]);
                count++;
            }
        }
        curr_byte_pos = next_byte_pos;
        token_idx++;
        
        if (curr_byte_pos >= byte_end) break;
    }
    return count > 0 ? (sum_conf / count) : 1.0f;
}

bool Phoneme::is_tone() const { 
    return lang == Lang::ZH && !value.empty() && std::isdigit(value[0]); 
}

namespace Utils {
    int utf8_char_len(unsigned char c) {
        if ((c & 0x80) == 0) return 1; if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3; if ((c & 0xF8) == 0xF0) return 4; return 1;
    }
    bool is_chinese(const std::string& char_bytes) {
        if (char_bytes.size() != 3) return false;
        unsigned char c = static_cast<unsigned char>(char_bytes[0]);
        return (c >= 0xE4 && c <= 0xE9); 
    }
    int lcs_length(const std::string& s1, const std::string& s2) {
        size_t m = s1.length(); size_t n = s2.length(); if (m == 0 || n == 0) return 0;
        std::vector<int> prev(n + 1, 0), curr(n + 1, 0);
        for (size_t i = 1; i <= m; ++i) {
            for (size_t j = 1; j <= n; ++j) {
                if (s1[i - 1] == s2[j - 1]) curr[j] = prev[j - 1] + 1;
                else curr[j] = std::max(prev[j], curr[j - 1]);
            }
            prev = curr;
        }
        return prev[n];
    }
}

const std::vector<std::string> PinyinSplitter::INITIALS = {
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"
};

SplitPhoneme PinyinSplitter::split(const std::string& hanzi, const std::string& tone3) {
    SplitPhoneme p; p.hanzi = hanzi;
    if (tone3.empty() || tone3 == hanzi) {
        p.initial = ""; p.final = hanzi; p.tone = ""; p.is_valid_pinyin = false; return p;
    }
    std::string s = tone3;
    // [FIX] Tone 5 alignment
    if (!s.empty() && std::isdigit(s.back())) {
        p.tone = s.back(); s.pop_back();
    } else { p.tone = "5"; }

    bool found = false;
    for (const auto& i : INITIALS) {
        if (s.size() >= i.size() && s.compare(0, i.size(), i) == 0) {
            if (i == "r" && s == "er") continue;
            p.initial = i; p.final = s.substr(i.size()); found = true; break;
        }
    }
    if (!found) { p.initial = ""; p.final = s; }
    if ((p.initial == "j" || p.initial == "q" || p.initial == "x" || p.initial == "y") && 
        (p.final == "v" || p.final == "yu")) { p.final = "u"; }
    if (p.initial == "y" && p.final == "ou") p.final = "ou";
    if (p.initial == "y" && p.final == "ao") p.final = "ao";
    return p;
}

std::unique_ptr<::Pinyin::Pinyin> PinyinProvider::g_pinyin = nullptr;
std::once_flag PinyinProvider::init_flag;
std::mutex PinyinProvider::_mutex;

void PinyinProvider::initialize(const std::string& dictPath) {
    std::call_once(init_flag, [&]() {
        ::Pinyin::setDictionaryPath(dictPath);
        g_pinyin = std::make_unique<::Pinyin::Pinyin>();
    });
}

size_t PinyinProvider::process_zh(const std::string& text, size_t pos, std::vector<Phoneme>& seq) {
    size_t scan_pos = pos; size_t len = text.length();
    while (scan_pos < len) {
        int char_len = Utils::utf8_char_len(text[scan_pos]);
        if (char_len + scan_pos > len) break;
        std::string ch = text.substr(scan_pos, char_len);
        if (!Utils::is_chinese(ch)) break;
        scan_pos += char_len;
    }
    std::string fragment = text.substr(pos, scan_pos - pos);
    if (fragment.empty()) return pos;
    if (!g_pinyin) return scan_pos;

    ::Pinyin::PinyinResVector res_tones;
    {
        std::lock_guard<std::mutex> lock(_mutex); 
        res_tones = g_pinyin->hanziToPinyin(fragment, ::Pinyin::ManTone::Style::TONE3, ::Pinyin::Error::Default, false, false, true);
    }


    size_t current_char_idx = pos;
    for (const auto& item : res_tones) {
        SplitPhoneme sp = PinyinSplitter::split(item.hanzi, item.pinyin);
        size_t start = current_char_idx;
        size_t end = current_char_idx + item.hanzi.length();
        current_char_idx = end;
        if (!sp.is_valid_pinyin) {
            seq.push_back({sp.final, Lang::ZH, true, true, start, end});
        } else {
            bool has_added = false;
            if (!sp.initial.empty()) { seq.push_back({sp.initial, Lang::ZH, true, false, start, end}); has_added = true; }
            if (!sp.final.empty()) { seq.push_back({sp.final, Lang::ZH, !has_added, false, start, end}); }
            if (!sp.tone.empty()) { seq.push_back({sp.tone, Lang::ZH, false, true, start, end}); }
            else if (!seq.empty()) seq.back().is_word_end = true;
        }
    }
    return scan_pos;
}

std::unordered_map<std::string, std::vector<std::string>> EnglishProvider::_dict;
std::once_flag EnglishProvider::init_flag;
std::mutex EnglishProvider::_mutex;

void EnglishProvider::initialize(const std::string& dict_path) {
    std::call_once(init_flag, [&]() {
        std::ifstream file(dict_path);
        if (!file.is_open()) {
            LOG(ERROR) << "Cannot open English dict file: " << dict_path;
            return;
        }

        std::string line;
        int count = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line.substr(0, 3) == ";;;") continue;
            std::stringstream ss(line);
            std::string word, phoneme;
            ss >> word;
            
            std::transform(word.begin(), word.end(), word.begin(), ::toupper);

            size_t paren_pos = word.find('(');
            if (paren_pos != std::string::npos) {
                word = word.substr(0, paren_pos);
            }

            std::vector<std::string> phonemes;
            while (ss >> phoneme) {
                if (!phoneme.empty() && std::isdigit(phoneme.back())) {
                    phoneme.pop_back();
                }
                phonemes.push_back(phoneme);
            }
            _dict[word] = phonemes;
            count++;
        }
        LOG(INFO) << "EnglishProvider loaded " << count << " words from " << dict_path;
    });
}

bool EnglishProvider::get_phonemes(const std::string& word, std::vector<std::string>& out) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_dict.empty()) return false;
    std::string upper_word = word;
    //std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), 
               [](unsigned char c) { return std::toupper(c); });

    auto it = _dict.find(upper_word);
    if (it != _dict.end()) {
        out = it->second;
        return true;
    }
    return false;
}

// size_t process_en_num(const std::string& text, size_t pos, std::vector<Phoneme>& seq, bool split_char) {
//     size_t start_pos = pos; size_t len = text.length();
//     while (pos < len) {
//         char c = text[pos];
//         if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) break;
//         pos++;
//     }
//     size_t end_pos = pos;
//     std::string token = text.substr(start_pos, end_pos - start_pos);
//     std::string token_lower = token;
//     std::transform(token_lower.begin(), token_lower.end(), token_lower.begin(), ::tolower);
//     Lang lang = (!token_lower.empty() && std::isdigit(token_lower[0])) ? Lang::NUM : Lang::EN;
//     if (split_char) {
//         for (size_t i = 0; i < token.length(); ++i) {
//             seq.push_back({std::string(1, token_lower[i]), lang, (i == 0), (i == token.length() - 1), start_pos + i, start_pos + i + 1});
//         }
//     } else { seq.push_back({token_lower, lang, true, true, start_pos, end_pos}); }
//     return end_pos;
// }


size_t process_en_num(const std::string& text, size_t pos, std::vector<Phoneme>& seq, bool split_char) {
    size_t start_pos = pos; size_t len = text.length();
    
    while (pos < len) {
        char c = text[pos];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) break;
        pos++;
    }
    size_t end_pos = pos;
    std::string token = text.substr(start_pos, end_pos - start_pos);
    
    bool is_alpha = std::all_of(token.begin(), token.end(), [](char c){ return std::isalpha(c); });

    std::vector<std::string> arpabet_phonemes;
    
    if (is_alpha && EnglishProvider::get_phonemes(token, arpabet_phonemes)) {
        for (size_t i = 0; i < arpabet_phonemes.size(); ++i) {
            seq.push_back({
                arpabet_phonemes[i], 
                Lang::EN, 
                (i == 0),             
                (i == arpabet_phonemes.size() - 1), 
                start_pos,            
                end_pos               
            });
        }
    } else if (is_alpha) {
        for (size_t i = 0; i < token.length(); ++i) {
            std::string letter_str(1, token[i]); 
            std::vector<std::string> letter_phones;
            
            if (EnglishProvider::get_phonemes(letter_str, letter_phones)) {
                for (const auto& ph : letter_phones) {
                    seq.push_back({
                        ph, Lang::EN,
                        (i == 0), (i == token.length() - 1),
                        start_pos + i, start_pos + i + 1 
                    });
                }
            } else {
                std::string lower_char = letter_str;
                std::transform(lower_char.begin(), lower_char.end(), lower_char.begin(), ::tolower);
                seq.push_back({
                    lower_char, Lang::EN,
                    (i == 0), (i == token.length() - 1),
                    start_pos + i, start_pos + i + 1
                });
            } 

        } 
    }else {
        // Fallback 
        std::string token_lower = token;
        std::transform(token_lower.begin(), token_lower.end(), token_lower.begin(), ::tolower);
        Lang lang = (!token_lower.empty() && std::isdigit(token_lower[0])) ? Lang::NUM : Lang::EN;
        
        if (split_char) {
            for (size_t i = 0; i < token.length(); ++i) {
                seq.push_back({std::string(1, token_lower[i]), lang, (i == 0), (i == token.length() - 1), start_pos + i, start_pos + i + 1});
            }
        } else { 
            seq.push_back({token_lower, lang, true, true, start_pos, end_pos}); 
        }
    }
    return end_pos;
}

static bool is_separator(unsigned char c) {
    return std::isspace(c) || c == ',' || c == '.' || c == '?' || c == '!' || c == ';' || c == ':';
}


std::vector<Phoneme> get_phoneme_info(const std::string& text) {
    std::vector<Phoneme> seq; 
    size_t pos = 0; 
    size_t len = text.length();
    while (pos < len) {
        unsigned char c = static_cast<unsigned char>(text[pos]);
        int char_len = Utils::utf8_char_len(c);

        if (char_len == 3 && c >= 0xE4 && c <= 0xE9) {
            pos = PinyinProvider::process_zh(text, pos, seq);
        }
        else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) 
            pos = process_en_num(text, pos, seq, true);
        else if (is_separator(c)) 
            pos ++;
        else {
            LOG(INFO) << "Found unrecognizable char in text: " << text << " , skipping this word.";
            return {}; 
        }
    }
    return seq;
}

float _get_tuple_cost(const Phoneme& t1, const Phoneme& t2) {
    if (t1.lang != t2.lang) return 5.0f;
    if (t1.value == t2.value) return 0.0f;

    if (t1.lang == Lang::ZH) {
        bool t1_is_tone = t1.is_tone();
        bool t2_is_tone = t2.is_tone();

        if (t1_is_tone != t2_is_tone) {
            return 1.0f; 
        }
        if (t1.is_tone()) {
            return 0.2f;
        }

        std::pair<std::string, std::string> key = {t1.value, t2.value};
        
        if (CONFUSION_MATRIX.count(key)) {
            return CONFUSION_MATRIX.at(key);
        }
        return 1.0f;
    }

    if (t1.lang == Lang::EN) {
        int max_len = std::max(t1.value.length(), t2.value.length());
        if (max_len > 0) return 1.0f - (static_cast<float>(Utils::lcs_length(t1.value, t2.value)) / max_len);
    }
    return 1.0f;
}

std::vector<std::tuple<float, int, int>> fuzzy_substring_search_constrained(
    const std::vector<Phoneme>& hw_info, const std::vector<Phoneme>& input_info, float threshold) 
{
    int n = hw_info.size(); int m = input_info.size();
    if (n == 0 || m == 0) return {};
    std::vector<std::vector<float>> dp(n + 1, std::vector<float>(m + 1, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<int>> path_start(n + 1, std::vector<int>(m + 1, 0));

    // [FIX] DP Initialization fix
    for (int j = 0; j <= m; ++j) {
        if (j == 0) { dp[0][0] = 0.0f; path_start[0][0] = 0; }
        if (j < m && input_info[j].is_word_start) { dp[0][j] = 0.0f; path_start[0][j] = j; }
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = _get_tuple_cost(hw_info[i - 1], input_info[j - 1]);
            float d_m = dp[i - 1][j - 1] + cost;
            float d_d = dp[i - 1][j] + 1.0f;
            float d_i = dp[i][j - 1] + 1.0f;
            float min_dist = std::min({d_m, d_d, d_i});
            dp[i][j] = min_dist;
            if (min_dist == d_m) path_start[i][j] = path_start[i - 1][j - 1];
            else if (min_dist == d_d) path_start[i][j] = path_start[i - 1][j];
            else path_start[i][j] = path_start[i][j - 1];
        }
    }
    std::vector<std::tuple<float, int, int>> results;
    std::unordered_map<int, std::tuple<float, int, int>> used_ends;
    for (int j = 1; j <= m; ++j) {
        if (!input_info[j - 1].is_word_end) continue;
        float dist = dp[n][j];
        if (dist >= n * 0.8f) continue;
        float score = 1.0f - (dist / n);
        if (score >= threshold) {
            int start_idx = path_start[n][j];
            if (used_ends.find(j) == used_ends.end() || score > std::get<0>(used_ends[j])) {
                used_ends[j] = {score, start_idx, j};
            }
        }
    }
    for (const auto& kv : used_ends) results.push_back(kv.second);
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    return results;
}

std::vector<std::tuple<float, int, int>> fuzzy_substring_search_constrained_with_confidence(
    const std::vector<Phoneme>& hw_info, 
    const std::vector<Phoneme>& input_info,
    float threshold,
    const std::vector<float>& token_confidences,  
    float confidence_threshold                  
) {
	int n = hw_info.size(); int m = input_info.size();
    if (n == 0 || m == 0) return {};

    std::vector<std::vector<float>> dp(n + 1, std::vector<float>(m + 1, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<int>> path_start(n + 1, std::vector<int>(m + 1, 0));    
    // [FIX] DP Initialization fix
    for (int j = 0; j <= m; ++j) {
         if (j == 0) { dp[0][0] = 0.0f; path_start[0][0] = 0; }
         if (j < m && input_info[j].is_word_start) { dp[0][j] = 0.0f; path_start[0][j] = j; }
    }
        
    for (int i = 1; i <= n; ++i) {
         for (int j = 1; j <= m; ++j) {

            float base_cost = _get_tuple_cost(hw_info[i - 1], input_info[j - 1]);
            float weighted_cost = base_cost;

		    if (!token_confidences.empty() && j-1 < token_confidences.size()) {
                float confidence = std::exp(token_confidences[j-1]);
                    float weight = 0.2f + 0.8f * confidence;
                    weight = std::max(0.1f, std::min(1.0f, weight));                    
                    weighted_cost = base_cost * weight;
            }
            
            
            float d_m = dp[i - 1][j - 1] + weighted_cost;
            float d_d = dp[i - 1][j] + 1.0f;
            float d_i = dp[i][j - 1] + 1.0f;
            float min_dist = std::min({d_m, d_d, d_i}); 
            dp[i][j] = min_dist;
            if (min_dist == d_m) path_start[i][j] = path_start[i - 1][j - 1];
            else if (min_dist == d_d) path_start[i][j] = path_start[i - 1][j];
            else path_start[i][j] = path_start[i][j - 1];
        }
    }   
    std::vector<std::tuple<float, int, int>> results;
    std::unordered_map<int, std::tuple<float, int, int>> used_ends; 
    Lang hw_lang = hw_info.empty() ? Lang::ZH : hw_info[0].lang;

    for (int j = 1; j <= m; ++j) {
        if (!input_info[j - 1].is_word_end) continue;
        float dist = dp[n][j];
        if (dist >= n * 0.8f) continue;
        float score = 1.0f - (dist / n);
        if (score >= threshold) {
            int start_idx = path_start[n][j];
            if (hw_lang == Lang::EN) {
                if (start_idx > 0 && input_info[start_idx - 1].lang == Lang::EN) continue;
                if (j < m && input_info[j].lang == Lang::EN) continue;
            }

            if (used_ends.find(j) == used_ends.end() || score > std::get<0>(used_ends[j])) {
                used_ends[j] = {score, start_idx, j};
            }
        }
    }   
    for (const auto& kv : used_ends) results.push_back(kv.second);
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    return results;

}


// PhonemeEncoder 实现
int PhonemeEncoder::encode(const std::string& p) {
    if (phoneme_to_code.find(p) == phoneme_to_code.end()) { phoneme_to_code[p] = next_code++; }
    return phoneme_to_code[p];
}
std::vector<int> PhonemeEncoder::encode_sequence(const std::vector<Phoneme>& phonemes) {
    std::vector<int> res; for (const auto& p : phonemes) res.push_back(encode(p.value)); return res;
}

// PhonemeIndex 实现
void PhonemeIndex::add(const std::string& hotword, const std::vector<Phoneme>& phonemes) {
    if (phonemes.empty()) return;
    std::vector<int> codes = encoder.encode_sequence(phonemes);
    std::vector<int> indices = {0};
    if (phonemes[0].lang == Lang::EN) { size_t limit = std::min(codes.size(), (size_t)2); for(size_t k=0; k<limit; ++k) indices.push_back(k); }
    for (int idx : indices) if (idx < (int)codes.size()) index[codes[idx]].push_back({hotword, codes});
}
// std::vector<std::pair<std::string, std::vector<int>>> PhonemeIndex::get_candidates(const std::vector<Phoneme>& input_phonemes) {
//     std::unordered_set<int> input_codes;
//     for (const auto& p : input_phonemes) {
//         input_codes.insert(encoder.encode(p.value));
//         if (p.lang == Lang::ZH) { for (const auto& s : SIMILAR_PHONEMES) { if (s.count(p.value)) for (const auto& v : s) input_codes.insert(encoder.encode(v)); } }
//     }
//     std::vector<std::pair<std::string, std::vector<int>>> candidates;
//     std::unordered_set<std::string> seen;
//     for (int code : input_codes) {
//         if (index.count(code)) for (const auto& item : index[code]) { if (seen.find(item.first) == seen.end()) { candidates.push_back(item); seen.insert(item.first); } }
//     }
//     return candidates;
// }

std::vector<std::pair<std::string, std::vector<int>>> PhonemeIndex::get_candidates(const std::vector<Phoneme>& input_phonemes) {
    std::unordered_set<int> input_codes;
    
    for (const auto& p : input_phonemes) {
        input_codes.insert(encoder.encode(p.value));

        if (p.lang == Lang::ZH) { 
            for (const auto& entry : CONFUSION_MATRIX) {
                const auto& pair_keys = entry.first;
                
                if (pair_keys.first == p.value) {
                    input_codes.insert(encoder.encode(pair_keys.second));
                }
            }
        }
    }

    std::vector<std::pair<std::string, std::vector<int>>> candidates;
    std::unordered_set<std::string> seen;
    for (int code : input_codes) {
        if (index.count(code)) {
            for (const auto& item : index[code]) { 
                if (seen.find(item.first) == seen.end()) { 
                    candidates.push_back(item); 
                    seen.insert(item.first); 
                } 
            }
        }
    }
    return candidates;
}

std::vector<int> PhonemeIndex::encode_input(const std::vector<Phoneme>& ps) { return encoder.encode_sequence(ps); }

float FastRAG::_distance(const std::vector<int>& main, const std::vector<int>& sub) {
    int n = sub.size(), m = main.size(); if (n == 0 || m == 0) return (float)n;
    std::vector<std::vector<float>> dp(n + 1, std::vector<float>(m + 1, 0.0f));
    for (int i = 1; i <= n; ++i) dp[i][0] = (float)i;
    for (int i = 1; i <= n; ++i) for (int j = 1; j <= m; ++j) {
        float cost = (sub[i - 1] == main[j - 1]) ? 0.0f : 1.0f;
        dp[i][j] = std::min({dp[i - 1][j] + 1.0f, dp[i][j - 1] + 1.0f, dp[i - 1][j - 1] + cost});
    }
    float min_val = std::numeric_limits<float>::infinity();
    for (int j = 1; j <= m; ++j) if (dp[n][j] < min_val) min_val = dp[n][j];
    return min_val;
}

FastRAG::FastRAG(float thr) : threshold(thr) {}
void FastRAG::add_hotwords(const std::unordered_map<std::string, std::vector<Phoneme>>& hotwords) {
    index = PhonemeIndex(); for (const auto& kv : hotwords) index.add(kv.first, kv.second);
}
std::vector<std::pair<std::string, float>> FastRAG::search(const std::vector<Phoneme>& input_phonemes, int top_k) {
    if (input_phonemes.empty()) return {};
    std::vector<int> input_codes = index.encode_input(input_phonemes);
    auto candidates = index.get_candidates(input_phonemes);
    std::vector<std::pair<std::string, float>> results;
    for (const auto& cand : candidates) {
        if (cand.second.size() > input_codes.size() + 3) continue;
        float score = 1.0f - (_distance(input_codes, cand.second) / cand.second.size());
        if (score >= threshold) results.push_back({cand.first, score});
    }
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b){ return a.second > b.second; });
    if (results.size() > (size_t)top_k) results.resize(top_k);
    return results;
}

// PhonemeCorrector 实现
PhonemeCorrector::PhonemeCorrector(float threshold, float threshold_en) : threshold(threshold), threshold_en(threshold_en){
    this->similar_threshold = threshold - 0.2f;
    fast_rag = std::make_unique<FastRAG>(std::min(threshold, similar_threshold) - 0.1f);
}

// void PhonemeCorrector::update_hotwords(const std::string& hotword_text) {
//     std::stringstream ss(hotword_text); std::string line;
//     std::unordered_map<std::string, std::vector<Phoneme>> new_hotwords;
//     LOG(INFO) << "corrector start to update hotwords";
//     while (std::getline(ss, line)) {
//         line.erase(0, line.find_first_not_of(" \t\r\n")); line.erase(line.find_last_not_of(" \t\r\n") + 1);
//         if (line.empty() || line[0] == '#') continue;
//         std::vector<Phoneme> phons = get_phoneme_info(line);
//         if (!phons.empty()) new_hotwords[line] = phons;
//     }
//     std::lock_guard<std::mutex> guard(_lock);
//     hotwords = new_hotwords; fast_rag->add_hotwords(hotwords);
//     LOG(INFO) << "Loaded " << hotwords.size() << " hotwords.";
// }

void PhonemeCorrector::update_hotwords(const std::string& hotword_text) {
    std::stringstream ss(hotword_text);
    std::string line;
    
    std::unordered_map<std::string, std::vector<Phoneme>> new_hotwords;
    new_hotwords.reserve(50000); 

    LOG(INFO) << "corrector start to update hotwords";

    while (std::getline(ss, line)) {
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue; 
        auto end = line.find_last_not_of(" \t\r\n");
        std::string trimmed_line = line.substr(start, end - start + 1);

        if (trimmed_line.empty() || trimmed_line[0] == '#') continue;
        LOG(INFO) << "current str" << trimmed_line;
        std::vector<Phoneme> phons = get_phoneme_info(trimmed_line);
        
        if (!phons.empty()) {
            new_hotwords[std::move(trimmed_line)] = std::move(phons);
        }
    }

    std::lock_guard<std::mutex> guard(_lock);

    hotwords.swap(new_hotwords); 
    fast_rag->add_hotwords(hotwords); 
    
    LOG(INFO) << "Loaded " << hotwords.size() << " hotwords.";
}

void PhonemeCorrector::update_dynamic_hotwords(const std::unordered_map<std::string, float>& word_boost_map) {
    std::lock_guard<std::mutex> guard(_lock);
    dynamic_hotword_boosts_ = word_boost_map;
}

CorrectionResult PhonemeCorrector::correct_with_confidence(const std::string& text, const std::vector<float>& token_confidences) {
    if (text.empty() || hotwords.empty()) return {text, {}};
    // NOTE: 测试期 hardcode 输入已移除，恢复使用真实 ASR 输出
    std::vector<Phoneme> input_phonemes = get_phoneme_info(text);
    if (input_phonemes.empty()) return {text, {}};
    
    std::vector<std::pair<std::string, float>> fast_results;
    { 
        std::lock_guard<std::mutex> guard(_lock); 
        fast_results = fast_rag->search(input_phonemes, 100); 
    }

    
    std::vector<MatchResult> matches;

    for (const auto& item : fast_results) {
        bool is_hw_en = false;
        auto it = hotwords.find(item.first);
        if (it != hotwords.end() && !it->second.empty()) {
            if (it->second[0].lang == Lang::EN) {
                is_hw_en = true;
            }
        }

        float boost = 0.0f;
        auto boost_it = dynamic_hotword_boosts_.find(item.first);
        if (boost_it != dynamic_hotword_boosts_.end()) {
            boost = boost_it->second;
        }
        float current_threshold = is_hw_en ? threshold_en : threshold;
        current_threshold -= boost;

        auto found = fuzzy_substring_search_constrained_with_confidence(hotwords[item.first], input_phonemes, similar_threshold - 0.1f, token_confidences);

        for (const auto& seg : found) {
            float score = std::get<0>(seg);
            size_t c_start = input_phonemes[std::get<1>(seg)].char_start;
            size_t c_end = input_phonemes[std::get<2>(seg) - 1].char_end;
            
            if (score >= current_threshold) {
                float avg_conf = CalculateAvgConfidenceInRange(text, token_confidences, c_start, c_end);
                float final_score = score + boost * 15;
                matches.push_back({c_start, c_end, final_score, avg_conf, item.first});
            } 
        }
    }

    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        if (std::abs(a.score - b.score) > 0.001f) return a.score > b.score; 
        return (a.end - a.start) > (b.end - b.start);
    });

    std::vector<MatchResult> final_matches; 
    std::vector<std::pair<size_t, size_t>> occupied;
    
    for (const auto& m : matches) {
        bool overlap = false;
        for (const auto& r : occupied) {
            if (!(m.end <= r.first || m.start >= r.second)) { 
                overlap = true; 
                break; 
            }
        }
        if (!overlap) {
            if (text.substr(m.start, m.end - m.start) != m.hotword) {
                final_matches.push_back(m);
                occupied.push_back({m.start, m.end});
            }
        }
    }

    std::sort(final_matches.begin(), final_matches.end(), [](const auto& a, const auto& b) { 
        return a.start > b.start; // start 大的排前面 (倒序)
    });
    
    std::string new_text = text;
    for (const auto& m : final_matches) {
        new_text.replace(m.start, m.end - m.start, m.hotword);
    }

    return {new_text, final_matches};
}

CorrectionResult PhonemeCorrector::correct(const std::string& text) {
    if (text.empty() || hotwords.empty()) return {text, {}};
    
    std::vector<Phoneme> input_phonemes = get_phoneme_info(text);
    if (input_phonemes.empty()) return {text, {}};
    
    std::vector<std::pair<std::string, float>> fast_results;
    { 
        std::lock_guard<std::mutex> guard(_lock); 
        fast_results = fast_rag->search(input_phonemes, 100); 
    }

    std::vector<MatchResult> matches;

    for (const auto& item : fast_results) {
        auto found = fuzzy_substring_search_constrained(hotwords[item.first], input_phonemes, similar_threshold - 0.1f);

        for (const auto& seg : found) {
            float score = std::get<0>(seg);
            size_t c_start = input_phonemes[std::get<1>(seg)].char_start;
            size_t c_end = input_phonemes[std::get<2>(seg) - 1].char_end;

            if (score >= threshold) {
                matches.push_back({c_start, c_end, score, 1.0f, item.first});
            } 
        }
    }

    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        if (std::abs(a.score - b.score) > 0.001f) return a.score > b.score; 
        return (a.end - a.start) > (b.end - b.start);
    });

    std::vector<MatchResult> final_matches; 
    std::vector<std::pair<size_t, size_t>> occupied;
    
    for (const auto& m : matches) {
        bool overlap = false;
        for (const auto& r : occupied) {
            if (!(m.end <= r.first || m.start >= r.second)) { 
                overlap = true; 
                break; 
            }
        }
        if (!overlap) {
            if (text.substr(m.start, m.end - m.start) != m.hotword) {
                final_matches.push_back(m);
                occupied.push_back({m.start, m.end});
            }
        }
    }

    std::sort(final_matches.begin(), final_matches.end(), [](const auto& a, const auto& b) { 
        return a.start > b.start; // start 大的排前面 (倒序)
    });
    
    std::string new_text = text;
    for (const auto& m : final_matches) {
        new_text.replace(m.start, m.end - m.start, m.hotword);
    }

    return {new_text, final_matches};
}

} // namespace HotwordCorrection
