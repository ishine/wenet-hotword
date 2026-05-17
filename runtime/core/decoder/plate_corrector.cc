#include "decoder/plate_corrector.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <cctype>

#include <cpp-pinyin/Pinyin.h>
#include <cpp-pinyin/G2pglobal.h>

enum class Lang { ZH, EN, NUM, OTHER, SPACER };

struct Phoneme {
    std::string value; 
    std::string text;  
    std::string pre_padding; // [新增] 专门存放该字符左侧紧邻的空白符
    Lang lang;
    std::string initial;
    std::string final;
    int tone;          
};

namespace {

static std::string kDefaultDictPath = "";  // 由 PlateCorrector::Initialize 注入

static const std::unordered_map<char, std::string> kStdAlphaPinyin = {
    {'a', "ei"}, 
    {'b', "bi"},  
    {'c', "sei"}, 
    {'d', "di"},  
    {'e', "yi"},
    {'f', "fu"},
    {'g', "ji"},
    {'h', "eh"},
    {'i', "ai"},  
    {'j', "jie"}, 
    {'k', "kei"}, 
    {'l', "el"},  
    {'m', "em"},  
    {'n', "en"},
    {'o', "ou"}, 
    {'p', "pi"},
    {'q', "kiu"},
    {'r', "ar"},
    {'s', "si"},
    {'t', "ti"},
    {'u', "you"},
    {'v', "wei"},
    {'w', "da"},
    {'x', "xi"},
    {'y', "wai"},
    {'z', "zei"}
       
};

const std::unordered_map<std::string, std::string> kWhitelist = {
    // Digit
    {"幺", "1"}, {"一", "1"},
    {"两", "2"}, {"二", "2"},
    {"三", "3"},
    {"四", "4"},
    {"五", "5"},
    {"六", "6"},
    {"拐", "7"}, {"七", "7"},
    {"八", "8"},
    {"九", "9"},
    {"洞", "0"}, {"零", "0"},

    // Letter
    {"叉", "X"},
    {"勾", "J"},
    {"尖", "A"}
};

const std::map<std::pair<std::string, std::string>, float> kConfusionMatrix = {
    {{"z", "zh"}, 0.1f}, {{"zh", "z"}, 0.1f},
    {{"c", "ch"}, 0.2f}, {{"ch", "c"}, 0.2f},
    {{"s", "sh"}, 0.2f}, {{"sh", "s"}, 0.2f},
    {{"l", "n"}, 0.2f},  {{"n", "l"}, 0.2f},
    {{"l", "r"}, 0.3f},  {{"r", "l"}, 0.3f},
    {{"f", "h"}, 0.6f},  {{"h", "f"}, 0.6f}, // 强拒绝
    {{"in", "ing"}, 0.1f}, {{"ing", "in"}, 0.1f},
    {{"en", "eng"}, 0.2f}, {{"eng", "en"}, 0.2f},
    {{"an", "ang"}, 0.2f}, {{"ang", "an"}, 0.2f},
    {{"b", "p"}, 0.4f}, {{"p", "b"}, 0.4f}, 
    {{"d", "t"}, 0.3f}, {{"t", "d"}, 0.3f}, 
    {{"g", "k"}, 0.3f}, {{"k", "g"}, 0.3f}, 
    {{"j", "q"}, 0.3f}, {{"q", "j"}, 0.3f}, 
    {{"x", "s"}, 0.4f}, {{"s", "x"}, 0.4f}, 
    {{"j", "z"}, 0.4f}, {{"z", "j"}, 0.4f}, 
    {{"m", "n"}, 0.3f}, {{"n", "m"}, 0.3f},
    {{"ia", "ie"}, 0.3f}, {{"ie", "ia"}, 0.3f}, 
    {{"uan", "uang"}, 0.3f}, {{"uang", "uan"}, 0.3f},
    {{"ei", "e"}, 0.2f},  {{"e", "ei"}, 0.2f},  
    {{"ai", "ei"}, 0.3f}, {{"ei", "ai"}, 0.3f}, 
    {{"ou", "u"}, 0.3f},  {{"u", "ou"}, 0.3f},  
    {{"u", "iu"}, 0.2f}, {{"iu", "u"}, 0.2f},  
    {{"w", "l"}, 0.2f},  {{"l", "w"}, 0.2f},    
    {{"g", "w"}, 0.3f},  {{"w", "g"}, 0.3f},    
    {{"l", "l"}, 0.0f},                         
    {{"f", "b"}, 0.3f}, {{"b", "f"}, 0.3f},     
    {{"s", "si"}, 0.1f}, 
    {{"q", "qi"}, 0.1f}, 
    {{"j", "jiu"}, 0.2f},
    {{"y", "yi"}, 0.1f}, 
    {{"l", "ling"}, 0.2f}
};

namespace Utils {
    int utf8_char_len(unsigned char c) {
        if ((c & 0x80) == 0) return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 1;
    }
    bool is_chinese(const std::string& char_bytes) {
        return (char_bytes.size() == 3 && (unsigned char)char_bytes[0] >= 0xE0);
    }
}

struct SplitPhoneme { 
    std::string initial; 
    std::string final; 
    int tone; 
};

class PinyinSplitter {
public:
    static inline const std::vector<std::string> INITIALS = {
        "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"
    };
    static SplitPhoneme split(const std::string& pinyin_with_tone) {
        SplitPhoneme p; p.tone = 5; 
        std::string raw_py = pinyin_with_tone;
        if (!raw_py.empty() && std::isdigit(raw_py.back())) {
            p.tone = raw_py.back() - '0';
            raw_py.pop_back();
        }
        if (raw_py.empty()) { p.initial = ""; p.final = ""; return p; }

        bool found = false;
        for (const auto& i : INITIALS) {
            if (raw_py.size() >= i.size() && raw_py.compare(0, i.size(), i) == 0) {
                if (i == "r" && raw_py == "er") continue;
                p.initial = i; p.final = raw_py.substr(i.size()); found = true; break;
            }
        }
        if (!found) { p.initial = ""; p.final = raw_py; }
        if ((p.initial=="j"||p.initial=="q"||p.initial=="x"||p.initial=="y") && (p.final=="v"||p.final=="yu")) p.final="u";
        if (p.initial == "y" && p.final == "ou") p.final = "ou";
        if (p.initial == "y" && p.final == "ao") p.final = "ao";
        return p;
    }
};

class PinyinProvider {
    static std::unique_ptr<::Pinyin::Pinyin> g_pinyin;
    static std::once_flag init_flag;
    static std::mutex _mutex;
public:
    static void initialize(const std::string& dictPath) {
        std::call_once(init_flag, [&]() {
            ::Pinyin::setDictionaryPath(dictPath);
            g_pinyin = std::make_unique<::Pinyin::Pinyin>();
        });
    }
    static size_t process_zh(const std::string& text, size_t pos, std::vector<Phoneme>& seq) {
        size_t scan_pos = pos; size_t len = text.length();
        while (scan_pos < len) {
            int char_len = Utils::utf8_char_len(text[scan_pos]);
            if (char_len + scan_pos > len) break;
            if (!Utils::is_chinese(text.substr(scan_pos, char_len))) break;
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
        for (const auto& item : res_tones) {
            SplitPhoneme sp = PinyinSplitter::split(item.pinyin);
            std::string toneless_val = item.pinyin;
            if (!toneless_val.empty() && std::isdigit(toneless_val.back())) toneless_val.pop_back();
            seq.push_back({toneless_val, item.hanzi, "", Lang::ZH, sp.initial, sp.final, sp.tone});
        }
        return scan_pos;
    }
};
std::unique_ptr<::Pinyin::Pinyin> PinyinProvider::g_pinyin = nullptr;
std::once_flag PinyinProvider::init_flag;
std::mutex PinyinProvider::_mutex;

size_t process_en_num(const std::string& text, size_t pos, std::vector<Phoneme>& seq) {
    size_t start_pos = pos;
    while (pos < text.length()) {
        char c = text[pos];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) break;
        pos++;
    }
    std::string token = text.substr(start_pos, pos - start_pos);
    for (char c : token) {
        Phoneme p; p.text = std::string(1, c); p.tone = 0;
        char lower_c = std::tolower(c);
        if (std::isdigit(c)) {
            p.lang = Lang::NUM;
            const char* map_py[] = {"ling","yi","er","san","si","wu","liu","qi","ba","jiu"};
            const int map_tone[] = {2, 1, 4, 1, 4, 3, 4, 1, 1, 3};
            int digit = c - '0';
            p.value = map_py[digit]; p.tone = map_tone[digit];
        } else {
            p.lang = Lang::EN;
            if (kStdAlphaPinyin.count(lower_c)) {
                p.value = kStdAlphaPinyin.at(lower_c);
            } else {
                p.value = std::string(1, lower_c);
            }
        }
        SplitPhoneme sp = PinyinSplitter::split(p.value);
        p.initial = sp.initial; p.final = sp.final;
        seq.push_back(p);
    }
    return pos;
}

std::vector<Phoneme> get_phoneme_info(const std::string& text) {
    std::vector<Phoneme> seq; 
    size_t pos = 0;
    std::string pending_space = ""; 

    while (pos < text.length()) {
        unsigned char c = static_cast<unsigned char>(text[pos]);

        // [修改 A] 遇到空格：放入缓存，跳过当前字符
        if (std::isspace(c)) { 
            pending_space += text.substr(pos, 1);
            pos++; 
            continue; 
        }

        // 记录处理前的 seq 大小，用于定位新生成的 Phoneme
        size_t old_size = seq.size();
        
        int char_len = Utils::utf8_char_len(c);
        
        // [保持原样] 调用原有的处理函数 (无需修改 process_zh/process_en_num)
        if (char_len == 3 && c >= 0xE0) {
            pos = PinyinProvider::process_zh(text, pos, seq);
        } else if (std::isalnum(c)) {
            pos = process_en_num(text, pos, seq);
        } else {
            Phoneme p; 
            p.text = text.substr(pos, char_len); 
            p.lang = Lang::OTHER; 
            p.tone = 0;
            seq.push_back(p); 
            pos += char_len;
        }

        // [修改 B] 属性吸附：如果产生了新节点，把刚才攒的空格贴给第一个新节点
        if (seq.size() > old_size) {
            seq[old_size].pre_padding = pending_space;
            pending_space = ""; // 清空缓存，防止重复粘贴
        }
    }
    
    return seq;
}

} // end anonymous namespace


PlateCorrector::PlateCorrector() {
    PinyinProvider::initialize(kDefaultDictPath);

    target_letters_.clear();
    for (const auto& pair : kStdAlphaPinyin) {
        target_letters_.insert(pair.second);
    }

    const std::vector<ProvinceTarget> kRawProvinceDB = {
        {"京", "jing", 1}, {"津", "jin", 1}, {"沪", "hu", 4}, {"豫", "yu", 4}, 
        {"冀", "ji", 4},   {"陕", "shan", 3}, {"蒙", "meng", 3}, {"辽", "liao", 2}, 
        {"黑", "hei", 1}, {"苏", "su", 1},   {"浙", "zhe", 4}, {"皖", "wan", 3}, 
        {"闽", "min", 3}, {"赣", "gan", 4},  {"鲁", "lu", 3},  {"鄂", "e", 4}, 
        {"湘", "xiang", 1}, {"粤", "yue", 4}, {"桂", "gui", 4}, {"琼", "qiong", 2}, 
        {"川", "chuan", 1}, {"云", "yun", 2}, {"藏", "zang", 4}, {"宁", "ning", 2}, 
        {"新", "xin", 1}, {"甘", "gan", 1}, {"吉", "ji", 2}
    };

    for (const auto& p : kRawProvinceDB) {
        prov_map_[p.pinyin].push_back(p);
    }

    target_digits_ = {
        "ling", "yi", "yao", "er", "liang", "san", "si", "wu", "liu", "qi", "ba", "jiu"
    };
}

PlateCorrector::~PlateCorrector() {}

void PlateCorrector::Initialize(const std::string& dict_path) {
    PinyinProvider::initialize(dict_path);
}

std::string PlateCorrector::CorrectText(const std::string& text) const {
    return EnhanceInternal(text);
}

float PlateCorrector::CalculateCost(const Phoneme& src, const std::string& target_py, int target_tone) const {

    if (kWhitelist.count(src.text)) {
        std::string target_str = kWhitelist.at(src.text);
        std::string digit_char = MapPinyinToChar(target_py, 2); 
        std::string letter_char = MapPinyinToChar(target_py, 1);
        if (digit_char == target_str) {
            return 0.0f; 
        } else if (letter_char == target_str) {
            return 0.0f;
        }
    }

    if (src.value == target_py) return 0.0f;

    float base_cost = 1.0f;

    if ((src.value == "yi" && target_py == "yao") || (src.value == "yao" && target_py == "yi")) base_cost = 0.0f;
    else if ((src.value == "er" && target_py == "liang") || (src.value == "liang" && target_py == "er")) base_cost = 0.0f;
    else if ((src.value == "ling" && target_py == "dong") || (src.value == "dong" && target_py == "ling")) base_cost = 0.0f;
    else if ((src.value == "qi" && target_py == "guai") || (src.value == "guai" && target_py == "qi")) base_cost = 0.0f;
    else if ((src.value == "da" && target_py == "wan") || (src.value == "wan" && target_py == "da")) base_cost = 0.0f;
    else if ((src.value == "xi" && target_py == "cha") || (src.value == "cha" && target_py == "xi")) base_cost = 0.0f;
    else if ((src.value == "bi" && target_py == "bo") || (src.value == "bo" && target_py == "bi")) base_cost = 0.0f;
    else {
        SplitPhoneme t_sp = PinyinSplitter::split(target_py);
        bool initial_exists = !src.initial.empty() && !t_sp.initial.empty();
        
        // case A: 声母完全相同
        if (initial_exists && src.initial == t_sp.initial) {
            if (src.final == t_sp.final) {
                base_cost = 0.0f;
            } else if (kConfusionMatrix.count({src.final, t_sp.final})) {
                base_cost = kConfusionMatrix.at({src.final, t_sp.final}); 
            } else {
                base_cost = 1.0f; 
            }
        }
        // case B: 声母不同，查询混淆矩阵
        else if (initial_exists && kConfusionMatrix.count({src.initial, t_sp.initial})) {
            if (src.final == t_sp.final) {
                base_cost = kConfusionMatrix.at({src.initial, t_sp.initial}); 
            } else if (kConfusionMatrix.count({src.final, t_sp.final})) {
                base_cost = 0.5f; 
            } else {
                base_cost = 0.6f; 
            }
        }
    }

    float tone_cost = 0.0f;
    if (base_cost < 0.6f && target_tone > 0 && src.tone > 0 && src.tone != 5) {
        if (src.tone != target_tone) {
            tone_cost = COST_TONE_MISMATCH;
        }
    }

    return std::min(1.0f, base_cost + tone_cost);
}

std::pair<float, std::string> PlateCorrector::BestMatch(const Phoneme& src, int type) const {
    float min_cost = 1.0f;
    std::string best_text = src.text;

    if (kWhitelist.count(src.text)) {
        std::string whitelist_target = kWhitelist.at(src.text); 
        
        bool is_type_match = false;
        
        if (type == 2) { 
            if (std::isdigit(whitelist_target[0])) is_type_match = true;
        } 
        else if (type == 1) { 
            if (std::isalpha(whitelist_target[0])) is_type_match = true;
        }

        if (is_type_match) {
            return {0.0f, src.text}; 
        }
    }


    const std::unordered_set<std::string>* targets;
    if (type == 0) {
         for (const auto& kv : prov_map_) {
            for (const auto& cand : kv.second) {
                float c = CalculateCost(src, cand.pinyin, cand.tone);
                if (c < min_cost) { min_cost = c; best_text = cand.text; }
            }
        }
        return {min_cost, best_text};
    } else {
        targets = (type == 1 ? &target_letters_ : &target_digits_);
    }

    std::string best_py = "";
    for (const auto& t_py : *targets) {
        float c = CalculateCost(src, t_py);
        if (c < min_cost) { 
            min_cost = c; 
            best_py = t_py; 
        }
    }

    if (min_cost < 0.6f) {
        std::string mapped_char = MapPinyinToChar(best_py, type);
        if (!mapped_char.empty()) {
            best_text = mapped_char;
        } else {
            best_text = src.text; 
            min_cost = 1.0f; 
        }
    }

    return {min_cost, best_text};
}

std::string PlateCorrector::MapPinyinToChar(const std::string& py, int type) const {
    if (type == 1) {
        for (const auto& pair : kStdAlphaPinyin) {
            if (pair.second == py) {
                return std::string(1, std::toupper(pair.first));
            }
        }


    } else if (type == 2) {
        if (py == "ling" || py == "dong") return "0";
        if (py == "yi" || py == "yao") return "1";
        if (py == "er" || py == "liang") return "2";
        if (py == "san") return "3";
        if (py == "si") return "4";
        if (py == "wu") return "5";
        if (py == "liu") return "6";
        if (py == "qi" || py == "guai") return "7";
        if (py == "ba") return "8";
        if (py == "jiu") return "9";
    }
    return "";
}

std::string PlateCorrector::EnhanceInternal(const std::string& input_text) const {
    std::vector<Phoneme> seq = get_phoneme_info(input_text);
    std::string result_text = "";
    int n = seq.size();
    int i = 0;

    while (i < n) {
        bool anchor_found = false;
        
        if (i + 1 < n) {
            auto m_prov = BestMatch(seq[i], 0);
            auto m_let = BestMatch(seq[i+1], 1);
            
            float s_prov = 1.0f - m_prov.first;
            float s_let = 1.0f - m_let.first;

            if (s_prov >= ANCHOR_MIN_PROV_SCORE && s_let >= ANCHOR_MIN_SCORE) {
                std::string plate_cand = seq[i].pre_padding + m_prov.second + 
                                         seq[i+1].pre_padding + m_let.second;

                float total_score = (s_prov * W_PROV) + (s_let * W_LETTER);
                float total_weight = W_PROV + W_LETTER;
                int valid_char_count = 2; 
                
                int k = i + 2;
                while (k < n && (k - i < 12)) {
                    auto m_dig = BestMatch(seq[k], 2);
                    auto m_alp = BestMatch(seq[k], 1);
                    
                    auto m_prov_check = BestMatch(seq[k], 0); 
                    if (1.0f - m_prov_check.first > ANCHOR_MIN_SCORE && (k-i) >= 6) {
                        break;
                    }
                    
                    float s_dig = 1.0f - m_dig.first;
                    float s_alp = 1.0f - m_alp.first;
                    float max_s = std::max(s_dig, s_alp);

                    if (max_s < 0.6f) {
                        break; 
                    }
                    
                    total_score += max_s * W_EXT;
                    total_weight += W_EXT;

                    std::string best_char;
                    
                    if (s_dig > s_alp) {
                        best_char = m_dig.second;
                    } else if (s_alp > s_dig) {
                        best_char = m_alp.second;
                    } 
                    else {
                        char raw_char = 0;
                        if (seq[k].text.length() == 1) {
                            raw_char = seq[k].text[0];
                        }

                        if (std::isalpha(raw_char)) {
                            best_char = m_alp.second;
                        } 
                        else if (std::isdigit(raw_char)) {
                            best_char = m_dig.second;
                        } else {
                            best_char = m_dig.second;
                        }
                    }

                    if (!best_char.empty() && std::isalpha(static_cast<unsigned char>(best_char[0]))) {
                        best_char += " ";
                    }

                    plate_cand += seq[k].pre_padding + best_char;
                    
                    valid_char_count++;
                    k++;
                }

                // 3. 提交判决
                if (valid_char_count >= 6 && (total_score / total_weight >= CONFIDENCE_THRESHOLD)) {
                    result_text += plate_cand;
                    i = k; 
                    anchor_found = true;
                }
            }
        }

        if (!anchor_found) {
            // [修改输出] 没有触发纠错，保留原文本和原空格
            result_text += seq[i].pre_padding + seq[i].text;
            i++;
        }
    }
    return result_text;
}