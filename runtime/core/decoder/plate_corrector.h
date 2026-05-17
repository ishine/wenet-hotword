#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <memory>

// 前置声明，隐藏内部实现细节
struct Phoneme; 

// [新增] 用于存储带声调的省份信息
struct ProvinceTarget {
    std::string text;    // 汉字 "赣"
    std::string pinyin;  // 拼音 "gan"
    int tone;            // 声调 4
};

class PlateCorrector {
public:
    PlateCorrector();
    ~PlateCorrector(); 

    // 主要对外接口
    std::string CorrectText(const std::string& text) const;

    // 静态初始化 (指定字典路径)
    static void Initialize(const std::string& dict_path);

private:
    // 权重配置
    const float W_PROV = 10.0f;
    const float W_LETTER = 8.0f;
    const float W_EXT = 2.0f;
    const float ANCHOR_MIN_SCORE = 0.9f; 
    const float ANCHOR_MIN_PROV_SCORE = 0.9f; // 对省份要求更高一点
    const float CONFIDENCE_THRESHOLD = 0.85f;
    const float COST_TONE_MISMATCH = 0.2f;    // [新增] 声调不匹配的惩罚

    // 目标集合
    // [变更] 省份改为 Map 以支持声调查找: key=pinyin ("gan"), value=list({"赣",4}, {"甘",1})
    std::unordered_map<std::string, std::vector<ProvinceTarget>> prov_map_;
    std::unordered_set<std::string> target_letters_;
    std::unordered_set<std::string> target_digits_;

    // 核心算法
    // [变更] 增加 target_tone 参数
    float CalculateCost(const Phoneme& src, const std::string& target_py, int target_tone = 0) const;
    
    // 返回 pair<cost, text>
    std::pair<float, std::string> BestMatch(const Phoneme& src, int type) const;
    
    // 拼音转字符 (处理 "yi"->E, "cha"->X 等)
    std::string MapPinyinToChar(const std::string& py, int type) const;

    // 内部实现流程
    std::string EnhanceInternal(const std::string& input_text) const;
};