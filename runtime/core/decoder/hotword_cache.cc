#include "decoder/hotword_cache.h"
#include <cmath>
namespace wenet {

HotwordCache::HotwordCache(size_t capacity, int activate_threshold)
    : capacity_(capacity), activate_threshold_(activate_threshold) {}

    void HotwordCache::Touch(const std::string& word) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(word);
        if (it != cache_map_.end()) {
            // item exist
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            it->second->hit_count++;
        } else {
            // no item
            if (cache_list_.size() >= capacity_) {
                cache_map_.erase(cache_list_.back().word);
                cache_list_.pop_back();
            }
            cache_list_.push_front({word, 1});
            cache_map_[word] = cache_list_.begin();
        }
    }

    std::vector<std::string> HotwordCache::GetActiveHotwords() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> res;
        for (const auto& item : cache_list_) {
            if (item.hit_count >= activate_threshold_) {
                res.push_back(item.word);
            }
        }
        return res;
    }

    std::unordered_map<std::string, float> HotwordCache::GetActiveHotwordsWithBoost() const {
            std::unordered_map<std::string, float> res;
            
            int index = 0;
            for (const auto& item : cache_list_) {
                if (item.hit_count >= activate_threshold_) {
                    
                    float base_boost = 0.03f; 
                    float count_boost = std::log(item.hit_count) * 0.015f; 
                    float recency_boost = (1.0f - (float)index / std::max(1.0f, (float)cache_list_.size())) * 0.06f;
                    float total_boost = base_boost + count_boost + recency_boost;
                    
                    if (total_boost > 0.15f) total_boost = 0.15f;
                    res[item.word] = total_boost;
                }
                index++;
            }
            return res;
        }

} // namespace wenet
