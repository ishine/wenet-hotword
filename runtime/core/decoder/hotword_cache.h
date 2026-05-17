#ifndef WENET_UTILS_HOTWORD_CACHE_H_
#define WENET_UTILS_HOTWORD_CACHE_H_

#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace wenet {

struct CacheItem {
    std::string word;
    int hit_count;
};

class HotwordCache {
public:
    explicit HotwordCache(size_t capacity = 20, int activate_threshold = 2);
    void Touch(const std::string& word);
    std::vector<std::string> GetActiveHotwords() const;
    std::unordered_map<std::string, float> GetActiveHotwordsWithBoost() const;
    size_t Size() const { return cache_list_.size(); }

private:
    size_t capacity_;
    int activate_threshold_;
    
    // LRU list
    std::list<CacheItem> cache_list_;
    // for quick query
    std::unordered_map<std::string, std::list<CacheItem>::iterator> cache_map_;
    mutable std::mutex mutex_;
};

} // namespace wenet
#endif // WENET_UTILS_HOTWORD_CACHE_H_
