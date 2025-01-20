import pandas as pd

# 读取三个预测文件
pred1 = pd.read_csv('predictions_attention.txt')
pred2 = pd.read_csv('predictions_concat.txt')
pred3 = pd.read_csv('predictions_weighted.txt')

# 合并三个数据框
merged = pred1.merge(pred2, on='guid', suffixes=('_attention', '_concat'))
merged = merged.merge(pred3, on='guid')
merged.rename(columns={'tag': 'tag_weighted'}, inplace=True)

# 找到预测不一致的guid
inconsistent = merged[(merged['tag_attention'] != merged['tag_concat']) |
                      (merged['tag_attention'] != merged['tag_weighted']) |
                      (merged['tag_concat'] != merged['tag_weighted'])]

# 输出不一致的guid及其预测结果
for index, row in inconsistent.iterrows():
    print(
        f"guid: {row['guid']}, attention: {row['tag_attention']}, concat: {row['tag_concat']}, weighted: {row['tag_weighted']}")


# 定义多数投票函数
def majority_vote(row):
    votes = [row['tag_attention'], row['tag_concat'], row['tag_weighted']]
    if votes.count(votes[0]) == 1 and votes.count(votes[1]) == 1:
        return 'neutral'
    return max(set(votes), key=votes.count)


# 应用多数投票函数
merged['tag'] = merged.apply(majority_vote, axis=1)

# 保存结果到predictions.txt
merged[['guid', 'tag']].to_csv('predictions.txt', index=False, header=True)
