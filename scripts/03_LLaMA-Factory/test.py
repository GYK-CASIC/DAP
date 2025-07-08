# import json

# def count_json_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)  # 加载JSON文件内容
#         if isinstance(data, list):
#             # 如果JSON文件中的数据是列表，则统计列表中的元素数量
#             data_count = len(data)
#             print(f"JSON文件中数据的数量为: {data_count}")
#         else:
#             # 如果不是列表形式的数据，可以根据具体结构统计
#             print(f"JSON文件中的数据不是列表形式。")
#     return data_count

# # 示例用法
# file_path = 'data/dpo_en.json'  # 替换为你的JSON文件路径
# count_json_data(file_path)

import json

# 读取 JSON 文件
with open('data/dpo_en_demo.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 计算每条数据的单词数量
word_count_per_data = []

for entry in data:
    # 初始化三个来源的单词数
    human_word_count = 0
    system_word_count = 0
    gpt_word_count = 0

    # 累加每一轮对话中的单词数量
    for conversation in entry["conversations"]:
        if conversation["from"] == "human":
            human_word_count += len(conversation["value"].split())
        elif conversation["from"] == "system":
            system_word_count += len(conversation["value"].split())
        elif conversation["from"] == "gpt":
            gpt_word_count += len(conversation["value"].split())

    # 计算 chosen 和 rejected 的单词数量
    chosen_value = entry["chosen"]["value"]
    rejected_value = entry["rejected"]["value"]

    chosen_word_count = len(chosen_value.split())
    rejected_word_count = len(rejected_value.split())

    # 计算总单词数
    total_word_count = (human_word_count + system_word_count + gpt_word_count +
                        chosen_word_count + rejected_word_count)

    # 将每条数据的总单词数添加到列表中
    word_count_per_data.append((total_word_count, human_word_count, system_word_count, gpt_word_count, chosen_word_count, rejected_word_count))


# 计算相邻两条数据单词数量之和，并排序
word_count_sum_pairs = []

# 只计算相邻的数据对
for i in range(0, len(word_count_per_data) - 1, 2):
    word_count_sum = word_count_per_data[i][0] + word_count_per_data[i + 1][0]
    word_count_sum_pairs.append(((i, i + 1), word_count_sum))

# 按单词数量和从大到小排序
word_count_sum_pairs.sort(reverse=True, key=lambda x: x[1])

# 打印相邻两条数据的单词数量之和的排序
print("\n相邻两条数据的单词数量之和从大到小排序:")
for (idx1, idx2), word_sum in word_count_sum_pairs:
    print(f"数据 {idx1 + 1} 和 数据 {idx2 + 1}: 单词数量之和: {word_sum}")


# 按单词数量从大到小排序
word_count_per_data.sort(reverse=True, key=lambda x: x[0])

# 打印每条数据的单词数量排序
print("每条数据的单词数量从大到小排序:")
for idx, (total_count, human_count, system_count, gpt_count, chosen_count, rejected_count) in enumerate(word_count_per_data):
    print(f"数据 {idx + 1}: 总单词数: {total_count}, human: {human_count}, system: {system_count}, gpt: {gpt_count}, chosen: {chosen_count}, rejected: {rejected_count}")



