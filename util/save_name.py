import re
import os


def get_save_name(base_name):
    output_path = './output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pattern = re.compile(f'^{re.escape(base_name)}\\((\\d+)\\)$')
    existing_nums = []

    for entry in os.listdir(output_path):
        entry_path = os.path.join(output_path, entry)
        if os.path.isdir(entry_path):
            match = pattern.match(entry)
            if match:
                num = int(match.group(1))
                existing_nums.append(num)

    if not existing_nums:
        return f'{base_name}(1)'
    else:
        max_num = max(existing_nums)
        return f'{base_name}({max_num + 1})'








