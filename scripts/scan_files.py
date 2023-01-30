import os


def max_lines_unique_numbers(folder_path):
    max_lines = 0
    max_unique_numbers = 0
    max_file = ""

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path) as f:
                lines = f.readlines()
                unique_numbers = set()
                for line in lines:
                    if len(line.split()) > 0:
                        first_word = line.split()[0]
                        if first_word.isdigit():
                            unique_numbers.add(first_word)
                if len(lines) > max_lines or (len(lines) == max_lines and len(unique_numbers) > max_unique_numbers):
                    max_lines = len(lines)
                    max_unique_numbers = len(unique_numbers)
                    max_file = filename

    return max_file


folder_path = r"D:\Downloads\test"
print(max_lines_unique_numbers(folder_path))
