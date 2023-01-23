#from https://www.shedloadofcode.com/blog/how-to-batch-rename-files-in-folders-with-python
import os

def rename_files(path):
    #replacements = ["_dualforecast", "_narrative", "_pf1", "_summary", "_txn"]
    #search_terms = ["CLAIM", "NARRATIVE", "PF1", "SUMMARY", "Txn"]
    
    replacements = ["1", "2", "3", "4"]
    search_terms = ["baba", "daba", "caba", "lava"]
    count = 0
    
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        name, extension = os.path.splitext(filename)

        for i, term in enumerate(search_terms):
            if term in name:
                prefix = name[:15]
                postfix = replacements[i]
                new_name = os.path.join(path, prefix + postfix + extension)
                os.rename(file_path, new_name)
                continue

        count += 1
    
    print(f"{count} files in folder {path} were renamed.")


if __name__ == "__main__":
    rename_files(r"C:\\Users\\shedloadofcode\\Documents\\TestFolder")

