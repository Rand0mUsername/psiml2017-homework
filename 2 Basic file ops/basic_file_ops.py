import re
import os

# regex that matches valid text files
FILE_PATTERN = re.compile(r"^PSIML_(\d{3}).txt$")

def count_files(root):
    """Return the number of files under root that satisfy the condition."""
    num_files = 0
    for dirpath, _, files in os.walk(root):
        for file in files:
            fmatch = FILE_PATTERN.match(file)
            assert fmatch
            fh = open(os.path.join(dirpath, file), 'r')
            text = fh.read()
            fh.close()
            # compare the number of occurrences with file name
            if text.count("PSIML") == int(fmatch.group(1)):
                num_files += 1
    return num_files

if __name__ == "__main__":
    root = raw_input()
    print(count_files(root))
