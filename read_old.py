import codecs

def find_lines(filename, query, context=15):
    try:
        with codecs.open(filename, 'r', encoding='utf-16le') as f:
            lines = f.readlines()
    except Exception:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
    for i, line in enumerate(lines):
        if query in line:
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            print(f"Match found in {filename} around line {i}:")
            for j in range(start, end):
                print(f"{j}: {lines[j].rstrip()}")
            print("-" * 40)

find_lines('old_pipe.py', 'latents = self.scheduler.step')
find_lines('old_infer.py', 'paste')
