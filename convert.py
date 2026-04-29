import codecs

def convert_to_utf8(input_file, output_file):
    try:
        with codecs.open(input_file, 'r', encoding='utf-16le') as f:
            content = f.read()
        with codecs.open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Failed to convert {input_file}: {e}")

convert_to_utf8('old_pipe.py', 'old_pipe_utf8.py')
convert_to_utf8('old_infer.py', 'old_infer_utf8.py')
