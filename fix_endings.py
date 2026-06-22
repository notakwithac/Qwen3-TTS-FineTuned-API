import sys

def fix_line_endings(filenames):
    for filename in filenames:
        try:
            with open(filename, 'rb') as f:
                content = f.read()
            new_content = content.replace(b'\r\n', b'\n')
            if new_content != content:
                with open(filename, 'wb') as f:
                    f.write(new_content)
                print(f"Fixed line endings for: {filename}")
            else:
                print(f"No changes needed for: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    fix_line_endings(sys.argv[1:])
