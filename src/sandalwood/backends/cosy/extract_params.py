
import re


def extract_params(filepath):
    params = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Matches PARAMETER(NAME=VALUE,...)
    # This is a bit simplistic and might need adjustment for multi-line PARAMETER statements
    matches = re.findall(r'PARAMETER\s*\((.*?)\)', content, re.DOTALL)
    
    for match in matches:
        # Remove newlines and spaces
        clean_match = match.replace('\n', '').replace(' ', '')
        # Split by comma
        items = clean_match.split(',')
        for item in items:
            if '=' in item:
                key, value = item.split('=')
                params[key.strip()] = value.strip()
            
    return params

params = extract_params('/home/mls/work/sandalwood/src/sandalwood/backends/cosy/cosy_src/dafox.f')
print(f"LVAR={params.get('LVAR')}")
print(f"LMEM={params.get('LMEM')}")
print(f"LDIM={params.get('LDIM')}")
print(f"LNO={params.get('LNO')}")
print(f"LNV={params.get('LNV')}")
print(f"LEA={params.get('LEA')}")
