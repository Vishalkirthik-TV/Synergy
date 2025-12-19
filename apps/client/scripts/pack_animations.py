
import os
import re
import json

SIGN_KIT_PATH = r"c:\Users\vkirt\Desktop\HACKS Projects\PS - 04\Sign-Kit-An-Avatar-based-ISL-Toolkit\client\src\Animations"
OUTPUT_PATH = r"c:\Users\vkirt\Desktop\HACKS Projects\PS - 04\TalkMateAI\apps\client\public\animations.json"

def parse_val(v):
    v = v.strip()
    # Handle Math.PI
    if "Math.PI" in v:
        v = v.replace("Math.PI", "3.14159265359")
        try:
            return eval(v)
        except:
            return 0
    try:
        return float(v)
    except:
        return v.replace('"', '').replace("'", "")

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    frames = []
    current_frame = []
    
    # Split by ref.animations.push(animations) which marks end of a frame block
    # But usually the variables are built up and pushed.
    # We can split by lines and track "animations = []" or "push".
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Check for frame push
        if "ref.animations.push(animations)" in line:
            if current_frame:
                frames.append(current_frame)
                current_frame = []
            continue
            
        # Check for bone move
        # animations.push(["mixamorigLeftHandIndex1", "rotation", "y", -Math.PI/9, "-"])
        match = re.search(r'animations\.push\(\[(.*?)\]\)', line)
        if match:
            args_str = match.group(1)
            # Split by comma but respect basic math? split(",") is unsafe if math has comma?
            # JS usually no comma in math.
            args = [a.strip() for a in args_str.split(',')]
            if len(args) >= 5:
                # [bone, type, axis, val, dir]
                bone = args[0].replace('"', '').replace("'", "")
                prop = args[1].replace('"', '').replace("'", "")
                axis = args[2].replace('"', '').replace("'", "")
                val = parse_val(args[3])
                op = args[4].replace('"', '').replace("'", "")
                current_frame.append([bone, prop, axis, val, op])

    return frames

def main():
    data = {}
    
    # Process Alphabets
    alpha_path = os.path.join(SIGN_KIT_PATH, "Alphabets")
    if os.path.exists(alpha_path):
        for fname in os.listdir(alpha_path):
            if fname.endswith(".js"):
                key = fname.replace(".js", "").upper() # A, B, C
                frames = process_file(os.path.join(alpha_path, fname))
                data[key] = frames
                print(f"Processed Alphabet: {key}, Frames: {len(frames)}")

    # Process Words
    words_path = os.path.join(SIGN_KIT_PATH, "Words")
    if os.path.exists(words_path):
        for fname in os.listdir(words_path):
            if fname.endswith(".js"):
                key = fname.replace(".js", "").upper() # HOME, YOU
                frames = process_file(os.path.join(words_path, fname))
                data[key] = frames
                print(f"Processed Word: {key}, Frames: {len(frames)}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(data, f)
    
    print(f"Saved {len(data)} animations to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
