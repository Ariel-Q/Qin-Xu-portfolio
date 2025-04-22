import pandas as pd

NEW_ATTRIBUTES = [
    "Male", "Young", "Attractive", "Blurry", "Oval_Face", "Chubby", "Double_Chin", "High_Cheekbones",
    "Rosy_Cheeks", "Pale_Skin", "Bald", "Receding_Hairline", "Black_Hair", "Blond_Hair", "Brown_Hair",
    "Gray_Hair", "Straight_Hair", "Wavy_Hair", "Bangs", "Wearing_Hat", "Arched_Eyebrows",
    "Bushy_Eyebrows", "Bags_Under_Eyes", "Narrow_Eyes", "Eyeglasses", "Big_Nose", "Pointy_Nose",
    "Big_Lips", "Mouth_Slightly_Open", "5_o_Clock_Shadow", "Mustache", "Goatee", "Sideburns",
    "No_Beard", "Heavy_Makeup", "Wearing_Lipstick", "Wearing_Earrings", "Wearing_Necklace",
    "Wearing_Necktie", "Smiling"
]

def convert_attr_txt_to_csv(txt_path, output_csv):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    header = lines[1].strip().split()
    selected_idx = [i for i, h in enumerate(header) if h in NEW_ATTRIBUTES]
    rows = []
    for line in lines[2:]:
        parts = line.strip().split()
        filename = parts[0]
        attrs = [(int(parts[i+1]) + 1) // 2 for i in selected_idx]
        rows.append([filename] + attrs)
    df = pd.DataFrame(rows, columns=['filename'] + NEW_ATTRIBUTES)
    df.to_csv(output_csv, index=False)
    print(f" Saved {output_csv}, shape = {df.shape}")

if __name__ == '__main__':
    convert_attr_txt_to_csv('data/list_attr_celeba.txt', 'data/celeba_multilabel.csv')
