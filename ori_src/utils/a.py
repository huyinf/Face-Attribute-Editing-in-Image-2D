import pandas as pd
from io import StringIO

# Set the display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Set the display option to expand the frame to multiple lines
pd.set_option('display.expand_frame_repr', False)

facial_attributes = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
    'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
    'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

details = {
    # '000023.jpg': [1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1,
    #                1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1],
    # '000025.jpg': [1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1,
    #                -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
    '000041.jpg': [1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1],
    '000053.jpg': [1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    '000054.jpg': [-1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1,
                   -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1],
    '000060.jpg': [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
    '000121.jpg': [-1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1,
                   1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1],
    '000209.jpg': [-1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1,
                   1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1],
    '000250.jpg': [-1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1,
                   1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
    '000316.jpg': [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1,
                   -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
    '000881.jpg': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    # '000916.jpg': [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1,
    #                -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
    # '002049.jpg': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1,
    #                -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1]
}

selected_facial_attributes = [
    "Bald", "Bangs", "Blond_Hair", "Eyeglasses", "Mustache", "Smiling", "Male"
]

df = pd.DataFrame.from_dict(details).transpose()
df.columns = facial_attributes

selected_df = df[selected_facial_attributes]
print(selected_df.info)
