import os

def load_bbc_news_dataset(base_path='datos/prensa'):
    data = []
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if not os.path.isdir(category_path):
            continue
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read().strip()
                    if text:
                        data.append((text, category))
    return data
