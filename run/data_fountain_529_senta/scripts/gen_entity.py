import json

from utils.config_utils import get_config, CONFIG_PATH, DATA_PATH
import pandas as pd
import os


def run():
    config = get_config(os.path.join(CONFIG_PATH, "data_fountain_529_senta.json"), "train")
    train_data_path = os.path.join(DATA_PATH, config.exp_name, config.train_filename)

    df = pd.read_csv(train_data_path, encoding="utf-8")

    all_labels = ['BANK', 'PRODUCT']
    result = {}
    for label in all_labels:
        result[label] = set()
    for idx, line in df.iterrows():
        text, bio = line.get("text", ""), line.get("BIO_anno", "")
        text, bio = list(text), bio.split()
        assert len(text) == len(bio), "ner label len error"

        cur_label, cur_name = "", ""
        for i, (k, v) in enumerate(zip(text, bio)):
            if v == 'O':
                if cur_label and cur_name:
                    result[cur_label].add(cur_name)
                    cur_label, cur_name = "", ""
                continue
            prefix, label = v.split('-')
            if label not in all_labels:
                continue
            if prefix == 'B':
                if cur_label and cur_name:
                    result[cur_label].add(cur_name)
                cur_label, cur_name = label, k
            elif prefix == 'I':
                cur_name += k

    with open("entity.json", "w", encoding="utf-8") as f:
        for k, v in result.items():
            result[k] = list(v)
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    run()
