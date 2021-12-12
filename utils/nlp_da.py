from sdk.baidu_translate import BaiduTranslate
import nlpcda
import random
import json


class NlpDA(object):
    rdw = None
    smw = None
    hmp = None
    rdc = None
    cpe = None
    eqc = None
    tra = None

    def __init__(self,
                 random_word_options: dict = None,
                 similar_word_options: dict = None,
                 homophone_options: dict = None,
                 random_delete_char_options: dict = None,
                 char_position_exchange_options: dict = None,
                 equivalent_char_options: dict = None,
                 translate_options: dict = None,
                 ner_labeled_entity_options: dict = None):
        """
        :param random_word_options: 随机实体替换参数
        :param similar_word_options: 随机同义词替换参数
        :param homophone_options: 随机近义近音词替换参数
        :param random_delete_char_options: 随机字删除参数
        :param char_position_exchange_options: 随机邻近字置换参数
        :param equivalent_char_options: 等价字替换
        :param translate_options: 回译参数
        :param ner_labeled_entity_options: 基于 NER 标注的实体替换参数
        :return:
        """
        if random_word_options:
            self.rdw = nlpcda.Randomword(**random_word_options)
        if similar_word_options:
            self.smw = nlpcda.Similarword(**similar_word_options)
        if homophone_options:
            self.hmp = nlpcda.Homophone(**homophone_options)
        if random_delete_char_options:
            self.rdc = nlpcda.RandomDeleteChar(**random_delete_char_options)
        if char_position_exchange_options:
            self.cpe = nlpcda.CharPositionExchange(**char_position_exchange_options)
        if equivalent_char_options:
            self.eqc = nlpcda.EquivalentChar(**equivalent_char_options)
        if translate_options:
            self.tra = BaiduTranslate(**translate_options)
        if ner_labeled_entity_options:
            self.nle = NerLabeledEntityDA(**ner_labeled_entity_options)

    def generate(self, data, extra=None):
        result = [data]

        # 随机实体替换
        if self.rdw:
            result += self.rdw.replace(data)

        # 随机同义词替换
        if self.smw:
            result += self.smw.replace(data)

        # 随机近义近音词替换
        if self.hmp:
            result += self.hmp.replace(data)

        # 随机字删除
        if self.rdc:
            result += self.rdc.replace(data)

        # 随机邻近字置换
        if self.cpe:
            result += self.cpe.replace(data)

        # 等价字替换
        if self.eqc:
            result += self.eqc.replace(data)

        # 回译
        if self.tra:
            result.append(self.tra.translate(data))

        # 基于 NER 标注的实体替换
        if self.nle:
            bio = extra.get("bio", "")
            result += self.nle.replace(data, bio)

        return list(set(filter(None, result)))


class NerLabeledEntityDA(object):
    def __init__(self, ner_labeled_entity_file_path: str, create_num: int, change_prop: float):
        self._load_entity(ner_labeled_entity_file_path)
        self.create_num = create_num
        self.change_prop = change_prop

    def _load_entity(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.entity = json.load(f)
            self.all_labels = self.entity.keys()

    def replace(self, data: str, bio: str):
        result = [data]
        if not bio:
            return result
        data, bio = list(data), bio.split()
        assert len(data) == len(bio), \
            "ner label len error, {} -> {}".format(''.join(data), ' '.join(bio))

        label_range = self._gen_label_range(data, bio)
        for label, idx_range in label_range.items():
            if label not in self.all_labels:
                continue
            da_data, idx = [], 0
            for start, end in idx_range:
                da_data += data[idx: start]
                if random.random() <= self.change_prop:
                    da_data += [random.choice(self.entity[label])]
                else:
                    da_data += data[start: end + 1]
                idx = end + 1
            da_data += data[idx:]
            result.append("".join(da_data))
        return result[:self.create_num]

    @staticmethod
    def _gen_label_range(data: str, bio: str):
        result = {}
        cur_label, start, end = "", -1, -1
        for i, (k, v) in enumerate(zip(data, bio)):
            if v == 'O':
                if cur_label:
                    result[cur_label].append((start, end))
                    cur_label, start, end = "", -1, -1
                continue
            prefix, label = v.split('-')
            if prefix == 'B':
                if cur_label:
                    result[cur_label].append((start, end))
                cur_label, start, end = label, i, i
                if cur_label not in result:
                    result[cur_label] = []
            elif prefix == 'I':
                end += 1
        return result


if __name__ == "__main__":
    test_str = "兴业银行今天用了我们本地的号码打我电话了0579，说今天下午6点还是没有结果把我拉进兴业银行黑名单，走法律流程，我欠兴业银行2万1..."

    nlp_da = NlpDA(
        random_word_options={"create_num": 2, "change_rate": 0.3},
        similar_word_options={"create_num": 2, "change_rate": 0.3},
        homophone_options={"create_num": 2, "change_rate": 0.3},
        random_delete_char_options={"create_num": 2, "change_rate": 0.1},
        char_position_exchange_options={"create_num": 2, "change_rate": 0.3, "char_gram": 1},
        equivalent_char_options={"create_num": 2, "change_rate": 0.3},
        translate_options={"domain": "finance", "trans_path": [("zh", "en"), ("en", "zh")]}
    )

    for text in nlp_da.generate(test_str):
        print(text)
