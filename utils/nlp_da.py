import time
import nlpcda


class NlpDA(object):
    rdw = None
    smw = None
    hmp = None
    rdc = None
    cpe = None
    eqc = None

    BAIDU_APP_ID = "20181215000248582"
    BAIDU_TRANS_SECRET_KEY = "h1QgNKc7SRDqwA3gVNpC"

    def __init__(self,
                 random_word_options: dict = None,
                 similar_word_options: dict = None,
                 homophone_options: dict = None,
                 random_delete_char_options: dict = None,
                 char_position_exchange_options: dict = None,
                 equivalent_char_options: dict = None):
        """
        :param random_word_options: 随机实体替换参数
        :param similar_word_options: 随机同义词替换参数
        :param homophone_options: 随机近义近音词替换参数
        :param random_delete_char_options: 随机字删除参数
        :param char_position_exchange_options 随机邻近字置换参数
        :param equivalent_char_options 等价字替换
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

    def generate(self, data):
        result = []

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

        # 英汉互译
        trs_path = [('zh', 'en'), ('en', 'zh')]
        trs_data = data
        for path in trs_path:
            t_from, t_to = path
            trs_data = nlpcda.baidu_translate(
                content=trs_data,
                appid=self.BAIDU_APP_ID,
                secretKey=self.BAIDU_TRANS_SECRET_KEY,
                t_from=t_from,
                t_to=t_to
            )
            time.sleep(1)
        result.append(trs_data)

        return list(set(result))


if __name__ == "__main__":
    test_str = '''兴业银行今天用了我们本地的号码打我电话了0579，说今天下午6点还是没有结果把我拉进兴业银行黑名单，走法律流程，我欠兴业银行2万1...'''

    nlp_da = NlpDA(
        random_word_options={"base_file": "../resource/entity/bank.txt", "create_num": 3, "change_rate": 0.3},
        similar_word_options={"create_num": 3, "change_rate": 0.3},
        homophone_options={"create_num": 3, "change_rate": 0.3},
        random_delete_char_options={"create_num": 3, "change_rate": 0.3},
        char_position_exchange_options={"create_num": 3, "change_rate": 0.3, "char_gram": 2},
        equivalent_char_options={"create_num": 3, "change_rate": 0.3}
    )

    for text in nlp_da.generate(test_str):
        print(text)
