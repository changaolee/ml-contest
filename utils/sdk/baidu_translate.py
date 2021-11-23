import http.client
import hashlib
import time
import urllib
import random
import json
import os


class BaiduTranslate(object):
    BAIDU_APP_ID = ""
    BAIDU_TRANS_SECRET_KEY = ""

    def __init__(self, domain, trans_path, trans_cache_file=None, app_id=None, secret_key=None):
        self.domain = domain
        self.app_id = app_id or self.BAIDU_APP_ID
        self.secret_key = secret_key or self.BAIDU_TRANS_SECRET_KEY
        self.trans_path = trans_path
        self.trans_cache_file = trans_cache_file
        self.trans_cache = {}
        if os.path.isfile(trans_cache_file):
            with open(trans_cache_file, "r") as f:
                self.trans_cache = json.load(f)
        self.client = http.client.HTTPConnection('api.fanyi.baidu.com')

    def translate(self, text):
        if text in self.trans_cache and self.trans_cache[text] != text:
            return self.trans_cache[text]

        raw_text = text
        try:
            for trans in self.trans_path:
                from_lang, to_lang = trans
                salt = str(random.randint(32768, 65536))
                sign = self.app_id + text + salt + self.domain + self.secret_key
                sign = hashlib.md5(sign.encode("utf-8")).hexdigest()
                url = '/api/trans/vip/fieldtranslate?appid={}&q={}&from={}&to={}&salt={}&domain={}&sign={}'.format(
                    self.app_id, urllib.parse.quote(text), from_lang, to_lang, salt, self.domain, sign
                )
                self.client.request('GET', url)

                response = self.client.getresponse()
                result_all = response.read().decode("utf-8")
                result = json.loads(result_all)

                text = result['trans_result'][0]['dst']

                time.sleep(0.1)
        except Exception as e:
            print(e)
        finally:
            if text != raw_text:
                self.trans_cache[raw_text] = text
                with open(self.trans_cache_file, "w") as f:
                    json.dump(self.trans_cache, f, indent=4, ensure_ascii=False)
        return text
