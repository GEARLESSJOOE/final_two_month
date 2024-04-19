import requests
from bs4 import BeautifulSoup
import re, json
from tqdm import tqdm
import traceback
import os

"""
定义爬虫代码，爬取湖北省电子招标交易平台中的中标信息，其中包括以下几个方面
1. 招标类型：房建市政、水利工程、交通工程、铁路工程
2. 招标项目名称
3. 招标项目详细url地址
4. 招标人
5. 招标项目介绍
6. 中标公司
7. 中标价格 
用于构建知识图谱
"""


# 定义爬虫类
class Crawler():
    def __init__(self):
        # 定义网站header，但是该爬取网站没有反爬策略，可以不定义
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }

    # 获取招标一级页面列表
    def get_tender_lst(self, url):
        res = []
        response = requests.get(url=url, headers=self.headers)  # 请求网址
        soup = BeautifulSoup(response.text, "html.parser")  # 利用html.parser将网址的text内容进行解析
        tender_span = soup.select(".clearfix span")  # 网站寻址，返回为列表形式
        # 获取招标类型
        tender_type = tender_span[0].text
        node_a = soup.select(".wb-data-item .wb-data-list .wb-data-infor a")
        # 获取招标url地址，并以元组形式组合，以列表形式返回
        for a in node_a:
            tender_project_name = a.text
            tender_project_url = "https://www.hbbidcloud.cn" + a.attrs["href"]
            res.append((tender_type, tender_project_name, tender_project_url))
        return res

    # 解析中标公司名称方法
    def parse_winning_company(self, info):
        obj = re.search("<td>(.*?)</td>", info)  # 利用正则匹配目标内容
        return obj.group(1)

    # 解析中标公司价格方法
    def parse_winning_price(self, info):
        obj = re.search("<td>([\d.]+)</td>", info)
        return obj.group(1)

    # 解析招标人方法
    def parse_tenderee(self, info):
        obj = re.search('<td class="tdleft">  招标人：(.*?)</td>', info)
        return obj.group(1)

    # 获取招标二级页面的详细信息
    def get_tender_detail(self, tender_detail: tuple):
        try:
            type, name, url = tender_detail
            tender = {}
            tender["tender_type"] = type
            tender["tender_name"] = name
            tender["tender_url"] = url
            # 获取页面源码
            response = requests.get(url=url, headers=self.headers)
            soup = BeautifulSoup(response.text, "html.parser")
            info = str(soup.select('.ewb-article-info'))
            # 小坑：网站上的特殊符号，直接匹配的话会报错，所以加载到本地，看解析信息，再进行匹配
            # with open("info.html", "w") as file:
            #     file.write(info)
            tender["winning_company"] = self.parse_winning_company(info)
            tender["winning_price"] = self.parse_winning_price(info)
            tender["tenderee"] = self.parse_tenderee(info)
            tender_intro = soup.select(".tdleft")[0].text
            tender["tender_intro"] = re.sub("\s{2,}", "", tender_intro)

            # 以json形式保存
            with open(os.path.join(os.path.dirname(__file__), "data/crawler/tender_data.json"), "a") as file:
                json_str = json.dumps(tender, ensure_ascii=False)  # 定义json形式的文件输入流
                file.write(json_str + "\n")  # 写入文件

        # 捕获异常，防止中途报错
        except Exception as e:
            print(traceback.print_exc())


if __name__ == '__main__':
    crawler = Crawler()
    for i in range(1, 7):
        for j in range(2, 7):
            url = f"https://www.hbbidcloud.cn/shengbenji/jyxx/004005/00400500{i}/{j}.html"
            tender_lst = crawler.get_tender_lst(url)
            for tender_detail in tqdm(tender_lst):
                crawler.get_tender_detail(tender_detail)
